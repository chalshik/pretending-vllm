"""The model runner. The only thing below the simulation boundary that the engine calls.

Upstream: vllm/v1/worker/gpu/model_runner.py
Tier: B

D6/F1: mirrors the **V2** runner, which is upstream's default at the pin. Its method
decomposition is kept because it *is* the interface -- `add_requests`,
`update_requests`, `finish_requests`, `free_states`, `prepare_inputs`, `prepare_attn`,
`execute_model`, `sample_tokens`. A monolithic `execute_model` would work but would
make a diff against upstream useless, which is the point of G2.

R8.1 fixes the order inside `execute_model`: update the persistent batch, build
attention metadata, resolve which slots are read and written, ask the cost model for a
duration, advance the clock, invoke `SimModel`, sample, return.

Everything here except the last two steps is real. The persistent-batch update
(R7.3), the input preparation, the block-table indexing, and the slot mapping (R8.3)
are the same computation upstream performs; only the forward pass is replaced.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.sim.cost_model import StepCost, StepProfile
from pvllm.sim.device import SimDevice
from pvllm.sim.model import SimModel
from pvllm.v1.attention.backends.sim_attn import SimAttentionMetadata
from pvllm.v1.core.sched.output import SchedulerOutput
from pvllm.v1.kv_cache_interface import KVCacheConfig
from pvllm.v1.outputs import LogprobsLists, ModelRunnerOutput
from pvllm.v1.worker.gpu.attn_utils import build_attn_metadata
from pvllm.v1.worker.gpu.block_table import BlockTables
from pvllm.v1.worker.gpu.input_batch import InputBatch, sort_batch_req_ids
from pvllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)

#: R8.4. Batch sizes a real runner would capture graphs for. A step whose batch
#: matches one of these, with no chunked prefill, pays the lower launch cost.
DEFAULT_CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)


class SimModelRunner:
    """Turns a `SchedulerOutput` into a `ModelRunnerOutput`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: SimDevice,
        sim_model: SimModel,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.sim_model = sim_model

        model_config = vllm_config.model_config
        assert vllm_config.scheduler_config is not None
        scheduler_config = vllm_config.scheduler_config

        self.max_num_reqs = scheduler_config.max_num_seqs
        self.max_model_len = scheduler_config.max_model_len
        assert scheduler_config.max_num_batched_tokens is not None
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        self.block_size = vllm_config.cache_config.block_size
        self.enforce_eager = model_config.enforce_eager
        # R14. Zero without speculation, which is what keeps the draft path off the
        # hot loop for the common case.
        self.num_spec_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None
            else 0
        )
        # A verified step's query is 1 + the drafts it carried.
        self.decode_query_len = 1 + self.num_spec_tokens

        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            vocab_size=model_config.get_vocab_size(),
        )
        self.block_tables: BlockTables | None = None
        self.kv_cache_config: KVCacheConfig | None = None
        self.captured_sizes: frozenset[int] = frozenset()

        #: R2.2. Pooling requests in flight: `req_id -> (dimensions, prompt tokens)`.
        #: Empty for a generation-only deployment, which is the common case.
        self.pooling_requests: dict[str, tuple[int, list[int]]] = {}

        #: R18.1. Encoder outputs resident on this device, by mm hash. The scheduler
        #: decides *whether* to encode and how much room there is; this is what is
        #: actually being held, and the two must agree -- which is the only way the
        #: eviction notice in `SchedulerOutput.free_encoder_mm_hashes` means
        #: anything.
        self.encoder_outputs: set[str] = set()

        #: The most recent step's metadata and cost, for the debug surface (D9).
        self.last_attn_metadata: SimAttentionMetadata | None = None

    # --- setup ---------------------------------------------------------------

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Build the block tables once the KV layout is known."""
        self.kv_cache_config = kv_cache_config
        block_sizes = [
            group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
        ]
        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            enable_caching=self.vllm_config.cache_config.enable_prefix_caching,
        )

    def capture_model(self) -> float:
        """Simulate graph capture. R8.4. Returns the modeled duration."""
        if self.enforce_eager:
            self.captured_sizes = frozenset()
            return 0.0
        self.captured_sizes = frozenset(
            size for size in DEFAULT_CAPTURE_SIZES if size <= self.max_num_reqs
        )
        return self.device.capture_graphs(len(self.captured_sizes))

    # --- persistent batch maintenance (R7.3) ---------------------------------

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        """Take slots for requests the worker has not seen before.

        D6: resumed requests arrive here too, not through `update_requests`. The V2
        runner discards a preempted request's state, so on resume it must be rebuilt
        from scratch -- patching state that was thrown away would silently produce a
        request whose token history is missing its middle.
        """
        assert self.block_tables is not None
        for new_req in scheduler_output.scheduled_new_reqs:
            prompt_token_ids = new_req.prompt_token_ids or []
            max_tokens = (
                new_req.sampling_params.max_tokens
                if new_req.sampling_params is not None
                else 0
            ) or 0
            # Prompt plus anything already generated. A *resumed* request has
            # output tokens the worker never saw, and rebuilding it from the prompt
            # alone puts its output position back at zero -- so the model
            # re-samples position 0 and the client gets a duplicate token (R5.5).
            prefill_token_ids = new_req.prefill_token_ids or prompt_token_ids
            req_idx = self.req_states.add_request(
                req_id=new_req.req_id,
                prompt_len=len(prompt_token_ids),
                all_token_ids=list(prefill_token_ids),
                num_computed_tokens=new_req.num_computed_tokens,
                max_tokens=max_tokens,
            )
            # A resumed request may already hold blocks, so install rather than
            # append.
            self.block_tables.set_block_ids(req_idx, new_req.block_ids)

            # R15. The constraint crosses the boundary inside `sampling_params`,
            # exactly as it does upstream -- no extra channel, and nothing above the
            # boundary needs to know what the simulated model does with it.
            self._maybe_set_constraint(new_req.req_id, new_req.sampling_params)
            # R18.1. Kept so the cost model can price this step's encoder work; the
            # scheduler sends input *ids*, and their sizes live on the request.
            if new_req.mm_features:
                self.req_states.mm_features[new_req.req_id] = list(new_req.mm_features)
            # R2.2. A pooling request never samples; it produces one vector on the
            # step its prefill completes. Kept by id because that is the only thing
            # `sample_tokens` gets back from the batch.
            if new_req.pooling_params is not None:
                self.pooling_requests[new_req.req_id] = (
                    new_req.pooling_params.dimensions
                    or self.sim_model.model.hidden_size,
                    list(prompt_token_ids),
                )

    def _maybe_set_constraint(self, req_id: str, sampling_params: Any) -> None:
        """Hand a constrained request's grammar to the model. R15."""
        params = getattr(sampling_params, "structured_outputs", None)
        if params is None or self.sim_model.constrained_plan(req_id) is not None:
            return
        from pvllm.v1.structured_output.request import get_structured_output_key

        kind, spec = get_structured_output_key(params)
        self.sim_model.set_constraint(req_id, kind.name.lower(), spec)

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        """Patch the state of requests the worker already holds. R7.3.

        The incremental path: usually one new token and no new block, which is why
        the scheduler sends a diff rather than a snapshot.
        """
        assert self.block_tables is not None
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is None:
                raise KeyError(
                    f"the scheduler sent request {req_id!r} as cached, but the worker "
                    f"holds no state for it. Its state was dropped without the "
                    f"scheduler being told, or it was resumed without being sent as "
                    f"new (R5.8)."
                )
            self.req_states.append_tokens(req_idx, cached.new_token_ids[i])
            self.req_states.set_num_computed_tokens(
                req_idx, cached.num_computed_tokens[i]
            )
            # R14. Taken from the scheduler rather than inferred from the tokens
            # just appended. Under speculation the two differ: a step schedules
            # `1 + num_drafts` tokens but only the *accepted* ones exist in the
            # request's history, so counting appends would leave the worker's idea
            # of the output position behind the scheduler's -- and the model would
            # re-sample positions it had already emitted.
            self.req_states.num_output_tokens[req_idx] = cached.num_output_tokens[i]
            new_blocks = cached.new_block_ids[i]
            if new_blocks is not None:
                self.block_tables.append_block_ids(req_idx, new_blocks)

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        """Drop state for requests the scheduler says are done. R5.8."""
        assert self.block_tables is not None
        for req_id in scheduler_output.finished_req_ids:
            # R15/R18. Per-request simulator state, dropped with the request. These
            # are keyed by request id and would otherwise grow for the life of the
            # process -- and hand a reused id the previous request's answer.
            self.sim_model.forget_request(req_id)
            self.req_states.mm_features.pop(req_id, None)
            self.pooling_requests.pop(req_id, None)
            req_idx = self.req_states.remove_request(req_id)
            if req_idx is not None:
                self.block_tables.clear(req_idx)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        """Drop state for preempted requests.

        Separate from `finish_requests` because a preempted request is coming back:
        its slot is released but its output tokens live on in the scheduler, and it
        will re-enter through `add_requests`.
        """
        assert self.block_tables is not None
        for req_id in scheduler_output.preempted_req_ids or ():
            req_idx = self.req_states.remove_request(req_id)
            if req_idx is not None:
                self.block_tables.clear(req_idx)

    # --- input preparation ---------------------------------------------------

    def prepare_inputs(self, scheduler_output: SchedulerOutput) -> InputBatch:
        """Flatten the step's work into batch-ordered arrays.

        A near-verbatim port of upstream's numpy half (F10).
        """
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)
        num_reqs = len(req_ids)

        num_scheduled_tokens = np.fromiter(
            (num_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        idx_mapping_np = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )

        query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        num_tokens = int(query_start_loc_np[-1])

        num_computed = self.req_states.num_computed_tokens[idx_mapping_np]
        seq_lens_np = (num_computed + num_scheduled_tokens).astype(np.int32)

        prefill_len_np = self.req_states.prefill_len[idx_mapping_np]
        num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[
            idx_mapping_np
        ]

        input_ids = np.zeros(num_tokens, dtype=np.int32)
        positions = np.zeros(num_tokens, dtype=np.int64)
        for batch_idx, req_idx in enumerate(idx_mapping_np):
            start = int(query_start_loc_np[batch_idx])
            end = int(query_start_loc_np[batch_idx + 1])
            first_position = int(num_computed[batch_idx])
            positions[start:end] = np.arange(
                first_position, first_position + (end - start)
            )

            tokens = self.req_states.all_token_ids[int(req_idx)]
            for offset, position in enumerate(
                range(first_position, first_position + (end - start))
            ):
                # A position past the known tokens is a slot being reserved ahead of
                # the token that will fill it; zero is the placeholder, as upstream.
                input_ids[start + offset] = (
                    tokens[position] if position < len(tokens) else 0
                )

        # A request is still prefilling if this step does not finish its prompt, and
        # only a request that has finished prefilling samples a token.
        is_prefilling_np = (
            num_computed_prefill_tokens_np + num_scheduled_tokens
        ) < prefill_len_np
        logits_indices = (query_start_loc_np[1 : num_reqs + 1] - 1)[~is_prefilling_np]

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping_np=idx_mapping_np,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            query_start_loc_np=query_start_loc_np,
            seq_lens_np=seq_lens_np,
            input_ids=input_ids,
            positions=positions,
            prefill_len_np=prefill_len_np.astype(np.int32),
            num_computed_prefill_tokens_np=num_computed_prefill_tokens_np.astype(
                np.int32
            ),
            is_prefilling_np=is_prefilling_np,
            logits_indices=logits_indices.astype(np.int32),
        )

    def prepare_attn(
        self, input_batch: InputBatch, scheduler_output: SchedulerOutput
    ) -> SimAttentionMetadata:
        """Build the attention metadata. R8.2, and R8.3's validation runs here.

        One set per KV cache group (R6.7). Each group has its own block table, so
        each has its own slot mapping -- and R8.3's oracle is only an oracle if it
        runs over all of them. Building group 0's alone would leave a hybrid model's
        windowed groups unchecked, which is exactly where an off-by-one in block
        accounting lands.

        Group 0's metadata is what the cost model reads, because the step's shape is
        the same for every group; the others are built for their validation.
        """
        assert self.block_tables is not None
        common = scheduler_output.num_common_prefix_blocks or []
        num_groups = self.block_tables.num_kv_cache_groups

        metadata = build_attn_metadata(
            input_batch,
            self.block_tables,
            num_common_prefix_blocks=common[0] if common else 0,
            decode_query_len=self.decode_query_len,
            group_id=0,
        )
        for group_id in range(1, num_groups):
            build_attn_metadata(
                input_batch,
                self.block_tables,
                num_common_prefix_blocks=(
                    common[group_id] if group_id < len(common) else 0
                ),
                decode_query_len=self.decode_query_len,
                group_id=group_id,
            )
        return metadata

    # --- execution -----------------------------------------------------------

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """R8.1. The one interface that crosses the simulation boundary."""
        plan = self._plan_step(scheduler_output)
        if plan is None:
            return ModelRunnerOutput.make_empty()
        input_batch, profile = plan
        return self._finish_step(
            input_batch, self.device.execute(profile), scheduler_output
        )

    async def execute_model_async(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """As `execute_model`, but yields to the event loop while time passes.

        Only under a real or scaled clock is there any time to yield -- a virtual
        clock returns instantly either way. But under `--clock-mode real` the async
        engine would otherwise block its loop for the modeled duration of every step,
        and a server that stops streaming for exactly as long as the step it is
        streaming through defeats the reason to run a real clock at all.

        The *result* is identical to the sync path by construction: both call
        `_plan_step` and `_finish_step`, and only the line that spends the duration
        differs. Two separately-written paths would eventually disagree about
        something, and the disagreement would look like a clock-mode-dependent bug.
        """
        plan = self._plan_step(scheduler_output)
        if plan is None:
            return ModelRunnerOutput.make_empty()
        input_batch, profile = plan
        return self._finish_step(
            input_batch, await self.device.execute_async(profile), scheduler_output
        )

    def _plan_step(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[InputBatch, StepProfile] | None:
        """Everything before the forward pass. `None` when there is nothing to run."""
        # Requests that left must be dropped before slots are needed for new ones.
        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)

        if scheduler_output.total_num_scheduled_tokens == 0:
            return None

        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)

        input_batch = self.prepare_inputs(scheduler_output)
        attn_metadata = self.prepare_attn(input_batch, scheduler_output)
        self.last_attn_metadata = attn_metadata

        # R8.4: a captured graph applies only to a uniform-decode batch. A mixed
        # batch has a shape that was never captured.
        graph_hit = (
            input_batch.num_reqs in self.captured_sizes
            and not attn_metadata.is_mixed_batch
            and attn_metadata.num_prefills == 0
        )

        # R18.1. Embeddings the scheduler evicted. Dropped here because the worker is
        # what holds them: the scheduler's cache manager tracks *slots*, and this
        # tracks what is actually resident on the device. The notice was carried in
        # `SchedulerOutput` and read by nobody, so the worker's set only ever grew --
        # the leak the eviction protocol exists to prevent.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_outputs.discard(mm_hash)

        # R18.1. What the scheduler told us to encode this step. Cached images are
        # absent from `scheduled_encoder_inputs` by construction, so a step that hit
        # the cache costs nothing extra -- which is the effect worth seeing.
        num_encoder_embeds = 0
        for req_id, input_ids in scheduler_output.scheduled_encoder_inputs.items():
            features = self.req_states.mm_features.get(req_id, ())
            for input_id in input_ids:
                if input_id >= len(features):
                    continue
                num_encoder_embeds += features[input_id].num_embeds
                self.encoder_outputs.add(features[input_id].identifier)

        return input_batch, StepProfile(
            num_tokens=input_batch.num_tokens,
            num_reqs=input_batch.num_reqs,
            query_lens=input_batch.num_scheduled_tokens.tolist(),
            seq_lens=input_batch.seq_lens_np.tolist(),
            is_graph_hit=graph_hit,
            num_encoder_embeds=num_encoder_embeds,
        )

    def _finish_step(
        self,
        input_batch: InputBatch,
        cost: StepCost,
        scheduler_output: SchedulerOutput | None = None,
    ) -> ModelRunnerOutput:
        """Everything after it."""
        sampler_output = self.sample_tokens(input_batch, scheduler_output)
        sampler_output.modeled_duration = cost.duration

        # The runner's own view of computed tokens must track what it just processed,
        # or the next step's positions would be wrong.
        self.postprocess_num_computed_tokens(input_batch)
        return sampler_output

    def sample_tokens(
        self, input_batch: InputBatch, scheduler_output: SchedulerOutput | None = None
    ) -> ModelRunnerOutput:
        """Produce this step's tokens per request, and next step's drafts. R14.

        Without speculation that is one token each. With it, a request whose drafts
        were verified this step emits the accepted prefix plus one -- the target
        model's own token at the first rejected position, which is what makes
        speculation lossless: the output is the same sequence either way, produced in
        fewer steps.
        """
        req_ids = input_batch.req_ids
        sampled: list[list[int]] = [[] for _ in req_ids]
        drafts: list[list[int]] = [[] for _ in req_ids]
        logprobs: LogprobsLists | None = None

        scheduled_drafts = (
            scheduler_output.scheduled_spec_decode_tokens
            if scheduler_output is not None
            else {}
        )

        # R2.2. One vector per pooling request whose prompt finished this step. A
        # pooling request is *always* "prefilling" as far as the batch is concerned
        # -- it never generates -- so it is never in `sampling_indices` and the two
        # paths cannot interfere.
        pooler_output: list[list[float] | None] | None = None
        if self.pooling_requests:
            pooler_output = [None] * len(req_ids)
            for pool_idx, pool_req_id in enumerate(req_ids):
                pooling = self.pooling_requests.get(pool_req_id)
                if pooling is None:
                    continue
                slot = int(input_batch.idx_mapping_np[pool_idx])
                if int(input_batch.seq_lens_np[pool_idx]) < int(
                    self.req_states.prompt_len[slot]
                ):
                    # Chunked prefill: the prompt is not all in yet, so there is
                    # nothing to pool over.
                    continue
                dimensions, prompt_token_ids = pooling
                pooler_output[pool_idx] = self.sim_model.embed(
                    prompt_token_ids, dimensions
                )

        sampling_indices = np.flatnonzero(~input_batch.is_prefilling_np)
        for batch_idx in sampling_indices:
            req_id = req_ids[int(batch_idx)]
            if req_id in self.pooling_requests:
                continue
            req_idx = int(input_batch.idx_mapping_np[batch_idx])
            position = int(self.req_states.num_output_tokens[req_idx])
            max_tokens = max(
                1,
                int(self.req_states.max_seq_len[req_idx])
                - int(self.req_states.prompt_len[req_idx]),
            )

            # Verify whatever drafts this step carried, then sample the token at the
            # first position they did not cover.
            num_drafts = len(scheduled_drafts.get(req_id, ()))
            accepted = self.sim_model.accepted_draft_count(req_id, num_drafts)
            tokens = [
                self.sim_model.sample_token(req_id, position + offset, max_tokens)
                for offset in range(accepted + 1)
            ]
            sampled[int(batch_idx)] = tokens
            drafts[int(batch_idx)] = self.sim_model.propose_drafts(
                req_id, position + len(tokens) - 1, self.num_spec_tokens, max_tokens
            )

        return ModelRunnerOutput(
            req_ids=list(req_ids),
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=logprobs,
            spec_token_ids=drafts if self.num_spec_tokens else None,
            pooler_output=pooler_output,
        )

    def postprocess_num_computed_tokens(self, input_batch: InputBatch) -> None:
        for batch_idx, req_idx in enumerate(input_batch.idx_mapping_np):
            self.req_states.set_num_computed_tokens(
                int(req_idx), int(input_batch.seq_lens_np[batch_idx])
            )

    # --- introspection -------------------------------------------------------

    @property
    def num_cached_requests(self) -> int:
        return self.req_states.num_reqs

    def __repr__(self) -> str:
        return (
            f"SimModelRunner(reqs={self.req_states.num_reqs}/{self.max_num_reqs}, "
            f"captured_sizes={sorted(self.captured_sizes)})"
        )
