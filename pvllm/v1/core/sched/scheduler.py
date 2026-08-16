"""The scheduler. The centerpiece.

Upstream: vllm/v1/core/sched/scheduler.py
Tier: A

C1 and C4 live here: the decision sequence per step, the total steps to drain a
workload, the preemption count, and victim selection.

**There is no prefill phase and no decode phase** (R5.1). Every request just has
`num_computed_tokens` and `num_tokens`, and each step hands out tokens so that the
former catches up to the latter. Chunked prefill, prefix caching, and speculative
decoding all fall out of that one idea rather than needing their own modes. A scheduler
written around a prefill/decode split cannot reproduce upstream's traces no matter how
carefully the rest is ported.

`schedule()` runs upstream's phase order (R5.2): running requests first, then encoder
inputs against a separate budget, then admission from waiting. The order is not
arbitrary -- serving in-progress requests before admitting new ones is what bounds
latency for work already accepted, and reversing it changes every trace.

Deferred to M4, with the call sites present so the shape is right: encoder inputs,
speculative decoding, structured output. Everything else -- continuous batching,
chunked prefill, prefix caching, and preemption by recompute -- is here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.v1.core.encoder_cache_manager import EncoderCacheManager
from pvllm.v1.core.kv_cache_manager import KVCacheManager
from pvllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from pvllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from pvllm.v1.core.sched.utils import check_stop
from pvllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
)
from pvllm.v1.kv_cache_interface import KVCacheConfig
from pvllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from pvllm.tracing import TraceSink
    from pvllm.v1.core.kv_cache_manager import KVCacheBlocks
    from pvllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


def _build_connector(vllm_config: VllmConfig) -> Any:
    """The scheduler-side KV connector, or `None`. R17.1."""
    transfer = vllm_config.kv_transfer_config
    if transfer is None or transfer.kv_connector is None:
        return None
    from pvllm.distributed.kv_transfer.base import KVConnectorRole
    from pvllm.distributed.kv_transfer.sim_connector import SimSharedStoreConnector

    return SimSharedStoreConnector(vllm_config, KVConnectorRole.SCHEDULER)


def _require_block_ids(blocks: KVCacheBlocks) -> tuple[list[int], ...]:
    """Block ids for a newly scheduled request, which must always have some.

    `get_block_ids(allow_none=True)` exists for the cached path, where "no new
    blocks this step" is normal. A *new* request that got none would mean it was
    scheduled without slots, so this narrows rather than tolerating None.
    """
    block_ids = blocks.get_block_ids()
    assert block_ids is not None
    return block_ids


class Scheduler:
    """Decides what runs each step, and folds the results back into request state."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        log_stats: bool = True,
        trace: TraceSink | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        assert vllm_config.scheduler_config is not None
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_config = kv_cache_config
        self.log_stats = log_stats

        # The trace is passed in rather than opened here: the engine core owns it,
        # because it also owns the clock that stamps every record (R19.1).
        self._trace = trace

        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        # SchedulerConfig resolves this in __post_init__; it is only Optional as an
        # input, so narrow it here rather than at every arithmetic site below.
        assert self.scheduler_config.max_num_batched_tokens is not None
        self.max_num_scheduled_tokens: int = (
            self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = self.scheduler_config.max_model_len
        self.max_num_partial_prefills = self.scheduler_config.max_num_partial_prefills
        self.policy = SchedulingPolicy(self.scheduler_config.policy)
        #: R16.1. `None` without LoRA, which is the common case and the reason the
        #: adapter-slot check below costs one attribute test.
        self.lora_config = vllm_config.lora_config
        #: R14. `None` without speculation. `num_spec_tokens` is read on the hot
        #: path, so it is unpacked once rather than reached through the config.
        self.speculative_config = vllm_config.speculative_config
        self.num_spec_tokens = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config is not None
            else 0
        )
        self.spec_disable_by_batch_size = (
            self.speculative_config.speculative_disable_by_batch_size
            if self.speculative_config is not None
            else None
        )

        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            log_stats=log_stats,
            hash_algo=self.cache_config.prefix_caching_hash_algo,
            seed=vllm_config.sim_config.seed,
        )

        #: Every request the scheduler knows about, by id.
        self.requests: dict[str, Request] = {}
        #: Admitted and progressing. Order is the scheduling order.
        self.running: list[Request] = []
        #: Waiting for admission. Preempted requests go back to the *front*.
        self.waiting = create_request_queue(self.policy)

        #: Drained into the next SchedulerOutput so the worker frees their state.
        self.finished_req_ids: set[str] = set()

        self.step_index = 0
        self.num_preemptions_total = 0
        #: Built by `_trace_step`, emitted by the engine core once stamped.
        self.pending_step_record: dict[str, Any] | None = None

        #: Requests admitted this step, awaiting a SCHEDULED event from the core.
        #: Same reason as the step record: only the core owns a clock (R19.1), and
        #: the queue wait is the interval between this stamp and the QUEUED one.
        self.pending_scheduled: list[Request] = []

        # R15. Requests set aside because something they need is not ready yet --
        # today only a compiling grammar. A separate queue rather than a flag,
        # because they must not hold the head of the waiting queue while they wait,
        # and they must not lose their place relative to each other.
        self.skipped_waiting = create_request_queue(self.policy)
        #: Requests whose grammar failed to compile, for the core to finish with an
        #: error. Held rather than raised: a bad schema fails its own request.
        self.grammar_compile_error_reqs: set[str] = set()
        # R18.1. The encoder output cache, and this step's encoder budget. Sized
        # from the token budget (see SchedulerConfig): an image whose embeddings
        # cannot fit one step could never be scheduled at all.
        self.encoder_cache_manager = EncoderCacheManager(
            self.scheduler_config.encoder_cache_size
        )
        self.max_num_encoder_input_tokens = (
            self.scheduler_config.max_num_encoder_input_tokens
        )
        # R17.1. The scheduler half of the KV connector, or `None`. Built here
        # because deciding what to load is a scheduling decision -- the worker half
        # lives in the worker and never calls this one.
        self.connector = _build_connector(vllm_config)
        #: R14. Rejected drafts from the last step, per request, for the metrics.
        self.last_num_invalid_spec_tokens: dict[str, int] = {}
        #: R14. Cumulative draft counters, for `vllm:spec_decode_*`.
        self.num_draft_tokens_total = 0
        self.num_accepted_tokens_total = 0

    # --- admission -----------------------------------------------------------

    def add_request(self, request: Request) -> None:
        # F8: hashing policy belongs to the KV manager, so the hasher is attached
        # here rather than at Request construction, where the frontend would have to
        # know the block size and the algorithm.
        if self.kv_cache_manager.block_hasher is not None:
            request.attach_block_hasher(self.kv_cache_manager.block_hasher)
        self.waiting.add_request(request)
        self.requests[request.request_id] = request

    def has_requests(self) -> bool:
        """Whether any work remains -- running, waiting, or awaiting cleanup.

        `skipped_waiting` counts: a request set aside for a compiling grammar is
        still work, and an engine that reported otherwise would stop stepping and
        never come back to it.
        """
        return (
            bool(self.running)
            or bool(self.waiting)
            or bool(self.skipped_waiting)
            or bool(self.finished_req_ids)
        )

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.skipped_waiting)

    def get_request_counts(self) -> tuple[int, int]:
        """`(running, waiting)`, for the metrics. A grammar-blocked request is
        waiting as far as anyone outside the scheduler is concerned."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def get_kv_cache_usage(self) -> float:
        return self.kv_cache_manager.usage

    def _num_partial_prefills(self) -> int:
        """Running requests whose prompt is not yet fully computed. R5.4."""
        return sum(1 for request in self.running if request.is_prefill_chunk)

    # --- the step ------------------------------------------------------------

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Decide what runs this step. R5.2."""
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # R18.1. Phase 2's state: which encoder inputs run this step, and what is
        # left of the separate budget they draw on.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        token_budget = self.max_num_scheduled_tokens
        # Read once: both the running loop and the admission loop cap a request's
        # share of the step by it (R5.4).
        threshold = self.scheduler_config.long_prefill_token_threshold

        # --- phase 1: requests already running -------------------------------
        #
        # Served before admission so that work already accepted keeps making
        # progress. Admitting first would let a stream of new arrivals starve the
        # requests a client is already waiting on.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # R14. Speculation is turned off above a batch size, as upstream does:
            # verification stops paying once the batch is large enough to saturate
            # the device on its own, because the wasted work competes with real
            # decodes. Checked per step against the *running* count, so a burst of
            # traffic disables it and a lull turns it back on.
            if (
                self.spec_disable_by_batch_size is not None
                and len(self.running) > self.spec_disable_by_batch_size
            ):
                request.spec_token_ids = []

            # R14. A decoding request with drafts in hand verifies all of them in
            # this step: one token for the position the model is at, plus one per
            # draft. `num_tokens_with_spec` is what makes that fall out of the same
            # arithmetic every other request uses, rather than needing a decode mode.
            num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
            assert num_new_tokens > 0, (
                f"request {request.request_id} has {request.num_computed_tokens} "
                f"computed tokens against {request.num_tokens_with_spec} total, so "
                f"this step would schedule {num_new_tokens}. The count ran ahead of "
                f"the request's history -- under speculation that means rejected "
                f"drafts were not rolled back (R14)."
            )
            if 0 < threshold < num_new_tokens:
                num_new_tokens = threshold
            num_new_tokens = min(num_new_tokens, token_budget)
            # Never advance past the length cap.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - request.num_computed_tokens
            )

            if num_new_tokens <= 0:
                # Nothing to do for this request this step. `continue`, not `break`:
                # a later request may still be schedulable, and upstream deliberately
                # relaxes strict FCFS here.
                req_index += 1
                continue

            # --- phase 2: encoder inputs -------------------------------------
            #
            # R18.1. Against a *separate* budget, because encoder work and decoder
            # work do not trade against each other: an image costs vision-encoder
            # time whatever the token budget is doing. A request whose images will
            # not fit this step has its token count trimmed to stop before the
            # first one, so it makes progress on the text instead of stalling.
            num_new_tokens, encoder_inputs, encoder_budget = self._schedule_encoder(
                request, num_new_tokens, encoder_budget
            )
            if num_new_tokens <= 0:
                req_index += 1
                continue
            if encoder_inputs:
                scheduled_encoder_inputs[request.request_id] = encoder_inputs

            # Allocate, preempting from the tail until it fits (R5.5).
            new_blocks = None
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )
                if new_blocks is not None:
                    break

                # Victim selection (C4). FCFS preempts the *last* running request --
                # the most recently admitted, which has computed the least and so
                # wastes the least work on recompute. Priority preempts the lowest
                # priority, breaking ties by latest arrival.
                if self.policy == SchedulingPolicy.PRIORITY:
                    victim = max(
                        self.running, key=lambda r: (r.priority, r.arrival_time)
                    )
                    self.running.remove(victim)
                    if victim in scheduled_running_reqs:
                        scheduled_running_reqs.remove(victim)
                        token_budget += num_scheduled_tokens.pop(victim.request_id)
                        req_to_new_blocks.pop(victim.request_id, None)
                        req_index -= 1
                else:
                    victim = self.running.pop()

                self._preempt_request(victim)
                preempted_reqs.append(victim)

                if victim is request:
                    # Preempted ourselves; there is nothing left to give up.
                    break

            if new_blocks is None:
                # Even preempting everything else did not free enough. Stop here
                # rather than skipping ahead: nothing later will fit either.
                break

            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

            # R14. Which drafts this step actually verifies. The budget may have
            # trimmed the batch, so only the prefix that fits is sent -- and the
            # rest is dropped rather than carried, because a draft proposed against
            # a token that has since been superseded is not a draft any more.
            if request.spec_token_ids:
                num_spec_scheduled = (
                    num_new_tokens + request.num_computed_tokens - request.num_tokens
                )
                if num_spec_scheduled > 0:
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids[:num_spec_scheduled]
                    )
                request.spec_token_ids = []

            req_index += 1

        # --- phase 2: encoder inputs -----------------------------------------
        # A separate budget from the token budget (R5.2). Multimodal lands in M4;
        # the phase is named here so its position in the order is not lost.

        # R16.1. Which adapters this step already has resident. Bounded by
        # `max_loras`: admitting a request for a further adapter would need a slot
        # that does not exist, so it waits -- a real source of queueing in a
        # multi-tenant deployment, and invisible to anyone who did not model it.
        scheduled_loras: set[int] = set()
        if self.lora_config is not None:
            scheduled_loras = {
                request.lora_request.lora_int_id
                for request in self.running
                if request.lora_request is not None
            }
            assert len(scheduled_loras) <= self.lora_config.max_loras, (
                f"{len(scheduled_loras)} adapters are resident but max_loras is "
                f"{self.lora_config.max_loras} (R16.1)"
            )

        # --- phase 3: admission from waiting ---------------------------------
        #
        # Skipped entirely if anything was preempted this step: the pool is already
        # oversubscribed, and admitting more would preempt the requests just
        # preempted, thrashing instead of draining.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) >= self.max_num_running_reqs:
                    break

                # R5.4: cap how many requests may be mid-prefill at once. Without
                # it, a burst of long prompts all start chunking together and each
                # one's first token waits for every other prompt to finish
                # prefilling -- the batch stays busy while every TTFT gets worse.
                if (
                    self.scheduler_config.enable_chunked_prefill
                    and self._num_partial_prefills() >= self.max_num_partial_prefills
                ):
                    break

                request = self.waiting.peek_request()

                # R16.1. No free adapter slot for this request's adapter. Set aside
                # rather than breaking the loop: a request behind it may want an
                # adapter that *is* resident, and stopping here would let one
                # tenant's queue block every other tenant's.
                if (
                    self.lora_config is not None
                    and request.lora_request is not None
                    and len(scheduled_loras) >= self.lora_config.max_loras
                    and request.lora_request.lora_int_id not in scheduled_loras
                ):
                    self.waiting.pop_request()
                    self.skipped_waiting.add_request(request)
                    continue

                # R15. A request whose grammar is still compiling cannot be admitted:
                # the first sampled token has to be constrained, and there is nothing
                # to constrain it with yet. Moved aside rather than left at the head,
                # so one slow schema does not block every request behind it -- which
                # is the whole reason compilation is asynchronous.
                blocked = (
                    request.status
                    == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
                )
                if blocked and not self._promote_grammar_request(request):
                    self.waiting.pop_request()
                    self.skipped_waiting.add_request(request)
                    continue

                # R17.1. What an external store holds beyond the local cache.
                # Asked before allocation, because the blocks the KV manager hands
                # out have to cover the tokens about to be pulled into them.
                # R6.4. On the real path, so admission accounting is identical
                # whether or not the cache is enabled.
                new_computed_blocks, num_new_local_computed_tokens = (
                    self.kv_cache_manager.get_computed_blocks(request)
                )
                num_computed_tokens = num_new_local_computed_tokens

                # R17.1. What an external store holds beyond the local cache.
                # Asked *after* the local lookup, because pulling KV the engine
                # already has in memory would be strictly worse than using it.
                num_external_tokens = 0
                if self.connector is not None:
                    num_external_tokens, _ = self.connector.get_num_new_matched_tokens(
                        request, num_computed_tokens
                    )
                    num_computed_tokens += num_external_tokens

                num_new_tokens = request.num_tokens - num_computed_tokens
                # R5.4: cap any single request's share of the step, so one very long
                # prompt cannot monopolize a step and stall every decode behind it.
                if 0 < threshold < num_new_tokens:
                    num_new_tokens = threshold

                if (
                    not self.scheduler_config.enable_chunked_prefill
                    and num_new_tokens > token_budget
                ):
                    # Without chunking the whole prompt must fit in this step. It
                    # may fit in a later one, so stop rather than skip.
                    break

                num_new_tokens = min(num_new_tokens, token_budget)
                if num_new_tokens <= 0:
                    break

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    # R17.1. The externally-held tokens count as *computed* -- the
                    # request will not run them -- but blocks still have to exist to
                    # receive them, so they are included here. Advancing
                    # `num_computed_tokens` without this allocates nothing for the
                    # pulled KV, and the slot-mapping oracle catches it as a write
                    # past the end of the block table (R8.3).
                    num_new_computed_tokens=(
                        num_new_local_computed_tokens + num_external_tokens
                    ),
                    new_computed_blocks=new_computed_blocks,
                )
                if new_blocks is None:
                    # Does not fit. A later step may have room; leave it at the head
                    # of the queue so it is tried first.
                    break

                # R18.1. Same encoder budget, same trimming rule, before the
                # request is committed to the batch.
                num_new_tokens, encoder_inputs, encoder_budget = self._schedule_encoder(
                    request, num_new_tokens, encoder_budget
                )
                if num_new_tokens <= 0:
                    self.waiting.pop_request()
                    self.skipped_waiting.add_request(request)
                    self.kv_cache_manager.free(request)
                    continue
                if encoder_inputs:
                    scheduled_encoder_inputs[request.request_id] = encoder_inputs

                request = self.waiting.pop_request()
                self.running.append(request)
                # Collected rather than stamped: the scheduler has no clock (R19.1),
                # so the engine core dates these before the step's outputs are built.
                self.pending_scheduled.append(request)

                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"request {request.request_id} was admitted from the waiting "
                        f"queue with unexpected status {request.status}"
                    )

                if self.connector is not None and num_external_tokens:
                    # The blocks are allocated; tell the connector which ones this
                    # step must fill from the store before the model reads them.
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request.request_id),
                        num_external_tokens,
                    )

                request.status = RequestStatus.RUNNING
                if request.lora_request is not None:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                request.num_computed_tokens = num_computed_tokens
                request.num_cached_tokens = num_new_local_computed_tokens
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens, (
            f"scheduled {total_num_scheduled_tokens} tokens against a budget of "
            f"{self.max_num_scheduled_tokens} (R5.3)"
        )
        assert len(self.running) <= self.max_num_running_reqs, (
            f"{len(self.running)} running requests exceeds max_num_seqs="
            f"{self.max_num_running_reqs} (R5.3)"
        )

        # R15. Set-aside requests go back before the step ends, so the next
        # step reconsiders them -- otherwise a grammar that compiles during this
        # step would not be noticed until something else disturbed the queue.
        self._restore_skipped_waiting()

        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    self.running[0].request_id
                )
            )

        # D6/F1: the V2 runner rebuilds a resumed request's state from scratch, so
        # resumed requests are handed over as new rather than cached. Sending them as
        # cached would ask the worker to patch state it discarded on preemption.
        scheduled_new_reqs.extend(scheduled_resumed_reqs)
        new_reqs_data = [
            NewRequestData.from_request(
                req, _require_block_ids(req_to_new_blocks[req.request_id])
            )
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs, num_scheduled_tokens, req_to_new_blocks
        )

        # R15. Which of this step's requests are constrained, and which bitmask row
        # belongs to each. Keyed by request id and assigned in sorted id order rather
        # than by batch position: the worker reorders the batch for its own reasons
        # (see sort_batch_req_ids), so a row index derived from the scheduler's
        # ordering would address the wrong request's row about half the time.
        #
        # A request on a non-final prefill chunk is excluded -- it samples no token
        # this step, so constraining it would consume a grammar position for a token
        # that never exists. Upstream excludes it for the same reason.
        constrained = sorted(
            request_id
            for request_id in num_scheduled_tokens
            if self.requests[request_id].use_structured_output
            and not self.requests[request_id].is_prefill_chunk
        )
        structured_output_request_ids = {
            request_id: row for row, request_id in enumerate(constrained)
        }

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            has_structured_output_requests=bool(structured_output_request_ids),
            structured_output_request_ids=structured_output_request_ids,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            kv_connector_metadata=(
                self.connector.build_connector_meta(None)
                if self.connector is not None
                else None
            ),
            preempted_req_ids={r.request_id for r in preempted_reqs} or None,
        )

        self._update_after_schedule(scheduler_output)
        self._trace_step(scheduler_output, preempted_reqs)
        return scheduler_output

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []

        for request in running_reqs:
            req_id = request.request_id
            req_ids.append(req_id)
            # The worker already holds every token up to num_computed_tokens; only
            # what it has not seen goes on the wire (R7.3).
            num_new = num_scheduled_tokens[req_id]
            start = request.num_computed_tokens
            new_token_ids.append(list(request.all_token_ids[start : start + num_new]))
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(request.num_computed_tokens)
            num_output_tokens.append(request.num_output_tokens)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=set(),
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Advance computed-token counts *after* the decision is built.

        The decision must carry the pre-advance counts, because that is what tells
        the worker which input ids to read. Advancing here rather than at the call
        site lets the next step schedule a partially-prefilled request immediately.
        """
        for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled
            request.is_prefill_chunk = request.num_computed_tokens < request.num_tokens

        # Cleared only after the output has been built, so the worker still sees the
        # ids it needs to drop.
        self.finished_req_ids = set()

    def _preempt_request(self, request: Request) -> None:
        """Preempt by recompute. R5.5.

        Everything the request computed is thrown away: its blocks go back to the
        pool and `num_computed_tokens` resets to zero. That is what "by recompute"
        means -- no KV is swapped out, it is simply recomputed on resume. The output
        tokens already produced are kept, so the request resumes mid-generation
        rather than restarting.
        """
        assert request.status == RequestStatus.RUNNING, (
            f"only running requests can be preempted, got {request.status}"
        )
        self.kv_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.num_preemptions += 1
        self.num_preemptions_total += 1
        # R14. Drafts were proposed against a KV cache this request no longer has.
        request.spec_token_ids = []

        # Back to the *front* of the waiting queue: sending it to the back would let
        # newer arrivals overtake it indefinitely.
        self.waiting.prepend_request(request)

    # --- folding results back ------------------------------------------------

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Map model output back onto request state. R5.7."""
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        still_running: list[Request] = []
        num_invalid_spec_tokens: dict[str, int] = {}

        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                still_running.append(request)
                continue

            index = model_runner_output.req_id_to_index.get(req_id)
            generated = (
                sampled_token_ids[index]
                if index is not None and index < len(sampled_token_ids)
                else []
            )

            # R14. A step schedules `1 + num_drafts` tokens and gets back
            # `1 + num_accepted`. Everything in between was computed against drafts
            # the target rejected, so it has to come back off the count -- otherwise
            # `num_computed_tokens` runs ahead of the request's actual history and
            # the next step schedules a negative number of tokens, which is a loop
            # that never terminates rather than an error that says anything.
            num_drafts = len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            )
            if num_drafts:
                self.num_draft_tokens_total += num_drafts
                self.num_accepted_tokens_total += max(0, len(generated) - 1)
                rejected = num_drafts - max(0, len(generated) - 1)
                if rejected > 0:
                    request.num_computed_tokens -= rejected
                    num_invalid_spec_tokens[req_id] = rejected

            new_token_ids: list[int] = []
            stopped = False
            if generated:
                # A request still being prefilled gets no tokens back; the runner
                # only samples once the whole prompt is computed.
                new_token_ids, stopped = self._update_request_with_output(
                    request, generated
                )

            if stopped:
                self._free_request(request)
            else:
                still_running.append(request)

            if new_token_ids or stopped:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        self.last_num_invalid_spec_tokens = num_invalid_spec_tokens

        # R6.7 + R14. Window eviction runs *here*, after the step's output has been
        # folded back and any rejected drafts rolled off `num_computed_tokens` --
        # not at schedule time. Scheduling inflates the count by the drafts it is
        # about to verify, and evicting on that inflated boundary freed blocks that
        # were still inside the true window. Those blocks went back to the pool and
        # were handed to other requests, which is cross-request KV corruption: the
        # exact failure the window is not allowed to cause.
        for request in still_running:
            self.kv_cache_manager.remove_skipped_blocks(request)

        # R14. The drafts the runner proposed for the *next* step. Stored on the
        # request rather than carried in the output, because whether they are still
        # usable depends on what the scheduler does next -- a request that gets
        # preempted before its next step must not verify stale drafts.
        if model_runner_output.spec_token_ids is not None:
            for request in still_running:
                index = model_runner_output.req_id_to_index.get(request.request_id)
                if index is not None and index < len(
                    model_runner_output.spec_token_ids
                ):
                    request.spec_token_ids = list(
                        model_runner_output.spec_token_ids[index]
                    )

        self.running = still_running

        engine_core_outputs = {
            client_index: EngineCoreOutputs(
                engine_index=client_index, outputs=client_outputs
            )
            for client_index, client_outputs in outputs.items()
        }
        return engine_core_outputs

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """Append tokens one at a time, stopping the moment a stop condition hits.

        One at a time, and checking after each, because a request that stops on its
        second of three sampled tokens must not emit the third. Speculative decoding
        makes multi-token batches common; without the trim, a stop string could be
        overshot by two tokens.
        """
        stopped = False
        for num_new, token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(token_id)
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]
                break
        return new_token_ids, stopped

    # --- termination ---------------------------------------------------------

    def finish_requests(
        self,
        request_ids: str | list[str],
        finished_status: RequestStatus,
    ) -> None:
        """Terminate requests from outside the step loop -- an abort, typically (R2.4).

        The blocks are freed here, within the same step, so a disconnected client's
        capacity is returned immediately rather than at the end of its generation.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = [request_ids]

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
            else:
                self.waiting.remove_request(request)

            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        # R17.1. Offered to the connector before the blocks go back, which is the
        # only moment its KV is both complete and still resident.
        if self.connector is not None:
            blocks = self.kv_cache_manager.get_blocks(request.request_id)
            block_ids = blocks.get_block_ids()
            self.connector.request_finished(request, block_ids or ())

        # R18.1. Dropped, not evicted: the embeddings stay resident so the next
        # request with the same image still hits.
        if request.mm_features:
            self.encoder_cache_manager.free(request)
        assert request.is_finished(), (
            f"request {request.request_id} freed while still {request.status}"
        )
        self.kv_cache_manager.free(request)
        # The worker keeps per-request state until it is told to drop it (R5.8).
        self.finished_req_ids.add(request.request_id)
        self.requests.pop(request.request_id, None)

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def shutdown(self) -> None:
        return None

    # --- tracing -------------------------------------------------------------

    def _trace_step(
        self, scheduler_output: SchedulerOutput, preempted: list[Request]
    ) -> None:
        """Build one step's trace record. R5.10, R19.3.

        Built here but *emitted* by the engine core, which stamps it. The scheduler
        has no clock to read (R19.1), and a record without a timestamp is useless
        for the thing traces exist for.
        """
        self.step_index += 1
        if self._trace is None:
            self.pending_step_record = None
            return

        record: dict[str, Any] = scheduler_output.to_trace_dict()
        cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        record.update(
            step=self.step_index,
            num_running=len(self.running),
            num_waiting=len(self.waiting),
            kv_usage=round(self.kv_cache_manager.usage, 6),
            num_preemptions_total=self.num_preemptions_total,
            prefix_cache_hits=cache_stats.hits,
            prefix_cache_queries=cache_stats.queries,
            # Sorted, and ids rather than just a count: a request starving behind a
            # long prefill is the thing a timeline is opened to find, and a count
            # cannot show *which* request waited.
            waiting_req_ids=sorted(r.request_id for r in self.waiting),
        )
        if preempted:
            record["preemptions"] = {r.request_id: r.num_preemptions for r in preempted}
        self.pending_step_record = record

    def take_step_record(self) -> dict[str, Any] | None:
        """Hand the last step's record to the engine core for stamping."""
        record, self.pending_step_record = self.pending_step_record, None
        return record

    def _schedule_encoder(
        self, request: Request, num_new_tokens: int, encoder_budget: int
    ) -> tuple[int, list[int], int]:
        """Decide which of a request's images run this step. R18.1.

        Returns `(num_new_tokens, input_ids, encoder_budget)`. `num_new_tokens` may
        come back *smaller*: if an image cannot be encoded this step, the request is
        trimmed to stop just before its first placeholder rather than being blocked
        entirely. Scheduling past a placeholder whose embeddings do not exist would
        mean attending to KV that was never written.

        A no-op for a text request, which is what keeps the whole encoder path off
        the hot loop for the overwhelmingly common case.
        """
        if not request.mm_features:
            return num_new_tokens, [], encoder_budget

        scheduled: list[int] = []
        start = request.num_computed_tokens
        end = start + num_new_tokens

        for input_id, feature in enumerate(request.mm_features):
            # Only items this step's token span actually touches.
            if feature.position >= end or feature.position + feature.length <= start:
                continue
            if input_id in self.encoder_cache_manager.get_cached_input_ids(request):
                continue

            hit = self.encoder_cache_manager.has_cache(request, input_id)
            self.encoder_cache_manager.record_lookup(hit)
            if hit:
                # Already resident from another request, or from this one's earlier
                # chunk. Take a reference; no encoder work and no budget.
                self.encoder_cache_manager.allocate(request, input_id)
                continue

            if feature.num_embeds > encoder_budget or not (
                self.encoder_cache_manager.can_allocate(request, input_id)
            ):
                # Trim to stop before this image. `max(0, ...)` because the
                # placeholder may start exactly at the request's current position,
                # in which case there is nothing to do this step at all.
                return max(0, feature.position - start), scheduled, encoder_budget

            self.encoder_cache_manager.allocate(request, input_id)
            encoder_budget -= feature.num_embeds
            scheduled.append(input_id)

        return num_new_tokens, scheduled, encoder_budget

    def _promote_grammar_request(self, request: Request) -> bool:
        """Whether a grammar-blocked request is ready to be admitted. R15.

        Mirrors upstream's promotion check, including the part that matters most: a
        compilation *failure* does not raise here. It marks the request so the engine
        core can finish it with an error, and returns False so admission moves on --
        one malformed schema must not take down the step that noticed it.
        """
        structured = request.structured_output_request
        if structured is None:
            request.status = RequestStatus.WAITING
            return True
        grammar = structured.grammar
        if grammar is None:
            return False
        if isinstance(grammar, Exception):
            self.grammar_compile_error_reqs.add(request.request_id)
            return False
        request.status = (
            RequestStatus.PREEMPTED
            if request.num_preemptions
            else RequestStatus.WAITING
        )
        return True

    def _restore_skipped_waiting(self) -> None:
        """Return set-aside requests to the head of the waiting queue. R15.

        At the head, not the back: they arrived before everything now queued behind
        them, and sending them to the back would let a steady stream of unconstrained
        requests starve every constrained one indefinitely.
        """
        if self.skipped_waiting:
            self.waiting.prepend_requests(self.skipped_waiting)
            self.skipped_waiting = create_request_queue(self.policy)

    def take_grammar_compile_errors(self) -> set[str]:
        """Hand the core the requests whose grammar failed, and forget them."""
        failed, self.grammar_compile_error_reqs = self.grammar_compile_error_reqs, set()
        return failed

    def take_newly_scheduled(self) -> list[Request]:
        """Hand this step's admissions to the core so it can date them.

        Must be drained between `schedule()` and `update_from_output()`: the latter
        calls `take_events()`, and an event recorded after that would ride out on the
        *next* step's output, attributing the wait to the wrong instant.
        """
        requests, self.pending_scheduled = self.pending_scheduled, []
        return requests

    def make_stats(self) -> dict[str, Any]:
        """A snapshot for the metrics layer (R12.1)."""
        stats: dict[str, Any] = {
            "num_running_reqs": len(self.running),
            "num_waiting_reqs": len(self.waiting),
            "kv_cache_usage": self.kv_cache_manager.usage,
            "num_preemptions": self.num_preemptions_total,
            "num_draft_tokens": self.num_draft_tokens_total,
            "num_accepted_tokens": self.num_accepted_tokens_total,
            "step_index": self.step_index,
        }
        stats.update(self.kv_cache_manager.make_prefix_cache_stats().as_dict())
        return stats


__all__ = ["EngineCoreEventType", "FinishReason", "Scheduler"]
