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

Deferred to later milestones, with the call sites present so the shape is right:
chunked prefill splitting (M2), prefix caching (M2), encoder inputs (M4), speculative
decoding (M4), structured output (M4). Preemption by recompute is *here*, because the
scheduler cannot make progress under a full KV pool without it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
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
        self.policy = SchedulingPolicy(self.scheduler_config.policy)

        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            log_stats=log_stats,
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

    # --- admission -----------------------------------------------------------

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request

    def has_requests(self) -> bool:
        """Whether any work remains -- running, waiting, or awaiting cleanup."""
        return bool(self.running) or bool(self.waiting) or bool(self.finished_req_ids)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def get_request_counts(self) -> tuple[int, int]:
        """`(running, waiting)`, for the metrics."""
        return len(self.running), len(self.waiting)

    def get_kv_cache_usage(self) -> float:
        return self.kv_cache_manager.usage

    # --- the step ------------------------------------------------------------

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        """Decide what runs this step. R5.2."""
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
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

            num_new_tokens = request.num_tokens - request.num_computed_tokens
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
            req_index += 1

        # --- phase 2: encoder inputs -----------------------------------------
        # A separate budget from the token budget (R5.2). Multimodal lands in M4;
        # the phase is named here so its position in the order is not lost.

        # --- phase 3: admission from waiting ---------------------------------
        #
        # Skipped entirely if anything was preempted this step: the pool is already
        # oversubscribed, and admitting more would preempt the requests just
        # preempted, thrashing instead of draining.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) >= self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # Prefix cache lookup. Empty until M2, but on the real path so
                # admission accounting does not change when caching lands.
                new_computed_blocks, num_new_local_computed_tokens = (
                    self.kv_cache_manager.get_computed_blocks(request)
                )
                num_computed_tokens = num_new_local_computed_tokens

                num_new_tokens = request.num_tokens - num_computed_tokens
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
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                )
                if new_blocks is None:
                    # Does not fit. A later step may have room; leave it at the head
                    # of the queue so it is tried first.
                    break

                request = self.waiting.pop_request()
                self.running.append(request)

                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"request {request.request_id} was admitted from the waiting "
                        f"queue with unexpected status {request.status}"
                    )

                request.status = RequestStatus.RUNNING
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

        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    self.running[0].request_id, len(self.running)
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

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=num_common_prefix_blocks,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=[],
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
        """One record per engine step. R5.10, R19.3."""
        self.step_index += 1
        if self._trace is None:
            return

        record: dict[str, Any] = scheduler_output.to_trace_dict()
        record.update(
            step=self.step_index,
            num_running=len(self.running),
            num_waiting=len(self.waiting),
            kv_usage=round(self.kv_cache_manager.usage, 6),
            num_preemptions_total=self.num_preemptions_total,
        )
        if preempted:
            record["preempted_num_computed"] = {
                r.request_id: r.num_preemptions for r in preempted
            }
        # `t` is filled in by the engine core, which owns the clock (R19.1).
        self._trace.emit("step", **record)

    def make_stats(self) -> dict[str, Any]:
        """A snapshot for the metrics layer (R12.1)."""
        queries, hits = self.kv_cache_manager.make_prefix_cache_stats()
        return {
            "num_running_reqs": len(self.running),
            "num_waiting_reqs": len(self.waiting),
            "kv_cache_usage": self.kv_cache_manager.usage,
            "prefix_cache_queries": queries,
            "prefix_cache_hits": hits,
            "num_preemptions": self.num_preemptions_total,
            "step_index": self.step_index,
        }


__all__ = ["EngineCoreEventType", "FinishReason", "Scheduler"]
