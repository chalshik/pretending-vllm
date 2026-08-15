"""Scheduler decisions. R5.1--R5.8, C1, C4."""

from __future__ import annotations

import pytest

from pvllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from pvllm.sampling_params import SamplingParams
from pvllm.v1.core.sched.scheduler import Scheduler
from pvllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from pvllm.v1.outputs import ModelRunnerOutput
from pvllm.v1.request import Request, RequestStatus


def make_scheduler(
    *,
    num_blocks: int = 64,
    block_size: int = 4,
    max_num_batched_tokens: int = 64,
    max_num_seqs: int = 8,
    max_model_len: int = 128,
    enable_chunked_prefill: bool = True,
    policy: str = "fcfs",
    long_prefill_token_threshold: int = 0,
    max_num_partial_prefills: int = 1,
) -> Scheduler:
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="tiny-test", max_model_len=max_model_len),
        cache_config=CacheConfig(block_size=block_size, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enable_chunked_prefill=enable_chunked_prefill,
            policy=policy,
            long_prefill_token_threshold=long_prefill_token_threshold,
            max_num_partial_prefills=max_num_partial_prefills,
        ),
    )
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=32,
        dtype="bfloat16",
        dtype_bytes=2,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
    )
    return Scheduler(vllm_config, kv_cache_config, log_stats=False)


def make_request(
    request_id: str = "r0",
    prompt_len: int = 8,
    max_tokens: int = 4,
    arrival_time: float = 0.0,
    priority: int = 0,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        arrival_time=arrival_time,
        priority=priority,
    )


def runner_output(scheduler, scheduler_output, token: int = 999) -> ModelRunnerOutput:
    """Stand in for `SimModelRunner`.

    A request still mid-prefill produces no logits and therefore no token, and the
    real runner returns an empty list for it. The first version of this helper
    guessed from `scheduled_new_reqs`, which only sees a request's *first* chunk --
    so from the second chunk onward it handed out tokens the real runner would never
    produce, and chunked-prefill tests measured a contract the runner does not honour.

    `schedule()` has already advanced `num_computed_tokens` and set
    `is_prefill_chunk` by the time this runs, so the request's own state is the
    authority.
    """
    req_ids = list(scheduler_output.num_scheduled_tokens)
    sampled = [
        [] if scheduler.requests[req_id].is_prefill_chunk else [token]
        for req_id in req_ids
    ]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=sampled,
    )


# --- the budget ------------------------------------------------------------


def test_token_budget_is_never_exceeded():
    """R5.3. Asserted inside schedule() too, but pinned here as the contract."""
    scheduler = make_scheduler(max_num_batched_tokens=16, block_size=4)
    for i in range(8):
        scheduler.add_request(make_request(f"r{i}", prompt_len=8))

    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens <= 16
    assert (
        sum(output.num_scheduled_tokens.values()) == output.total_num_scheduled_tokens
    )


def test_max_num_seqs_caps_concurrency():
    """R5.3."""
    scheduler = make_scheduler(max_num_seqs=2, max_num_batched_tokens=256)
    for i in range(6):
        scheduler.add_request(make_request(f"r{i}", prompt_len=4))

    scheduler.schedule()
    assert len(scheduler.running) == 2


def test_empty_queue_schedules_nothing():
    scheduler = make_scheduler()
    output = scheduler.schedule()
    assert output.total_num_scheduled_tokens == 0
    assert output.scheduled_new_reqs == []


# --- continuous batching ---------------------------------------------------


def test_running_requests_are_served_before_new_admissions():
    """R5.2. Reversing this lets a stream of arrivals starve accepted work."""
    scheduler = make_scheduler(max_num_batched_tokens=12, block_size=4)
    scheduler.add_request(make_request("first", prompt_len=8))
    first = scheduler.schedule()
    assert [r.req_id for r in first.scheduled_new_reqs] == ["first"]
    scheduler.update_from_output(first, runner_output(scheduler, first))

    scheduler.add_request(make_request("second", prompt_len=8))
    second = scheduler.schedule()
    # "first" is decoding and gets its token before "second" is admitted.
    assert "first" in second.num_scheduled_tokens
    assert second.scheduled_cached_reqs.req_ids == ["first"]


def test_decode_costs_one_token_per_step():
    scheduler = make_scheduler()
    scheduler.add_request(make_request("r0", prompt_len=8, max_tokens=4))

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens["r0"] == 8
    scheduler.update_from_output(prefill, runner_output(scheduler, prefill))

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens["r0"] == 1


def test_new_requests_carry_block_ids_and_cached_ones_carry_diffs():
    """R7.3: the worker patches state incrementally rather than rebuilding it."""
    scheduler = make_scheduler(block_size=4)
    scheduler.add_request(make_request("r0", prompt_len=8))

    first = scheduler.schedule()
    assert first.scheduled_new_reqs[0].block_ids == ([0, 1],)
    scheduler.update_from_output(first, runner_output(scheduler, first))

    second = scheduler.schedule()
    cached = second.scheduled_cached_reqs
    assert cached.req_ids == ["r0"]
    assert cached.num_computed_tokens == [8]
    # One new token arrived; it is what the worker has not seen.
    assert cached.new_token_ids == [[999]]


# --- termination -----------------------------------------------------------


def test_request_finishes_at_max_tokens_and_frees_its_blocks():
    scheduler = make_scheduler()
    scheduler.add_request(make_request("r0", prompt_len=8, max_tokens=2))

    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output))
    for _ in range(4):
        if not scheduler.running:
            break
        output = scheduler.schedule()
        scheduler.update_from_output(output, runner_output(scheduler, output))

    assert scheduler.running == []
    assert scheduler.get_kv_cache_usage() == 0.0


def test_finished_ids_reach_the_worker_exactly_once():
    """R5.8: the worker drops its cached state when it sees the id."""
    scheduler = make_scheduler()
    scheduler.add_request(make_request("r0", prompt_len=4, max_tokens=1))

    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output))
    assert scheduler.running == []

    following = scheduler.schedule()
    assert following.finished_req_ids == {"r0"}

    after = scheduler.schedule()
    assert after.finished_req_ids == set()


def test_stop_token_finishes_the_request():
    """R11.5."""
    scheduler = make_scheduler()
    request = make_request("r0", prompt_len=4, max_tokens=16)
    assert request.sampling_params is not None
    request.sampling_params.stop_token_ids = [42]
    scheduler.add_request(request)

    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output, token=42))

    assert request.status is RequestStatus.FINISHED_STOPPED
    assert request.stop_reason == 42


def test_abort_frees_capacity_within_one_step():
    """R2.4: a disconnected client's blocks come back immediately."""
    scheduler = make_scheduler(block_size=4)
    scheduler.add_request(make_request("r0", prompt_len=16, max_tokens=8))
    scheduler.schedule()
    assert scheduler.get_kv_cache_usage() > 0

    scheduler.finish_requests("r0", RequestStatus.FINISHED_ABORTED)
    assert scheduler.get_kv_cache_usage() == 0.0
    assert scheduler.running == []


def test_aborting_a_waiting_request_removes_it_from_the_queue():
    scheduler = make_scheduler(max_num_seqs=1)
    scheduler.add_request(make_request("r0", prompt_len=4))
    scheduler.add_request(make_request("r1", prompt_len=4))
    scheduler.schedule()

    scheduler.finish_requests("r1", RequestStatus.FINISHED_ABORTED)
    assert len(scheduler.waiting) == 0


# --- preemption (C4) -------------------------------------------------------


def test_preemption_frees_blocks_and_requeues_at_the_front():
    """R5.5. By recompute: computed tokens reset, output tokens kept."""
    # Four blocks total; two requests of 8 tokens each need two blocks apiece.
    scheduler = make_scheduler(num_blocks=4, block_size=4, max_num_batched_tokens=64)
    scheduler.add_request(make_request("a", prompt_len=8, max_tokens=16))
    scheduler.add_request(make_request("b", prompt_len=8, max_tokens=16))

    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output))

    # Both are now decoding into a pool with no spare blocks; the next append
    # forces a preemption.
    preempted_seen = False
    for _ in range(6):
        output = scheduler.schedule()
        if output.preempted_req_ids:
            preempted_seen = True
            break
        scheduler.update_from_output(output, runner_output(scheduler, output))

    assert preempted_seen, "a full KV pool must eventually force preemption"
    assert scheduler.num_preemptions_total >= 1


def test_fcfs_preempts_the_most_recently_admitted():
    """C4 victim selection: the last running request has computed the least, so it
    wastes the least work on recompute."""
    scheduler = make_scheduler(num_blocks=4, block_size=4)
    scheduler.add_request(make_request("first", prompt_len=8, max_tokens=16))
    scheduler.add_request(make_request("second", prompt_len=8, max_tokens=16))

    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output))

    for _ in range(6):
        output = scheduler.schedule()
        if output.preempted_req_ids:
            assert output.preempted_req_ids == {"second"}
            return
        scheduler.update_from_output(output, runner_output(scheduler, output))
    pytest.fail("expected a preemption")


def test_preempted_request_resets_computed_tokens_but_keeps_output():
    scheduler = make_scheduler(num_blocks=4, block_size=4)
    request = make_request("a", prompt_len=8, max_tokens=16)
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output))
    produced = request.num_output_tokens

    scheduler._preempt_request(request)
    assert request.status is RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert request.num_output_tokens == produced  # resumes mid-generation
    assert request.num_preemptions == 1


def test_admission_pauses_on_a_step_that_preempted():
    """Admitting into an oversubscribed pool would preempt what was just preempted,
    thrashing instead of draining."""
    scheduler = make_scheduler(num_blocks=4, block_size=4)
    scheduler.add_request(make_request("a", prompt_len=8, max_tokens=16))
    scheduler.add_request(make_request("b", prompt_len=8, max_tokens=16))
    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output))
    scheduler.add_request(make_request("c", prompt_len=8, max_tokens=16))

    for _ in range(6):
        output = scheduler.schedule()
        if output.preempted_req_ids:
            assert output.scheduled_new_reqs == []
            return
        scheduler.update_from_output(output, runner_output(scheduler, output))
    pytest.fail("expected a preemption")


# --- draining --------------------------------------------------------------


def test_a_mixed_workload_drains_completely():
    """R21.1: every admitted request terminates, and the pool returns to empty."""
    scheduler = make_scheduler(
        num_blocks=64, block_size=4, max_num_batched_tokens=64, max_num_seqs=4
    )
    lengths = [4, 12, 8, 20, 4, 16, 8, 4]
    for i, prompt_len in enumerate(lengths):
        scheduler.add_request(
            make_request(f"r{i}", prompt_len=prompt_len, max_tokens=3, arrival_time=i)
        )

    steps = 0
    while scheduler.has_requests() and steps < 500:
        output = scheduler.schedule()
        scheduler.update_from_output(output, runner_output(scheduler, output))
        steps += 1

    assert not scheduler.running
    assert scheduler.get_num_unfinished_requests() == 0
    assert scheduler.get_kv_cache_usage() == 0.0
    assert steps < 500, "workload failed to drain"


def test_the_same_workload_yields_the_same_step_count():
    """C1: total engine steps to drain a workload is part of the contract."""

    def drain() -> tuple[int, list[int]]:
        scheduler = make_scheduler(max_num_batched_tokens=32, max_num_seqs=3)
        for i in range(6):
            scheduler.add_request(
                make_request(f"r{i}", prompt_len=8, max_tokens=3, arrival_time=i)
            )
        per_step: list[int] = []
        steps = 0
        while scheduler.has_requests() and steps < 200:
            output = scheduler.schedule()
            per_step.append(output.total_num_scheduled_tokens)
            scheduler.update_from_output(output, runner_output(scheduler, output))
            steps += 1
        return steps, per_step

    assert drain() == drain()


def test_without_chunked_prefill_a_prompt_waits_for_a_whole_step():
    """A prompt that does not fit in what remains of the budget is deferred whole,
    not split. Note the config raises the budget to max_model_len when chunking is
    off, so the budget here is 32, not 16."""
    scheduler = make_scheduler(
        max_num_batched_tokens=16, enable_chunked_prefill=False, max_model_len=32
    )
    assert scheduler.max_num_scheduled_tokens == 32

    scheduler.add_request(make_request("small", prompt_len=8, max_tokens=1))
    scheduler.add_request(make_request("big", prompt_len=32, arrival_time=1.0))

    output = scheduler.schedule()
    # "small" takes 8, leaving 24 -- less than "big" needs, and it is not split.
    assert output.num_scheduled_tokens == {"small": 8}

    # "big" needs the whole budget, so it waits for "small" to finish rather than
    # being chunked into the leftovers.
    scheduler.update_from_output(output, runner_output(scheduler, output))
    assert scheduler.running == []

    following = scheduler.schedule()
    assert following.num_scheduled_tokens == {"big": 32}


def test_priority_policy_admits_by_priority():
    """R5.6."""
    scheduler = make_scheduler(policy="priority", max_num_seqs=1)
    scheduler.add_request(
        make_request("low", prompt_len=4, arrival_time=0.0, priority=9)
    )
    scheduler.add_request(
        make_request("high", prompt_len=4, arrival_time=1.0, priority=0)
    )

    output = scheduler.schedule()
    assert [r.req_id for r in output.scheduled_new_reqs] == ["high"]


def test_stats_snapshot_reports_the_live_state():
    scheduler = make_scheduler(max_num_seqs=1)
    scheduler.add_request(make_request("a", prompt_len=4))
    scheduler.add_request(make_request("b", prompt_len=4))
    scheduler.schedule()

    stats = scheduler.make_stats()
    assert stats["num_running_reqs"] == 1
    assert stats["num_waiting_reqs"] == 1
    assert stats["num_preemptions"] == 0
    assert 0.0 <= stats["kv_cache_usage"] <= 1.0
