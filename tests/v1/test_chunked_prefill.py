"""Chunked prefill. R5.4."""

from __future__ import annotations

from tests.v1.test_scheduler import make_request, make_scheduler, runner_output


def test_a_prompt_longer_than_the_budget_splits_across_steps():
    """The defining behaviour: a step's budget bounds work, not admission."""
    scheduler = make_scheduler(max_num_batched_tokens=16, max_model_len=128)
    scheduler.add_request(make_request("big", prompt_len=40, max_tokens=2))

    chunks = []
    for _ in range(4):
        output = scheduler.schedule()
        if "big" not in output.num_scheduled_tokens:
            break
        chunks.append(output.num_scheduled_tokens["big"])
        scheduler.update_from_output(output, runner_output(scheduler, output))

    # 40 tokens over a 16-token budget: 16, 16, 8.
    assert chunks[:3] == [16, 16, 8]


def test_no_token_is_sampled_until_the_prompt_is_complete():
    """A request mid-prefill has no logits to sample from, so the runner returns
    nothing for it -- and the scheduler must not treat that as a finished step."""
    scheduler = make_scheduler(max_num_batched_tokens=16, max_model_len=128)
    request = make_request("big", prompt_len=40, max_tokens=2)
    scheduler.add_request(request)

    first = scheduler.schedule()
    scheduler.update_from_output(first, runner_output(scheduler, first))
    assert request.num_output_tokens == 0
    assert request.is_prefill_chunk


def test_the_budget_is_still_respected_while_chunking():
    """R5.3 holds during chunked prefill too."""
    scheduler = make_scheduler(max_num_batched_tokens=16, max_model_len=128)
    for i in range(3):
        scheduler.add_request(make_request(f"r{i}", prompt_len=40, max_tokens=2))

    for _ in range(6):
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens <= 16
        scheduler.update_from_output(output, runner_output(scheduler, output))


def test_long_prefill_threshold_caps_one_request_share():
    """R5.4. Without it a single very long prompt monopolizes a step and stalls
    every decode behind it."""
    scheduler = make_scheduler(
        max_num_batched_tokens=128, max_model_len=256, long_prefill_token_threshold=8
    )
    scheduler.add_request(make_request("big", prompt_len=64, max_tokens=2))

    output = scheduler.schedule()
    assert output.num_scheduled_tokens["big"] == 8


def test_max_num_partial_prefills_bounds_concurrent_prefills():
    """R5.4. Without the cap a burst of long prompts all start chunking together,
    and each one's first token waits for every other prompt to finish prefilling --
    the batch stays busy while every TTFT gets worse."""
    scheduler = make_scheduler(
        max_num_batched_tokens=16,
        max_model_len=256,
        max_num_seqs=8,
        max_num_partial_prefills=1,
    )
    for i in range(3):
        scheduler.add_request(make_request(f"r{i}", prompt_len=40, max_tokens=2))

    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output))
    # Only one request is mid-prefill, so only one was admitted.
    assert len(scheduler.running) == 1

    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output))
    assert len(scheduler.running) == 1


def test_a_higher_cap_admits_more_prefills():
    scheduler = make_scheduler(
        max_num_batched_tokens=64,
        max_model_len=256,
        max_num_seqs=8,
        max_num_partial_prefills=3,
    )
    for i in range(4):
        scheduler.add_request(make_request(f"r{i}", prompt_len=40, max_tokens=2))

    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output))
    assert 1 < len(scheduler.running) <= 3


def test_the_cap_does_not_block_decodes():
    """Only *prefills* are capped: a request that finished its prompt keeps
    decoding regardless."""
    scheduler = make_scheduler(
        max_num_batched_tokens=64,
        max_model_len=256,
        max_num_seqs=8,
        max_num_partial_prefills=1,
    )
    scheduler.add_request(make_request("short", prompt_len=8, max_tokens=8))
    output = scheduler.schedule()
    scheduler.update_from_output(output, runner_output(scheduler, output))
    assert not scheduler.running[0].is_prefill_chunk

    scheduler.add_request(make_request("next", prompt_len=8, max_tokens=8))
    output = scheduler.schedule()
    # "short" decodes and "next" is admitted: neither is blocked.
    assert "short" in output.num_scheduled_tokens
    assert "next" in output.num_scheduled_tokens


def test_a_chunked_workload_drains():
    """R21.1: every admitted request terminates."""
    scheduler = make_scheduler(
        max_num_batched_tokens=16, max_model_len=256, max_num_seqs=4
    )
    for i in range(4):
        scheduler.add_request(
            make_request(f"r{i}", prompt_len=30 + i * 10, max_tokens=3)
        )

    steps = 0
    while scheduler.has_requests() and steps < 400:
        output = scheduler.schedule()
        scheduler.update_from_output(output, runner_output(scheduler, output))
        steps += 1

    assert not scheduler.running
    assert scheduler.get_kv_cache_usage() == 0.0
    assert steps < 400
