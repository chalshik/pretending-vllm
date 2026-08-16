"""Parallel sampling: `n > 1`. R11.7.

The engine core has no notion of `n`. Four completions of one prompt are four
requests that share a prompt, fanned out by the frontend and gathered back into one
response -- which is how upstream does it, and the reason this belongs in
`v1/engine/` rather than in the scheduler.

That structure is the point rather than an implementation detail. The children queue
independently, are preempted independently, and share the prompt's KV through the
ordinary prefix cache. A product sending `n=4` gets one response and four times the
decode pressure, and a capacity plan needs to see both.
"""

from __future__ import annotations

import pytest

from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import RequestOutputKind, SamplingParams

BASE = {
    "model": "tiny-test",
    "max_model_len": 512,
    "block_size": 16,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 8,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
    "seed": 3,
}


def test_n_completions_come_back_under_one_request_id():
    llm = LLM(**BASE)
    try:
        outputs = llm.generate(
            ["tell me something", "and another"], SamplingParams(n=4, max_tokens=12)
        )
        assert len(outputs) == 2
        for output in outputs:
            assert len(output.outputs) == 4
            # In index order, whatever order the children actually finished in.
            assert [item.index for item in output.outputs] == [0, 1, 2, 3]
            assert all(item.finish_reason is not None for item in output.outputs)
    finally:
        llm.shutdown()


def test_the_completions_differ_from_each_other():
    """Four identical completions would make the whole feature pointless. Each child
    is its own request id, and sampling is keyed by `(request_id, position)`, so they
    diverge for the same reason two different requests do."""
    llm = LLM(**BASE)
    try:
        output = llm.generate(["one prompt"], SamplingParams(n=6, max_tokens=16))[0]
        texts = [item.text for item in output.outputs]
        assert len(set(texts)) == len(texts)
    finally:
        llm.shutdown()


def test_the_children_are_separate_requests_to_the_engine():
    """Not one request with four heads: `n=4` costs four slots in `max_num_seqs` and
    four requests' worth of decode, which is exactly the capacity effect a product
    needs to see."""
    llm = LLM(**BASE)
    try:
        engine = llm.llm_engine
        engine.add_request("parent", "a prompt", SamplingParams(n=4, max_tokens=16))
        scheduler = engine.engine_core.engine_core.scheduler
        assert len(scheduler.requests) == 4
        assert sorted(scheduler.requests) == [f"{i}_parent" for i in range(4)]
        while engine.has_unfinished_requests():
            engine.step()
    finally:
        llm.shutdown()


def test_the_children_share_the_prompt_through_the_prefix_cache():
    """The one place `n > 1` is cheaper than `n` separate requests, and it comes for
    free: the children have identical prompts, so all but the first hit the cache."""
    llm = LLM(**BASE, enable_prefix_caching=True)
    try:
        prompt = "a prompt long enough to fill more than one block of KV cache indeed"
        llm.generate([prompt], SamplingParams(n=4, max_tokens=8))
        stats = llm.llm_engine.make_stats()
        assert stats["prefix_cache_hits"] > 0
    finally:
        llm.shutdown()


def test_the_same_seed_gives_the_same_n_completions():
    """B4 still holds with the fan-out: same seed, same `n`, byte-identical set."""

    def run() -> list[list[int]]:
        llm = LLM(**BASE)
        try:
            output = llm.generate(["repeat me"], SamplingParams(n=3, max_tokens=10))[0]
            return [list(item.token_ids) for item in output.outputs]
        finally:
            llm.shutdown()

    assert run() == run()


def test_a_seeded_parent_offsets_its_children():
    """Upstream gives child `i` seed `seed + i`, so a seeded request still gets `n`
    *different* completions rather than `n` copies of one."""
    from pvllm.v1.engine.parallel_sampling import ParentRequest

    parent = ParentRequest("r", SamplingParams(n=3, seed=100, max_tokens=4))
    seeds = [parent.child_info(index)[1].seed for index in range(3)]
    assert seeds == [100, 101, 102]

    unseeded = ParentRequest("r", SamplingParams(n=3, max_tokens=4))
    # No seed means the children are interchangeable, so one params object serves
    # all of them rather than `n` copies.
    first = unseeded.child_info(0)[1]
    assert unseeded.child_info(1)[1] is first
    assert first.n == 1


def test_n_equals_one_takes_the_old_path_untouched():
    """The overwhelmingly common case must not pay for the feature: no parent, and
    the engine request id is the client's own."""
    llm = LLM(**BASE)
    try:
        engine = llm.llm_engine
        engine.add_request("solo", "a prompt", SamplingParams(max_tokens=8))
        scheduler = engine.engine_core.engine_core.scheduler
        assert list(scheduler.requests) == ["solo"]
        assert engine.output_processor.request_states["solo"].parent_request is None
        while engine.has_unfinished_requests():
            engine.step()
    finally:
        llm.shutdown()


def test_a_child_that_finishes_early_does_not_report_twice():
    """Children finish at different steps. The first one to finish must not be
    emitted again on every later step, and the parent must not be declared finished
    until the last one lands."""
    llm = LLM(**BASE, output_length_policy="lognormal")
    try:
        engine = llm.llm_engine
        engine.add_request("parent", "a prompt", SamplingParams(n=4, max_tokens=64))
        finished_reports = 0
        aggregate = None
        while engine.has_unfinished_requests():
            for output in engine.step():
                assert output.request_id == "parent"
                if output.finished:
                    finished_reports += 1
                    aggregate = output
        assert finished_reports == 1
        assert aggregate is not None
        assert len(aggregate.outputs) == 4
    finally:
        llm.shutdown()


def test_streaming_passes_each_child_through_as_it_arrives():
    from pvllm.outputs import CompletionOutput
    from pvllm.v1.engine.parallel_sampling import ParentRequest

    parent = ParentRequest(
        "r", SamplingParams(n=2, max_tokens=4, output_kind=RequestOutputKind.DELTA)
    )
    for index in range(2):
        parent.child_info(index)

    chunk = CompletionOutput(index=0, text="a", token_ids=[1], finish_reason=None)
    outputs, finished = parent.collect("0_r", chunk)
    assert outputs == [chunk] and not finished

    done_zero = CompletionOutput(index=0, text="b", token_ids=[2], finish_reason="stop")
    outputs, finished = parent.collect("0_r", done_zero)
    assert outputs == [done_zero] and not finished

    # The same child again: already returned, so nothing goes out twice.
    outputs, finished = parent.collect("0_r", done_zero)
    assert outputs == [] and not finished

    done_one = CompletionOutput(index=1, text="c", token_ids=[3], finish_reason="stop")
    outputs, finished = parent.collect("1_r", done_one)
    assert outputs == [done_one] and finished


@pytest.mark.parametrize("n", [2, 5])
async def test_cancelling_the_consumer_aborts_every_child(n):
    """R2.4 with a fan-out. The client's request id names no engine request at all,
    so aborting it would free nothing -- the children have to be named."""
    from pvllm.engine.arg_utils import AsyncEngineArgs
    from pvllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM(AsyncEngineArgs(**BASE).create_engine_config())
    try:
        stream = engine.generate(
            "a prompt", SamplingParams(n=n, max_tokens=64), "client-request"
        )
        await anext(stream)
        await stream.aclose()

        assert engine.output_processor.num_requests == 0
        core = engine.engine_core.engine_core
        while core.has_requests():
            core.step()
        assert core.get_num_unfinished_requests() == 0
    finally:
        engine.shutdown()
