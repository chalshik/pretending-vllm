"""The engine core, sync engine, and async engine. R4, R2.4, R11.5."""

from __future__ import annotations

import asyncio

import pytest

from pvllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from pvllm.sampling_params import RequestOutputKind, SamplingParams
from pvllm.v1.engine.async_llm import AsyncLLM
from pvllm.v1.engine.core_client import EngineCoreClient
from pvllm.v1.engine.llm_engine import LLMEngine
from pvllm.v1.executor.uniproc_executor import UniProcExecutor

BASE = {
    "model": "dense-0.6b",
    "max_model_len": 512,
    "block_size": 16,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 4,
    "enable_prefix_caching": False,
    "device_card": "workstation-24gb",
    "disable_log_stats": True,
}


def make_engine(**overrides) -> LLMEngine:
    return LLMEngine.from_engine_args(EngineArgs(**{**BASE, **overrides}))


def make_async_engine(**overrides) -> AsyncLLM:
    return AsyncLLM.from_engine_args(AsyncEngineArgs(**{**BASE, **overrides}))


def run_to_completion(engine: LLMEngine, limit: int = 500) -> dict:
    final = {}
    steps = 0
    while engine.has_unfinished_requests() and steps < limit:
        for output in engine.step():
            if output.finished:
                final[output.request_id] = output
        steps += 1
    return final


# --- the sync engine -------------------------------------------------------


def test_a_request_produces_text():
    engine = make_engine()
    engine.add_request("r0", "hello", SamplingParams(max_tokens=8))
    final = run_to_completion(engine)

    output = final["r0"]
    assert output.finished
    assert output.outputs[0].text
    assert len(output.outputs[0].token_ids) == 8
    engine.shutdown()


def test_generated_text_is_reproducible():
    """R19.2 end to end: one seed reproduces the whole run."""

    def run() -> str:
        engine = make_engine(seed=99)
        engine.add_request("r0", "hello", SamplingParams(max_tokens=10))
        text = run_to_completion(engine)["r0"].outputs[0].text
        engine.shutdown()
        return text

    assert run() == run()


def test_a_different_seed_gives_different_text():
    def run(seed: int) -> str:
        engine = make_engine(seed=seed)
        engine.add_request("r0", "hello", SamplingParams(max_tokens=10))
        text = run_to_completion(engine)["r0"].outputs[0].text
        engine.shutdown()
        return text

    assert run(1) != run(2)


def test_truncation_reports_length_not_stop():
    """The distinction a client uses to decide whether to continue. Emitting EOS on
    the final allowed token would report `stop` for every request."""
    engine = make_engine()
    engine.add_request("r0", "hi", SamplingParams(max_tokens=6))
    output = run_to_completion(engine)["r0"]
    assert output.outputs[0].finish_reason == "length"
    assert len(output.outputs[0].token_ids) == 6
    engine.shutdown()


def test_a_model_that_stops_early_reports_stop():
    engine = make_engine(output_length_policy="fixed", output_length_fixed=4)
    engine.add_request("r0", "hi", SamplingParams(max_tokens=100))
    output = run_to_completion(engine)["r0"]
    assert output.outputs[0].finish_reason == "stop"
    assert len(output.outputs[0].token_ids) == 4
    engine.shutdown()


def test_concurrent_requests_all_complete():
    engine = make_engine()
    for i in range(6):
        engine.add_request(f"r{i}", f"prompt {i}", SamplingParams(max_tokens=5))
    final = run_to_completion(engine)
    assert len(final) == 6
    assert all(output.finished for output in final.values())
    engine.shutdown()


def test_a_stop_string_ends_the_request():
    """R11.5: invisible to the scheduler, so the frontend ends it and aborts in the
    core -- otherwise it keeps generating and holding blocks."""
    # Generation is seeded, so the same prompt produces the same text; take a slice
    # of it and use that as the stop string.
    baseline = make_engine()
    baseline.add_request("r0", "hi", SamplingParams(max_tokens=40))
    text = run_to_completion(baseline)["r0"].outputs[0].text
    baseline.shutdown()
    marker = text[8:12]

    engine = make_engine()
    engine.add_request("r0", "hi", SamplingParams(max_tokens=40, stop=[marker]))
    output = run_to_completion(engine)["r0"]

    assert output.outputs[0].finish_reason == "stop"
    assert marker not in output.outputs[0].text
    assert len(output.outputs[0].text) < len(text)
    # The core was told, so nothing is left holding blocks.
    assert engine.make_stats()["kv_cache_usage"] == 0.0
    engine.shutdown()


def test_aborting_frees_capacity():
    """R2.4."""
    engine = make_engine()
    engine.add_request("r0", "hello", SamplingParams(max_tokens=100))
    engine.step()
    assert engine.make_stats()["kv_cache_usage"] > 0

    engine.abort_request(["r0"])
    assert engine.make_stats()["kv_cache_usage"] == 0.0
    assert not engine.has_unfinished_requests()
    engine.shutdown()


def test_context_length_errors_before_admission():
    """R2.5: the client gets an error, not a request that queues and never fits."""
    engine = make_engine(max_model_len=128)
    with pytest.raises(ValueError, match="maximum context length"):
        engine.add_request("r0", list(range(200)), SamplingParams(max_tokens=4))
    with pytest.raises(ValueError, match="maximum context length"):
        engine.add_request("r1", list(range(100)), SamplingParams(max_tokens=100))
    engine.shutdown()


def test_an_empty_prompt_is_rejected():
    engine = make_engine()
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        engine.add_request("r0", [], SamplingParams(max_tokens=4))
    engine.shutdown()


def test_logprobs_beyond_the_configured_max_are_rejected():
    engine = make_engine(max_logprobs=5)
    with pytest.raises(ValueError, match="exceeds the max allowed"):
        engine.add_request("r0", "hi", SamplingParams(max_tokens=4, logprobs=20))
    engine.shutdown()


def test_token_ids_can_be_supplied_directly():
    """R3.3: trace replay must not depend on the tokenizer reproducing the original
    tokenization."""
    engine = make_engine()
    engine.add_request("r0", [10, 11, 12, 13], SamplingParams(max_tokens=4))
    output = run_to_completion(engine)["r0"]
    assert output.prompt_token_ids == [10, 11, 12, 13]
    assert output.prompt is None
    engine.shutdown()


# --- clock ownership (R19.1, R4.4) -----------------------------------------


def test_the_engine_core_owns_the_clock_and_stamps_outputs():
    engine = make_engine()
    client = engine.engine_core
    start = client.clock_time
    engine.add_request("r0", "hello", SamplingParams(max_tokens=6))
    run_to_completion(engine)
    assert client.clock_time > start
    engine.shutdown()


def test_stats_declare_that_durations_are_modeled():
    """R12.4: discoverable, not implied."""
    engine = make_engine()
    stats = engine.make_stats()
    assert stats["durations_are_modeled"] is True
    assert stats["clock_mode"] == "virtual"
    engine.shutdown()


def collect_finished(engine: LLMEngine, limit: int = 500) -> list:
    """Drain, keeping every step's finished-request stats rather than the last."""
    finished = []
    steps = 0
    while engine.has_unfinished_requests() and steps < limit:
        engine.step()
        finished.extend(engine.last_iteration_stats.finished_requests)
        steps += 1
    return finished


def test_ttft_splits_into_queue_wait_and_prefill():
    """R12.1. Two requests with the same TTFT can call for opposite responses -- one
    queued for 200ms and prefilled in 5, the other prefilled for 205 -- and only the
    split distinguishes them. The interval comes from the SCHEDULED event, which the
    engine core dates because the scheduler has no clock (R19.1)."""
    engine = make_engine(max_num_seqs=1)
    for index in range(4):
        engine.add_request(
            f"r{index}", f"prompt number {index}", SamplingParams(max_tokens=8)
        )
    finished = collect_finished(engine)
    engine.shutdown()

    assert len(finished) == 4
    # The first request in is admitted immediately; the ones behind it wait.
    assert min(f.queue_time for f in finished) == 0.0
    assert max(f.queue_time for f in finished) > 0.0
    for stats in finished:
        assert stats.queue_time >= 0.0
        assert stats.prefill_time >= 0.0
        # The two halves reconstruct TTFT rather than merely correlating with it.
        assert stats.queue_time + stats.prefill_time == pytest.approx(
            stats.time_to_first_token
        )


def test_a_preempted_request_does_not_report_a_second_queue_wait():
    """Only the *first* admission ends the wait. Taking the later stamp would report
    a queue time longer than the request's whole lifetime, since a preempted request
    is admitted again well after it arrived."""
    engine = make_engine(
        max_num_seqs=4,
        max_model_len=256,
        block_size=8,
        num_gpu_blocks_override=32,
        enable_prefix_caching=False,
    )
    for index in range(4):
        engine.add_request(
            f"r{index}",
            f"a prompt long enough to need several blocks, number {index}",
            SamplingParams(max_tokens=24),
        )
    finished = collect_finished(engine)
    engine.shutdown()

    assert finished
    # Preemption is what makes this test mean anything: without it there is only one
    # SCHEDULED event and nothing to take the wrong one of.
    assert engine.make_stats()["num_preemptions"] > 0, (
        "the workload did not preempt, so this test would pass on an engine that "
        "took the *last* SCHEDULED stamp instead of the first"
    )
    for stats in finished:
        # The queue wait ends at the *first* admission. Taking a later one would put
        # `scheduled_time` after the first token, making prefill_time clamp to zero
        # while queue_time swallowed the prefill -- which is what this pins.
        assert stats.queue_time <= stats.time_to_first_token
        assert stats.queue_time + stats.prefill_time == pytest.approx(
            stats.time_to_first_token
        )


def test_a_custom_executor_cannot_cross_a_process_boundary():
    """Named rather than failing obscurely inside the child: a class defined in a
    test module does not survive the spawn start method, and the failure would
    otherwise be an unpickling error from a process whose traceback nobody sees."""
    config = EngineArgs(**BASE).create_engine_config()
    with pytest.raises(NotImplementedError, match="custom executor_class"):
        EngineCoreClient.make_client(
            config, multiprocess_mode=True, executor_class=UniProcExecutor
        )


# --- the async engine (R4.3) -----------------------------------------------


async def test_async_generation_streams_deltas():
    engine = make_async_engine()
    deltas = []
    async for output in engine.generate(
        "hello",
        SamplingParams(max_tokens=8, output_kind=RequestOutputKind.DELTA),
        "s0",
    ):
        deltas.append(output.outputs[0].text)

    assert len(deltas) == 8
    assert "".join(deltas)
    engine.shutdown()


async def test_async_cumulative_output_grows():
    engine = make_async_engine()
    seen = ""
    async for output in engine.generate("hello", SamplingParams(max_tokens=6), "s0"):
        current = output.outputs[0].text
        assert current.startswith(seen)
        seen = current
    engine.shutdown()


async def test_async_requests_run_concurrently():
    engine = make_async_engine()

    async def collect(request_id: str) -> str:
        text = ""
        async for output in engine.generate(
            "hello", SamplingParams(max_tokens=6), request_id
        ):
            text = output.outputs[0].text
        return text

    results = await asyncio.gather(*(collect(f"s{i}") for i in range(3)))
    assert len(results) == 3
    assert all(results)
    engine.shutdown()


async def test_cancelling_a_generator_frees_its_blocks_within_one_step():
    """R2.4, and the failure this project exists to let a product test for: a
    disconnected client whose request keeps generating and holding blocks."""
    engine = make_async_engine()

    generator = engine.generate("hello", SamplingParams(max_tokens=200), "s0")
    await anext(generator)
    assert (await engine.make_stats())["kv_cache_usage"] > 0

    await generator.aclose()
    assert (await engine.make_stats())["kv_cache_usage"] == 0.0
    assert await engine.get_num_unfinished_requests() == 0
    engine.shutdown()


async def test_explicit_abort_removes_the_request():
    engine = make_async_engine()
    generator = engine.generate("hello", SamplingParams(max_tokens=200), "s0")
    await anext(generator)

    await engine.abort("s0")
    assert (await engine.make_stats())["kv_cache_usage"] == 0.0
    await generator.aclose()
    engine.shutdown()


async def test_readiness_reports_true_only_after_startup():
    """R2.7. True by construction: the core runs load and profiling in its
    constructor, so an engine that exists is one that finished starting up."""
    engine = make_async_engine()
    assert await engine.is_ready()
    await engine.check_health()
    engine.shutdown()


async def test_a_stop_string_aborts_in_the_core_on_the_async_path_too():
    """R11.5 has two implementations and had one test.

    `LLMEngine.step()` and `AsyncLLM._run_output_handler` each detect a stop string in
    the frontend and must then tell the core, because the scheduler does not know the
    request ended. Only the synchronous one was covered -- and the async one is the
    path `pvllm serve` and every HTTP endpoint take. Deleting its `abort_requests`
    call left the whole suite green while a served request kept generating into blocks
    nobody would ever read.
    """
    # The marker comes from the async engine's own output, under the SAME request id.
    # R19.2 derives each request's RNG from `(seed, request_id)`, so text is
    # reproducible per id and differs between ids -- take a slice of a run under a
    # different id and it simply never appears, the request ends on length, and the
    # test passes while proving nothing.
    baseline = make_async_engine()
    try:
        text = ""
        async for output in baseline.generate(
            "hi", SamplingParams(max_tokens=40), "s0"
        ):
            text = output.outputs[0].text
    finally:
        baseline.shutdown()
    marker = text[8:12]

    engine = make_async_engine()
    try:
        final = None
        async for output in engine.generate(
            "hi", SamplingParams(max_tokens=40, stop=[marker]), "s0"
        ):
            final = output

        assert final is not None
        assert final.outputs[0].finish_reason == "stop"
        assert marker not in final.outputs[0].text
        assert len(final.outputs[0].text) < len(text)

        # The core was told. Without the abort the request stays running there.
        assert (await engine.make_stats())["kv_cache_usage"] == 0.0
    finally:
        engine.shutdown()
