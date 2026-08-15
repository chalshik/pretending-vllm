"""The engine core, sync engine, and async engine. R4, R2.4, R11.5."""

from __future__ import annotations

import asyncio

import pytest

from pvllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from pvllm.sampling_params import RequestOutputKind, SamplingParams
from pvllm.v1.engine.async_llm import AsyncLLM
from pvllm.v1.engine.core_client import EngineCoreClient
from pvllm.v1.engine.llm_engine import LLMEngine

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


def test_the_multiprocess_client_names_what_is_missing():
    config = EngineArgs(**BASE).create_engine_config()
    with pytest.raises(NotImplementedError, match="multiprocess engine core"):
        EngineCoreClient.make_client(config, multiprocess_mode=True)


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
    assert engine.make_stats()["kv_cache_usage"] > 0

    await generator.aclose()
    assert engine.make_stats()["kv_cache_usage"] == 0.0
    assert await engine.get_num_unfinished_requests() == 0
    engine.shutdown()


async def test_explicit_abort_removes_the_request():
    engine = make_async_engine()
    generator = engine.generate("hello", SamplingParams(max_tokens=200), "s0")
    await anext(generator)

    await engine.abort("s0")
    assert engine.make_stats()["kv_cache_usage"] == 0.0
    await generator.aclose()
    engine.shutdown()


async def test_readiness_reports_true_only_after_startup():
    """R2.7. True by construction: the core runs load and profiling in its
    constructor, so an engine that exists is one that finished starting up."""
    engine = make_async_engine()
    assert await engine.is_ready()
    await engine.check_health()
    engine.shutdown()
