"""Regressions for defects an adversarial review of M3 confirmed.

Each of these shipped, passed the suite at the time, and was found by reading the code
against its own claims rather than by a failing test. The tests exist so the same
defect cannot return quietly; the docstrings say what the original symptom was,
because that is the part that makes a future failure here legible.
"""

from __future__ import annotations

import json

import pytest
from prometheus_client import CollectorRegistry

from pvllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from pvllm.sampling_params import SamplingParams
from pvllm.v1.engine.llm_engine import LLMEngine

BASE = {
    "model": "tiny-test",
    "max_model_len": 256,
    "block_size": 8,
    "max_num_batched_tokens": 64,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
}


# --- Tier A: the defaultdict phantom (finding 5) ---------------------------


def test_a_failed_allocation_leaves_no_phantom_holder():
    """`get_num_blocks_to_allocate` used to index a defaultdict, inserting an empty
    entry for any request merely *considered* for allocation. `allocate_slots` calls
    it before deciding whether the request fits, so every rejected admission left a
    phantom behind -- and `get_num_common_prefix_blocks` counts holders, so the
    common-prefix count collapsed to zero for the rest of the run.

    The visible symptom was `/debug/cost_model` reporting zero shared blocks while
    `/debug/blocks` showed five requests holding the same nine.
    """
    from pvllm.v1.core.kv_cache_manager import KVCacheManager
    from pvllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from pvllm.v1.request import Request

    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=32, dtype="bfloat16", dtype_bytes=2
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=8,
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
        ),
        max_model_len=128,
    )

    def request(name: str, prompt_len: int) -> Request:
        return Request(
            request_id=name,
            prompt_token_ids=list(range(prompt_len)),
            sampling_params=SamplingParams(max_tokens=4),
            arrival_time=0.0,
        )

    manager.allocate_slots(request("a", 8), 8)
    manager.allocate_slots(request("b", 8), 8)
    before = manager.get_num_common_prefix_blocks("a")

    # A request far too large for what is left. It must be rejected *and* leave no
    # trace -- the pool is the only state a failed attempt may touch, and it must
    # touch none of it.
    assert manager.allocate_slots(request("c", 400), 400) is None

    assert manager.get_num_common_prefix_blocks("a") == before
    single = manager.coordinator.single_type_managers[0]
    assert "c" not in single.req_to_blocks


# --- benchmarks: double-counted finishers (finding 10) ---------------------


def test_a_step_with_no_output_does_not_re_report_the_last_one():
    """`LLMEngine.step()` returned early on a barren step without replacing
    `last_iteration_stats`, so a caller accumulating `finished_requests` across steps
    re-counted the previous step's finishers every time. Every benchmark number
    derived from them was inflated, and inflated most at low concurrency -- exactly
    the cells a capacity decision is read off.
    """
    engine = LLMEngine.from_engine_args(
        EngineArgs(**{**BASE, "max_num_seqs": 1, "max_num_batched_tokens": 16})
    )
    try:
        for index in range(3):
            engine.add_request(
                f"r{index}",
                list(range(60 + index)),
                SamplingParams(max_tokens=3),
            )
        finished = []
        for _ in range(500):
            if not engine.has_unfinished_requests():
                break
            engine.step()
            finished.extend(engine.last_iteration_stats.finished_requests)

        assert len(finished) == 3, f"counted {len(finished)} finishers for 3 requests"
    finally:
        engine.shutdown()


def test_synthetic_prompts_do_not_alias():
    """`synthetic_prompt` promised distinct prompts and repeated exactly every
    `vocab_size - 200` requests. At tiny-test's 1024-token vocabulary that is every
    824, so a `--num-prompts 1000` throughput run silently became a prefix-cache
    benchmark reporting a hit rate nobody asked for."""
    from pvllm.benchmarks.lib.runner import synthetic_prompt

    vocab = 1024
    prompts = {tuple(synthetic_prompt(index, 16, vocab)) for index in range(3000)}
    assert len(prompts) == 3000


def test_bench_latency_reports_per_iteration_steps(tmp_path):
    """`num_steps` came from the engine's lifetime counter while `bench latency`
    reuses one engine across iterations, so every iteration after the first reported
    the sum of all of them."""
    from pvllm.entrypoints.cli.main import main

    out = tmp_path / "latency.json"
    main(
        [
            "bench",
            "latency",
            "--model",
            "tiny-test",
            "--device-card",
            "tiny-2gb",
            "--max-model-len",
            "256",
            "--disable-log-stats",
            "--input-len",
            "32",
            "--output-len",
            "8",
            "--batch-size",
            "2",
            "--num-iters",
            "3",
            "--output-json",
            str(out),
        ]
    )
    payload = json.loads(out.read_text())
    # One iteration's worth: 8 output tokens over a batch of 2 is well under 30
    # steps. The lifetime counter after three iterations would be triple.
    assert payload["num_steps"] < 30, payload["num_steps"]


# --- metrics: the phase split must reconstruct e2e (finding 11) ------------


def test_queue_and_inference_do_not_double_count_the_wait():
    """`inference_time` measured from arrival, so it included the queue wait. Once
    `queue_time` became real in M3d, a dashboard stacking queue + inference reported
    more time than the request took."""
    engine = LLMEngine.from_engine_args(EngineArgs(**{**BASE, "max_num_seqs": 1}))
    try:
        for index in range(3):
            engine.add_request(
                f"r{index}", f"prompt {index}", SamplingParams(max_tokens=4)
            )
        finished = []
        for _ in range(500):
            if not engine.has_unfinished_requests():
                break
            engine.step()
            finished.extend(engine.last_iteration_stats.finished_requests)

        assert finished
        assert any(stats.queue_time > 0 for stats in finished), "nothing queued"
        for stats in finished:
            assert stats.queue_time + stats.inference_time == pytest.approx(
                stats.e2e_latency
            )
    finally:
        engine.shutdown()


# --- metrics: the C6 blind spot (finding 8) --------------------------------


def test_request_success_declares_its_label_before_anything_finishes():
    """The label only appeared once a request finished, so the C6 golden recorded
    the family with no labels -- and renaming `finished_reason` passed the whole
    conformance suite while emptying every dashboard panel querying it."""
    from pvllm.entrypoints.openai.api_server import build_app

    registry = CollectorRegistry()
    build_app(
        EngineArgs(**BASE).create_engine_config(),
        registry=registry,
    )
    families = {metric.name: metric for metric in registry.collect()}
    samples = families["vllm:request_success"].samples
    assert samples, "no series exists before the first request finishes"
    assert "finished_reason" in samples[0].labels


# --- multiprocess mode reaches the HTTP layer (findings 1, 3, 14, 15, 16) ---


async def test_the_openai_surface_works_over_a_process_boundary(monkeypatch):
    """Five separate defects met here: two `assert isinstance(..., InprocClient)`
    guards left in the serving layer, and `/metrics` plus `/reset_prefix_cache`
    calling synchronous client methods the async multiprocess client refuses. Every
    completion returned 500 in the exact configuration the README documents."""
    monkeypatch.setenv("PVLLM_ENABLE_V1_MULTIPROCESSING", "1")

    import httpx

    from pvllm.entrypoints.openai.api_server import build_app

    config = AsyncEngineArgs(
        **{**BASE, "served_model_name": "m"}
    ).create_engine_config()
    app = build_app(config, registry=CollectorRegistry())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        completion = await client.post(
            "/v1/completions",
            json={"model": "m", "prompt": "hello", "max_tokens": 4},
        )
        assert completion.status_code == 200, completion.text
        assert completion.json()["choices"][0]["text"]

        chat = await client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
        )
        assert chat.status_code == 200, chat.text

        metrics = await client.get("/metrics")
        assert metrics.status_code == 200, metrics.text
        assert "vllm:num_requests_running" in metrics.text

        reset = await client.post("/reset_prefix_cache")
        assert reset.status_code == 200, reset.text

    # The lifespan is not running under ASGITransport, so the engine child process
    # has to be reaped explicitly or it outlives the test.
    app.state.server.shutdown()


def test_debug_endpoints_refuse_multiprocess_at_startup(monkeypatch):
    """They used to accept the flag and 500 from every route -- a failure that shows
    up only when someone reaches for the debugging tools, which is the worst possible
    moment to learn the configuration is unsupported."""
    monkeypatch.setenv("PVLLM_ENABLE_V1_MULTIPROCESSING", "1")

    from pvllm.entrypoints.openai.api_server import build_app

    with pytest.raises(NotImplementedError, match="in-process engine core"):
        build_app(
            AsyncEngineArgs(**BASE).create_engine_config(),
            registry=CollectorRegistry(),
            enable_debug_endpoints=True,
        )


# --- IPC: a malformed frame must not deafen the engine (finding 2) ---------


def test_an_undecodable_frame_does_not_kill_the_input_thread():
    """A single frame the core could not decode raised out of the socket thread,
    killing it. The process kept running with nobody draining the input socket -- an
    engine that is not dead but permanently deaf, whose symptom was a utility call
    timing out thirty seconds later naming the wrong cause."""
    import msgspec

    from pvllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
    from pvllm.v1.engine.core_client import EngineCoreClient

    config = EngineArgs(**BASE).create_engine_config()
    client = EngineCoreClient.make_client(config, multiprocess_mode=True)
    try:
        # Well-formed frame, malformed payload.
        client.input_socket.send_multipart(
            [EngineCoreRequestType.ADD.value, msgspec.msgpack.encode({"junk": True})]
        )
        # The engine must still be serving.
        client.add_request(
            EngineCoreRequest(
                request_id="r0",
                prompt_token_ids=[1, 2, 3],
                sampling_params=SamplingParams(max_tokens=3),
            )
        )
        tokens: list[int] = []
        for _ in range(200):
            outputs = client.get_output()
            if not outputs:
                break
            for client_outputs in outputs.values():
                for output in client_outputs.outputs:
                    tokens.extend(output.new_token_ids)
            if len(tokens) >= 3:
                break
        assert len(tokens) == 3
    finally:
        client.shutdown()
