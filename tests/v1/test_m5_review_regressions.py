"""Regressions for defects an adversarial review of M5 confirmed.

Fifteen of them, and the pattern is worth naming: most are places where prose was more
confident than code. A comment describing a rotation that did nothing, a warning saying
padding had been added before refusing the pattern it claimed to have padded, a
docstring promising a seeded request reproducible output when the field was read
nowhere, a test asserting a spread that held for a different reason than the one it
named. The engine was mostly right; the account of it was not.

Each test fails without its fix.
"""

from __future__ import annotations

import httpx
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from pvllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from pvllm.entrypoints.llm import LLM
from pvllm.entrypoints.openai.api_server import build_app
from pvllm.sampling_params import SamplingParams
from pvllm.v1.engine.llm_engine import LLMEngine

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


# --- n > 1 ------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3])
def test_abort_by_client_id_frees_every_child(n):
    """M5a fixed this reasoning in `generate`'s finally and wrote it into the commit
    message -- and never applied it to the abort path. Under `n > 1` the client's id
    names no engine request, so the abort freed nothing and the children ran to
    completion holding KV. Upstream keeps a parent map for exactly this.

    Driven through `LLMEngine` because stepping is explicit there: under the virtual
    clock `AsyncLLM`'s output handler drains a request before the consumer sees its
    first chunk, so there would be nothing left to abort.
    """
    engine = LLMEngine(EngineArgs(**BASE).create_engine_config())
    try:
        engine.add_request(
            "client-request",
            "a prompt",
            SamplingParams(n=n, max_tokens=400, ignore_eos=True),
        )
        engine.step()
        core = engine.engine_core.engine_core
        assert len(core.scheduler.requests) == n

        engine.abort_request(["client-request"])
        assert core.get_num_unfinished_requests() == 0
        assert engine.output_processor.request_states == {}
        assert engine.output_processor.parent_requests == {}
        assert core.scheduler.kv_cache_manager.block_pool.get_usage() == 0.0
    finally:
        engine.shutdown()


def test_the_frontend_expands_a_parent_id_into_its_children():
    """The unit the abort path turns on: what the frontend hands the core is not
    what the client handed the frontend."""
    engine = LLMEngine(EngineArgs(**BASE).create_engine_config())
    try:
        engine.add_request(
            "client-request", "a prompt", SamplingParams(n=3, max_tokens=8)
        )
        # `child_requests` is a set, so the order is not the contract -- the
        # membership is.
        assert sorted(engine.output_processor.abort_requests(["client-request"])) == [
            "0_client-request",
            "1_client-request",
            "2_client-request",
        ]
        # An id that names nothing expands to nothing rather than to itself.
        assert engine.output_processor.abort_requests(["nobody"]) == []
    finally:
        engine.shutdown()


def test_an_abort_the_core_never_held_writes_no_trace_record(tmp_path):
    """R19.3. The core traced an `aborted` event for the client id under `n > 1`,
    which it never held -- a trace claiming an abort that did not happen is worse
    than one that is silent, because it is read as evidence."""
    import json

    path = tmp_path / "run.jsonl"
    engine = LLMEngine(EngineArgs(**BASE, trace_path=str(path)).create_engine_config())
    try:
        engine.add_request(
            "client-request",
            "a prompt",
            SamplingParams(n=2, max_tokens=400, ignore_eos=True),
        )
        engine.step()
        engine.abort_request(["client-request"])
    finally:
        engine.shutdown()

    aborted = {
        record["request_id"]
        for record in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
        if record.get("event") == "aborted"
    }
    assert aborted == {"0_client-request", "1_client-request"}


async def test_the_n_histogram_observes_the_parent_once():
    """C6. A child's params always say `n=1` -- that is what makes it one engine
    request -- so reading them put `n` observations of 1 into `vllm:request_params_n`
    instead of one observation of `n`."""
    app = build_app(
        AsyncEngineArgs(**{**BASE, "served_model_name": "m"}).create_engine_config(),
        registry=(registry := CollectorRegistry()),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/v1/completions",
            json={"model": "m", "prompt": "hi", "n": 3, "max_tokens": 4},
        )
        await client.get("/metrics")

    exported = generate_latest(registry).decode()

    def value(suffix: str) -> float:
        prefix = f"vllm:request_params_n_{suffix}"
        line = next(item for item in exported.splitlines() if item.startswith(prefix))
        return float(line.split()[1])

    # One request, of n=3 -- not three requests of n=1.
    assert value("count") == 1.0
    assert value("sum") == 3.0


def test_a_client_seed_reproduces_a_completion():
    """`SamplingParams.seed` was read nowhere in the engine, while
    `ParentRequest._child_params` offset it per child under a comment claiming that
    is what stops `n` children being identical. The field is real now: a seed means
    *this* completion, so the request id drops out of the derivation."""
    llm = LLM(**BASE)
    try:
        first, second = llm.generate(
            ["x", "x"],
            [
                SamplingParams(max_tokens=8, seed=1),
                SamplingParams(max_tokens=8, seed=1),
            ],
        )
        other = llm.generate(["x"], SamplingParams(max_tokens=8, seed=2))[0]
        assert first.outputs[0].text == second.outputs[0].text
        assert first.outputs[0].text != other.outputs[0].text
        # And a seeded parent's children are still distinct, which is what the
        # per-child offset is for.
        parallel = llm.generate(["y"], SamplingParams(n=3, max_tokens=8, seed=5))[0]
        assert len({item.text for item in parallel.outputs}) == 3
    finally:
        llm.shutdown()


# --- pooling ----------------------------------------------------------------


def test_an_empty_document_is_refused_rather_than_admitted():
    """A zero-token pooling request was admitted, scheduled with nothing to compute,
    and never advanced -- and a pooling request has no sampled token to end it, so it
    hung the engine rather than merely stalling. The generation path has carried this
    guard, with a comment explaining it, since M1."""
    llm = LLM(**BASE)
    try:
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            llm.embed([[]])
    finally:
        llm.shutdown()


async def test_the_embeddings_endpoint_routes_the_adapter():
    """R16.1. It returned 200 with `model: adapter-a` while the engine request
    carried no adapter, so the corpus landed in the base model's prefix-cache
    partition and cost no adapter memory."""
    app = build_app(
        AsyncEngineArgs(
            **BASE,
            served_model_name="m",
            enable_lora=True,
            max_loras=2,
            lora_modules=["adapter-a=/x"],
        ).create_engine_config(),
        registry=CollectorRegistry(),
    )
    core = app.state.server.engine.engine_core.engine_core
    seen: dict[str, object] = {}
    original = core.add_request

    def spy(request):
        seen[request.request_id] = request.lora_request
        return original(request)

    core.add_request = spy
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/embeddings", json={"model": "adapter-a", "input": "hello"}
        )
    assert response.status_code == 200
    adapter = next(iter(seen.values()))
    assert adapter is not None and adapter.lora_name == "adapter-a"


async def test_a_failed_document_does_not_leave_its_siblings_running():
    """R2.4. `gather` propagates the first failure but waits for every task, so a 400
    was returned while the good documents were still prefilling -- and their
    generators were never closed, so the abort that returns their blocks never ran."""
    app = build_app(
        AsyncEngineArgs(
            **{**BASE, "served_model_name": "m", "max_num_batched_tokens": 32}
        ).create_engine_config(),
        registry=CollectorRegistry(),
    )
    core = app.state.server.engine.engine_core.engine_core
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/embeddings",
            json={"model": "m", "input": [[7] * 499, [7] * 600]},
        )
    assert response.status_code == 400
    # Nothing left behind the moment the error is returned.
    assert list(core.scheduler.requests) == []


# --- hybrid KV groups -------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "pipeline_parallel_size"),
    [
        ("dense-8b", 1),
        ("dense-8b", 7),
        ("dense-0.6b", 5),
        ("hybrid-4b", 1),
        ("hybrid-4b", 2),
    ],
)
def test_the_startup_number_is_the_pool_the_scheduler_gets(
    model, pipeline_parallel_size
):
    """They sized the pool independently and agreed only when the layer count divided
    evenly. The startup line is what a capacity plan reads, and R10.6's "no request
    could ever be served" guard runs on it -- so a config that could not fit a single
    request passed startup and then hung forever with no error and no log line."""
    llm = LLM(
        model=model,
        device_card="datacenter-80gb",
        block_size=16,
        max_model_len=2048,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        pipeline_parallel_size=pipeline_parallel_size,
        disable_log_stats=True,
    )
    try:
        core = llm.llm_engine.engine_core.engine_core
        profile = core.executor.driver_worker.memory_profile
        assert profile is not None
        assert core.kv_cache_config.num_blocks == profile.num_gpu_blocks
    finally:
        llm.shutdown()


def test_a_layer_pattern_that_does_not_divide_evenly_still_starts():
    """The grouping logged "1 padding layer(s) added" and then refused the pattern,
    because no padding was ever added. Upstream allows a short group and sizes the
    pool from the longest one."""
    llm = LLM(
        model="hybrid-4b",
        device_card="datacenter-80gb",
        block_size=16,
        max_model_len=2048,
        pipeline_parallel_size=2,
        disable_log_stats=True,
    )
    try:
        groups = llm.llm_engine.engine_core.engine_core.kv_cache_config.kv_cache_groups
        # Unequal group lengths are fine; unequal *per-layer* page sizes are not.
        assert len({len(group.layer_names) for group in groups}) > 1
        assert len({group.kv_cache_spec.page_size_bytes for group in groups}) == 1
    finally:
        llm.shutdown()


def test_the_ownership_oracle_runs_for_a_windowed_model_and_does_not_cry_wolf(
    monkeypatch,
):
    """R8.3, both halves. The worker read the config's caching *request* rather than
    the pool's effective setting, so the oracle was skipped for every windowed model
    -- and when it did run it raised "the block pool allocated it twice" for a block
    the manager had legitimately handed back, naming the wrong component entirely."""
    monkeypatch.setenv("PVLLM_DEBUG_INVARIANTS", "1")
    llm = LLM(
        model="hybrid-4b",
        device_card="datacenter-80gb",
        max_model_len=2048,
        block_size=16,
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        num_gpu_blocks_override=1000,
        enable_prefix_caching=False,
        disable_log_stats=True,
        seed=5,
    )
    try:
        runner = (
            llm.llm_engine.engine_core.engine_core.executor.driver_worker.model_runner
        )
        # Effective, not requested: a windowed group turns caching off pool-wide.
        assert runner.block_tables.enable_caching is False
        assert any(runner.block_tables.windowed_groups)
        engine = llm.llm_engine
        for index in range(2):
            engine.add_request(
                f"r{index}", [9 + index] * 1200, SamplingParams(max_tokens=200)
            )
        while engine.has_unfinished_requests():
            engine.step()
    finally:
        llm.shutdown()


def test_a_kv_connector_refuses_a_card_driven_window_too():
    """The connector checked `--sliding-window` only, so M5c's card-driven window
    slipped past: it started cleanly, issued no lookup ever, and reported zero hits
    forever."""
    with pytest.raises(NotImplementedError, match="sliding-window"):
        LLM(
            model="hybrid-4b",
            device_card="datacenter-80gb",
            max_model_len=2048,
            enable_prefix_caching=True,
            disable_log_stats=True,
            kv_transfer_config={"kv_connector": "SimSharedStoreConnector"},
        )


# --- data parallelism -------------------------------------------------------


def test_reset_prefix_cache_reaches_every_replica():
    """`all()` over a generator short-circuits: it wiped the replicas before the
    first refusal, skipped the ones after it, and reported failure for the whole
    deployment -- leaving an operator told nothing happened with half the cache
    gone."""
    engine = LLMEngine(
        EngineArgs(**{**BASE, "data_parallel_size": 3}).create_engine_config()
    )
    try:
        client = engine.engine_core
        calls: list[int] = []

        for index, core in enumerate(client.engine_cores):
            # Replica 0 refuses, as a busy replica does. `all()` over a generator
            # stopped there, so replicas 1 and 2 were never asked -- and the caller
            # was told the reset failed while replica 0's cache was already gone.
            def spy(index=index):
                calls.append(index)
                return index != 0

            core.reset_prefix_cache = spy

        assert engine.engine_core.reset_prefix_cache() is False
        assert sorted(calls) == [0, 1, 2]
    finally:
        engine.shutdown()


def test_the_shared_kv_store_is_not_counted_once_per_replica():
    """C6. Every replica's connector resolves to the same process-global store, so
    summing their readings reported `data_parallel_size` times the transfers that
    happened -- straight into `vllm:external_prefix_cache_*`."""
    from pvllm.sim.kv_store import get_store, reset_stores

    reset_stores()
    engine = LLMEngine(
        EngineArgs(
            **{**BASE, "data_parallel_size": 4},
            enable_prefix_caching=True,
            kv_transfer_config={
                "kv_connector": "SimSharedStoreConnector",
                "kv_connector_extra_config": {"store_name": "shared"},
            },
        ).create_engine_config()
    )
    try:
        for index in range(8):
            engine.add_request(
                f"r{index}",
                "a shared prefix long enough to fill several blocks " * 3,
                SamplingParams(max_tokens=4),
            )
        while engine.has_unfinished_requests():
            engine.step()
        stats = engine.make_stats()
        assert stats["external_prefix_cache_queries"] == get_store("shared").num_lookups
        assert stats["external_prefix_cache_hits"] == get_store("shared").num_hits
    finally:
        engine.shutdown()
        reset_stores()


def test_the_benchmark_runner_accepts_a_data_parallel_deployment():
    """It asserted `InprocClient` by name, so every `pvllm bench ... -dp N` run built
    all N replicas and then died with an error blaming multiprocessing the operator
    never enabled."""
    from pvllm.benchmarks.lib.runner import BenchRequest, run_workload

    llm = LLM(**{**BASE, "data_parallel_size": 2})
    try:
        result = run_workload(
            llm,
            [BenchRequest(prompt_token_ids=[7] * 32, max_tokens=8) for _ in range(4)],
        )
        assert len(result.finished) == 4
    finally:
        llm.shutdown()


async def test_the_debug_endpoints_serve_a_data_parallel_deployment():
    """They rejected DP at app construction with a NotImplementedError telling the
    operator to unset an environment variable they never set."""
    app = build_app(
        AsyncEngineArgs(
            **{**BASE, "served_model_name": "m", "data_parallel_size": 2}
        ).create_engine_config(),
        registry=CollectorRegistry(),
        enable_debug_endpoints=True,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/debug/scheduler")
    assert response.status_code == 200


def test_the_router_does_not_claim_a_rotation_it_does_not_perform():
    """The scan start was rotated per request under a comment, a commit message and a
    test all saying it was what spread a burst. The in-flight counter already breaks
    the tie, so the rotation was inert and the test held for a different reason than
    the one it named -- it passed identically with the rotation removed."""
    engine = LLMEngine(
        EngineArgs(
            **{**BASE, "data_parallel_size": 4, "max_num_seqs": 2}
        ).create_engine_config()
    )
    try:
        client = engine.engine_core
        # Not a multiple of the replica count: rotating once per request would leave
        # `scan_start` at 10 % 4 == 2, so a round number would have made this
        # assertion blind to the very thing it pins.
        for index in range(10):
            engine.add_request(
                f"r{index}", f"prompt {index}", SamplingParams(max_tokens=8)
            )
        # The spread is real; the reason is the in-flight counter, and the scan start
        # never moves.
        assert client.engine_inflight == [3, 3, 2, 2]
        assert client.scan_start == 0
    finally:
        engine.shutdown()
