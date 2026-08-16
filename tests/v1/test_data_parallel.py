"""Data parallelism: several engine replicas behind one router. R13.3.

Data parallelism is not sharding. Each replica is a whole engine -- its own weights,
its own device, its own KV pool, its own scheduler -- and a request lives entirely
inside one of them for its whole life.

Three consequences a capacity plan turns on, and they are what these tests pin:
capacity multiplies while per-request latency does not improve; the prefix cache is
partitioned, so a shared preamble is recomputed on every replica that sees one; and
the router picks by load rather than round-robin.
"""

from __future__ import annotations

import pytest

from pvllm.config.parallel import ParallelConfig
from pvllm.engine.arg_utils import EngineArgs
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.v1.engine.llm_engine import LLMEngine

BASE = {
    "model": "tiny-test",
    "max_model_len": 1024,
    "block_size": 16,
    "max_num_batched_tokens": 512,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
    "seed": 2,
}

PREAMBLE = "a long shared system prompt that fills several blocks of KV cache " * 3


def engine(**overrides) -> LLMEngine:
    return LLMEngine(EngineArgs(**{**BASE, **overrides}).create_engine_config())


# --- what it is, and is not -------------------------------------------------


def test_replicas_are_whole_engines_not_shards():
    """`world_size` counts the tensor and pipeline dimensions and not this one: a
    data-parallel replica holds the same weights as its siblings, not a share."""
    config = ParallelConfig(data_parallel_size=4, tensor_parallel_size=2)
    assert config.world_size == 2


def test_each_replica_gets_its_own_pool_and_its_own_clock():
    llm = LLM(**BASE, data_parallel_size=3)
    try:
        client = llm.llm_engine.engine_core
        assert len(client.engine_cores) == 3
        # Separate pools, separate schedulers, separate clocks.
        pools = {
            id(core.scheduler.kv_cache_manager.block_pool)
            for core in client.engine_cores
        }
        clocks = {id(core.clock) for core in client.engine_cores}
        assert len(pools) == 3
        assert len(clocks) == 3
    finally:
        llm.shutdown()


# --- the router -------------------------------------------------------------


def test_a_burst_spreads_across_the_replicas():
    """Without the rotating scan start, a burst arriving at an idle deployment would
    all score equally and land on replica 0."""
    llm_engine = engine(data_parallel_size=4, max_num_seqs=2)
    try:
        for index in range(12):
            llm_engine.add_request(
                f"r{index}", f"prompt {index}", SamplingParams(max_tokens=8)
            )
        assert llm_engine.engine_core.engine_inflight == [3, 3, 3, 3]
    finally:
        llm_engine.shutdown()


def test_a_loaded_replica_is_skipped():
    """The score is load, not position. A replica already holding work is passed over
    for an idle one."""
    llm_engine = engine(data_parallel_size=2, max_num_seqs=8)
    try:
        client = llm_engine.engine_core
        # Pin four requests onto replica 0, then let the router choose.
        for index in range(4):
            request = llm_engine.input_processor.process_inputs(
                f"pinned{index}", "x", SamplingParams(max_tokens=8)
            )
            request.data_parallel_rank = 0
            client.add_request(request)
        assert client.engine_inflight == [4, 0]

        llm_engine.add_request("free", "y", SamplingParams(max_tokens=8))
        assert client.request_to_engine["free"] == 1
    finally:
        llm_engine.shutdown()


def test_a_pinned_rank_is_honoured_and_a_bad_one_is_refused():
    llm_engine = engine(data_parallel_size=3)
    try:
        client = llm_engine.engine_core
        request = llm_engine.input_processor.process_inputs(
            "p0", "x", SamplingParams(max_tokens=2)
        )
        request.data_parallel_rank = 2
        client.add_request(request)
        assert client.request_to_engine["p0"] == 2

        bad = llm_engine.input_processor.process_inputs(
            "p1", "x", SamplingParams(max_tokens=2)
        )
        bad.data_parallel_rank = 9
        with pytest.raises(ValueError, match="out of range"):
            client.add_request(bad)
    finally:
        llm_engine.shutdown()


def test_an_abort_reaches_the_replica_actually_holding_the_request():
    """The deployment has no shared registry of requests: an abort broadcast to every
    replica would be wrong, and one sent to the wrong replica would free nothing."""
    llm_engine = engine(data_parallel_size=4, max_num_seqs=2)
    try:
        client = llm_engine.engine_core
        for index in range(8):
            llm_engine.add_request(
                f"r{index}", f"prompt {index}", SamplingParams(max_tokens=16)
            )
        before = list(client.engine_inflight)
        target = client.request_to_engine["r3"]
        llm_engine.abort_request(["r3"])
        after = list(client.engine_inflight)
        assert after[target] == before[target] - 1
        assert sum(after) == sum(before) - 1
    finally:
        llm_engine.shutdown()


def test_the_in_flight_count_returns_to_zero_when_everything_drains():
    llm_engine = engine(data_parallel_size=3)
    try:
        for index in range(9):
            llm_engine.add_request(
                f"r{index}", f"prompt {index}", SamplingParams(max_tokens=6)
            )
        while llm_engine.has_unfinished_requests():
            llm_engine.step()
        assert llm_engine.engine_core.engine_inflight == [0, 0, 0]
        assert llm_engine.engine_core.request_to_engine == {}
    finally:
        llm_engine.shutdown()


# --- what it buys, and what it costs ----------------------------------------


def test_capacity_multiplies_and_the_clock_takes_the_slowest_replica():
    """The replicas run concurrently on separate devices, so the deployment's elapsed
    time is the slowest replica's and never their sum."""

    def run(data_parallel_size: int) -> tuple[int, float]:
        llm = LLM(**BASE, data_parallel_size=data_parallel_size)
        try:
            llm.generate(
                [f"prompt number {index}" for index in range(16)],
                SamplingParams(max_tokens=16),
            )
            stats = llm.llm_engine.make_stats()
            return stats["step_index"], stats["elapsed"]
        finally:
            llm.shutdown()

    one_steps, one_elapsed = run(1)
    four_steps, four_elapsed = run(4)
    # Four replicas drain the same workload in roughly a quarter of the rounds.
    assert four_steps < one_steps / 2
    assert four_elapsed < one_elapsed / 2


def test_the_prefix_cache_is_partitioned_across_replicas():
    """The most surprising thing about turning DP on: a workload whose shared
    preamble hits almost every time on one engine loses much of that hit rate when
    the router spreads it over replicas that cannot see each other's blocks."""

    def hit_rate(data_parallel_size: int) -> float:
        llm = LLM(
            **BASE, data_parallel_size=data_parallel_size, enable_prefix_caching=True
        )
        try:
            llm.generate(
                [PREAMBLE + f"question {index}" for index in range(16)],
                SamplingParams(max_tokens=16),
            )
            stats = llm.llm_engine.make_stats()
            return stats["prefix_cache_hits"] / max(1, stats["prefix_cache_queries"])
        finally:
            llm.shutdown()

    assert hit_rate(1) > hit_rate(4)


def test_the_stats_show_the_spread_rather_than_averaging_it_away():
    """An imbalance is the failure mode a DP experiment looks for, so it has to
    survive aggregation."""
    llm_engine = engine(data_parallel_size=4, max_num_seqs=2)
    try:
        for index in range(8):
            llm_engine.add_request(
                f"r{index}", f"prompt {index}", SamplingParams(max_tokens=16)
            )
        llm_engine.step()
        stats = llm_engine.make_stats()
        assert stats["data_parallel_size"] == 4
        assert len(stats["per_engine_running"]) == 4
        assert sum(stats["per_engine_running"]) == stats["num_running_reqs"]
        assert 0.0 <= stats["kv_cache_usage"] <= 1.0
    finally:
        llm_engine.shutdown()


def test_every_replica_writes_its_own_trace(tmp_path):
    """A timeline is a property of one device. N replicas sharing a file would
    interleave steps from engines whose clocks are independent -- a trace that reads
    as one engine behaving impossibly."""
    llm = LLM(**BASE, data_parallel_size=3, trace_path=str(tmp_path / "run.jsonl"))
    try:
        llm.generate(["a", "b", "c", "d"], SamplingParams(max_tokens=4))
    finally:
        llm.shutdown()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "run.dp0.jsonl",
        "run.dp1.jsonl",
        "run.dp2.jsonl",
    ]


# --- what it refuses --------------------------------------------------------


def test_a_real_clock_with_replicas_refuses_rather_than_misleading():
    """Stepping the replicas from one process spends their durations in sequence, so
    a real-clock run would take N times as long as the deployment it models -- and
    real mode exists precisely to exercise a client's timeouts."""
    with pytest.raises(NotImplementedError, match="clock-mode"):
        LLM(**BASE, data_parallel_size=2, clock_mode="real")


def test_expert_parallelism_widens_the_group_across_the_replicas():
    """R13.4. This is the part of EP that surprises: the replicas stop being
    independent copies and jointly shard one set of experts, so the EP group spans
    both the tensor and data dimensions -- `ep_size = dp * tp`, which is what
    upstream's `FusedMoEParallelConfig.make` derives after flattening TP across DP."""
    assert (
        ParallelConfig(
            enable_expert_parallel=True, data_parallel_size=2, tensor_parallel_size=2
        ).expert_parallel_size
        == 4
    )
    # `world_size` stays per replica: EP does not change how many devices a replica
    # spans, only what lives on them.
    assert (
        ParallelConfig(
            enable_expert_parallel=True, data_parallel_size=2, tensor_parallel_size=2
        ).world_size
        == 2
    )
    # Off, and on a single device, it is 1 -- upstream ignores the flag there too.
    assert ParallelConfig(data_parallel_size=4).expert_parallel_size == 1
    assert ParallelConfig(enable_expert_parallel=True).expert_parallel_size == 1


async def test_the_http_surface_serves_over_replicas():
    import httpx
    from prometheus_client import CollectorRegistry

    from pvllm.engine.arg_utils import AsyncEngineArgs
    from pvllm.entrypoints.openai.api_server import build_app

    app = build_app(
        AsyncEngineArgs(
            model="tiny-test",
            served_model_name="m",
            max_model_len=512,
            device_card="tiny-2gb",
            data_parallel_size=2,
            disable_log_stats=True,
        ).create_engine_config(),
        registry=CollectorRegistry(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = [
            await client.post(
                "/v1/completions",
                json={"model": "m", "prompt": f"hi {index}", "max_tokens": 4},
            )
            for index in range(4)
        ]
    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["choices"][0]["text"] for response in responses)
