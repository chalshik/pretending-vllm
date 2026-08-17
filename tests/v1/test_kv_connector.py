"""KV transfer and disaggregation. R17.

Disaggregated prefill splits a deployment in two: one pool of machines runs prompts,
another runs generation, and KV moves between them. Whether that is worth doing is
arithmetic -- **is pulling a prefix's KV cheaper than recomputing it?** -- and both
sides of the comparison are here: the store's bandwidth and latency on one side, the
cost model's prefill time on the other.

No bytes move. What is reproduced is the state machine and the timing: which requests
find their prefix externally, what that costs on the engine's clock, and how the
scheduler behaves around it. Those are the parts a product's latency depends on.
"""

from __future__ import annotations

import pytest

from pvllm.config.kv_transfer import KVTransferConfig
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.sim.kv_store import SimKVStore, get_store, reset_stores

BASE = {
    "model": "tiny-test",
    "max_model_len": 512,
    "block_size": 16,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
    "enable_prefix_caching": True,
}

PROMPT = "a long shared document a prefill node would have processed already. " * 3


@pytest.fixture(autouse=True)
def _clean_stores():
    """Stores are module-level so two engines can share one; they must not leak
    between tests."""
    reset_stores()
    yield
    reset_stores()


def connected(store: str = "shared", **extra) -> dict:
    return {
        "kv_transfer_config": {
            "kv_connector": "SimSharedStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"store_name": store, **extra},
        }
    }


# --- the store -------------------------------------------------------------


def test_a_prefix_stops_at_the_first_miss():
    """Like the local prefix cache, and for the same reason: KV for a gap does not
    exist, so a hit beyond one cannot be read."""
    store = SimKVStore(name="t")
    store.write([b"a", b"b", b"d"], num_bytes=3)

    assert store.longest_prefix([b"a", b"b", b"c", b"d"]) == 2


def test_transfer_time_is_latency_plus_size_over_bandwidth():
    store = SimKVStore(name="t", bandwidth_bytes_per_second=1e9, latency_seconds=0.002)

    assert store.transfer_seconds(0) == 0.0
    assert store.transfer_seconds(int(1e9)) == pytest.approx(1.002)
    # Latency dominates a small transfer, which is why a remote store can be
    # *slower* than recomputing a short prompt and much faster for a long one.
    assert store.transfer_seconds(1000) == pytest.approx(0.002, abs=1e-5)


def test_capacity_evicts_oldest_first():
    store = SimKVStore(name="t", capacity_blocks=2)
    store.write([b"a"], 1)
    store.write([b"b"], 1)
    store.write([b"c"], 1)

    assert list(store.resident) == [b"b", b"c"]


def test_rewriting_a_block_makes_it_most_recent():
    """Which is what makes the eviction order an LRU rather than a FIFO."""
    store = SimKVStore(name="t", capacity_blocks=2)
    store.write([b"a"], 1)
    store.write([b"b"], 1)
    store.write([b"a"], 1)
    store.write([b"c"], 1)

    assert list(store.resident) == [b"a", b"c"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bandwidth_bytes_per_second": 0}, "bandwidth_bytes_per_second"),
        ({"latency_seconds": -1}, "latency_seconds"),
    ],
)
def test_nonsense_store_parameters_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SimKVStore(name="t", **kwargs)


def test_two_engines_share_a_store_by_name():
    """The registry exists because the two halves of a disaggregated pair are
    configured separately -- neither can be handed a reference to the other's."""
    assert get_store("shared") is get_store("shared")
    assert get_store("shared") is not get_store("other")


# --- the config surface ----------------------------------------------------


def test_a_real_transport_is_refused_by_name():
    """Their bandwidth and failure modes are the whole question a disaggregation
    experiment is asking, so substituting the simulated store would answer a
    different one."""
    with pytest.raises(NotImplementedError, match="LMCacheConnectorV1"):
        KVTransferConfig(kv_connector="LMCacheConnectorV1")


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match="unknown kv_role"):
        KVTransferConfig(kv_connector="SimSharedStoreConnector", kv_role="kv_something")


def test_no_connector_means_no_connector():
    engine = LLM(**BASE)
    try:
        assert engine.llm_engine.engine_core.engine_core.scheduler.connector is None
    finally:
        engine.shutdown()


# --- disaggregation end to end (R17.2) -------------------------------------


def run_on(**overrides):
    """Run one prompt on a fresh engine. Returns `(steps, connector)`."""
    engine = LLM(**{**BASE, **overrides})
    try:
        engine.generate([PROMPT], SamplingParams(max_tokens=4))
        scheduler = engine.llm_engine.engine_core.engine_core.scheduler
        return engine.llm_engine.make_stats()["step_index"], scheduler.connector
    finally:
        engine.shutdown()


def test_a_prefill_node_publishes_its_kv():
    run_on(**connected())
    store = get_store("shared")

    assert len(store.resident) > 0
    assert store.bytes_written > 0
    # The producer pays for its writes and reads nothing: both halves asserted,
    # because a connector that charged the wrong side would still publish blocks.
    assert store.bytes_read == 0


def test_a_decode_node_finds_the_published_prefix():
    """The whole point: the second engine does not recompute what the first already
    did, and the two never spoke to each other -- they share a store name."""
    run_on(**connected())
    store = get_store("shared")
    published = len(store.resident)

    _, connector = run_on(**connected())

    assert store.num_hits > 0
    assert store.bytes_read > 0, "a hit that transferred nothing"
    assert connector.load_seconds > 0.0, "nothing was pulled"
    # `kv_role: kv_both`, so this node pulls the published prefix *and* publishes the
    # tokens it goes on to generate -- both directions are charged, and separately.
    # `save_seconds` is the other half of the pair, and it stayed unasserted long
    # enough to become an inert counter; the first assertion written here assumed a
    # decode node publishes nothing, which this config disproves.
    assert connector.save_seconds > 0.0, "a kv_both node published nothing"
    assert connector.load_seconds != connector.save_seconds, (
        "the two directions are counted separately, not from one accumulator"
    )
    assert len(store.resident) == published


def test_a_cold_store_costs_nothing():
    """No hit, no transfer. A connector that charged for a miss would make
    disaggregation look worse than it is at exactly the moment it does nothing."""
    _, connector = run_on(**connected())
    assert connector.load_seconds == 0.0


def test_a_different_prompt_does_not_match():
    run_on(**connected())
    hits_before = get_store("shared").num_hits

    engine = LLM(**{**BASE, **connected()})
    try:
        engine.generate(
            ["a completely unrelated question"], SamplingParams(max_tokens=4)
        )
    finally:
        engine.shutdown()

    assert get_store("shared").num_hits == hits_before


def test_the_transfer_cost_reaches_the_engine_clock():
    """R17.2's point: the pull's modeled duration is spent, so it shows up next to
    the prefill it replaced rather than being free."""
    run_on(**connected(bandwidth=1e6, latency=0.05))

    engine = LLM(**{**BASE, **connected(bandwidth=1e6, latency=0.05)})
    try:
        start = engine.llm_engine.engine_core.clock_time
        engine.generate([PROMPT], SamplingParams(max_tokens=4))
        elapsed = engine.llm_engine.engine_core.clock_time - start
        connector = engine.llm_engine.engine_core.engine_core.scheduler.connector
        assert connector.load_seconds > 0.0
        # The transfer is a real part of the run's modeled duration.
        assert elapsed > connector.load_seconds
    finally:
        engine.shutdown()


def test_a_slow_store_costs_more_than_a_fast_one():
    """The comparison a deployment is actually making. An NVMe-backed store and an
    object store differ by orders of magnitude, and which one you have decides
    whether disaggregation helps."""

    def load_time(bandwidth: float) -> float:
        reset_stores()
        run_on(**connected(bandwidth=bandwidth))
        _, connector = run_on(**connected(bandwidth=bandwidth))
        return connector.load_seconds

    assert load_time(1e6) > load_time(1e10)


def test_the_pulled_request_still_produces_every_token():
    """Pulling KV must not change the answer -- only how it was arrived at."""
    engine = LLM(**BASE)
    try:
        plain = list(
            engine.generate([PROMPT], SamplingParams(max_tokens=8))[0]
            .outputs[0]
            .token_ids
        )
    finally:
        engine.shutdown()

    run_on(**connected())
    engine = LLM(**{**BASE, **connected()})
    try:
        pulled = list(
            engine.generate([PROMPT], SamplingParams(max_tokens=8))[0]
            .outputs[0]
            .token_ids
        )
    finally:
        engine.shutdown()

    assert pulled == plain


def test_concurrent_requests_against_a_store_all_drain():
    run_on(**connected())
    engine = LLM(**{**BASE, **connected()})
    try:
        outputs = engine.generate([PROMPT] * 4, SamplingParams(max_tokens=6))
        assert len(outputs) == 4
        assert all(len(o.outputs[0].token_ids) == 6 for o in outputs)
    finally:
        engine.shutdown()


def test_asking_for_kv_cache_events_is_refused_by_name():
    """R12.5's block store/remove event stream is not implemented.

    The flag used to be accepted, threaded into `BlockPool`, stored, and read by
    nothing -- so a client that asked for events got a pool that published none and
    said so nowhere. That is the silent no-op this project refuses everywhere else,
    and it survived because an unread attribute is invisible until something looks
    for one.
    """
    import pytest

    from pvllm.v1.core.block_pool import BlockPool

    with pytest.raises(NotImplementedError, match="KV cache event publishing"):
        BlockPool(num_gpu_blocks=8, enable_kv_cache_events=True)

    # And the ordinary construction is untouched.
    assert BlockPool(num_gpu_blocks=8).num_gpu_blocks == 8
