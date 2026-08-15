"""Tensor and pipeline parallelism. R13.1, R13.2.

Neither is *executed* here -- there is one process and no devices -- so what is tested
is the only thing a simulator can honestly claim about them: the numbers they change.

Those numbers are the reason the config refused to accept `tensor_parallel_size=2`
until now. "How many blocks do I get on four GPUs" is the question a capacity plan
turns on, and answering it with single-device memory would look like it worked.

The assertions are on direction and ordering, never magnitude: the cost model is
uncalibrated (R9.5), so a test pinning a millisecond count would pin a number nobody
should trust.
"""

from __future__ import annotations

import pytest

from pvllm.config.parallel import ParallelConfig
from pvllm.entrypoints.llm import LLM
from pvllm.sim.cost_model import RooflineCostModel, StepProfile
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.model_db import load_model_card

BASE = {
    "model": "dense-8b",
    "max_model_len": 2048,
    "device_card": "datacenter-80gb",
    "cost_model_profile": "roofline",
    "disable_log_stats": True,
}


def profile_for(**overrides) -> tuple[int, int, int]:
    """`(num_gpu_blocks, weight_bytes, kv_bytes_per_token)` for a configuration."""
    engine = LLM(**{**BASE, **overrides})
    try:
        memory = engine.llm_engine.engine_core.engine_core.executor.driver_worker.memory_profile
        assert memory is not None
        return memory.num_gpu_blocks, memory.weight_bytes, memory.kv_bytes_per_token
    finally:
        engine.shutdown()


def cost(tp: int = 1, pp: int = 1, *, tokens: int, reqs: int, context: int):
    model = load_model_card("dense-8b")
    device = load_device_card("datacenter-80gb")
    return RooflineCostModel(
        model,
        device,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        tp_size=tp,
        pp_size=pp,
    ).step_cost(
        StepProfile(
            num_tokens=tokens,
            num_reqs=reqs,
            query_lens=[tokens // reqs] * reqs,
            seq_lens=[context] * reqs,
        )
    )


# --- what the config accepts -----------------------------------------------


def test_tensor_and_pipeline_parallelism_are_accepted():
    assert (
        ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=2).world_size == 8
    )


def test_data_parallelism_points_at_running_more_engines():
    """Replicas are independent engines behind a router, not a sharded one. The
    interesting behavior would be the router's, not this engine's."""
    with pytest.raises(NotImplementedError, match="separate pretending-vllm instances"):
        ParallelConfig(data_parallel_size=2)


# --- memory (the capacity answer) ------------------------------------------


def test_tensor_parallelism_shards_the_kv_cache_and_the_weights():
    one_blocks, one_weights, one_kv = profile_for(tensor_parallel_size=1)
    four_blocks, four_weights, four_kv = profile_for(tensor_parallel_size=4)

    # KV shards exactly with the head count.
    assert four_kv == one_kv // 4
    assert four_weights < one_weights
    # More room per device *and* cheaper per token, so blocks rise superlinearly.
    assert four_blocks > one_blocks * 4


def test_pipeline_parallelism_shards_layers_across_stages():
    one_blocks, one_weights, one_kv = profile_for(pipeline_parallel_size=1)
    two_blocks, two_weights, two_kv = profile_for(pipeline_parallel_size=2)

    assert two_weights == pytest.approx(one_weights / 2, rel=0.01)
    # Each stage holds half the layers, so it caches half the KV per token.
    assert two_kv == one_kv // 2
    assert two_blocks > one_blocks


def test_the_two_compose():
    plain, _, _ = profile_for()
    both, _, _ = profile_for(tensor_parallel_size=2, pipeline_parallel_size=2)
    assert both > plain * 4


# --- latency (R9's regimes) ------------------------------------------------


def test_tensor_parallelism_speeds_up_a_compute_bound_prefill():
    """Nearly linear, because prefill is compute-bound and the work divides."""
    one = cost(tp=1, tokens=2048, reqs=1, context=2048).duration
    four = cost(tp=4, tokens=2048, reqs=1, context=2048).duration
    assert four < one / 3


def test_tensor_parallelism_helps_decode_less_than_prefill():
    """Decode is memory-bound and carries a fixed launch overhead, so the speedup
    is sublinear -- the shape a capacity plan needs to see, since it is why eight
    GPUs do not decode eight times faster."""
    one = cost(tp=1, tokens=8, reqs=8, context=512).duration
    eight = cost(tp=8, tokens=8, reqs=8, context=512).duration

    assert eight < one
    assert eight > one / 8


def test_tensor_parallelism_costs_communication():
    assert cost(tp=1, tokens=8, reqs=8, context=512).comm_seconds == 0.0
    assert cost(tp=4, tokens=8, reqs=8, context=512).comm_seconds > 0.0


def test_pipeline_parallelism_does_not_speed_up_a_step():
    """The documented limitation, asserted so it stays documented.

    A batch traverses every stage before a token comes out, so one step costs the
    whole model's work whatever `pp_size` is. Real pipeline parallelism recovers
    throughput by overlapping microbatches; there are no virtual engines here, so PP
    is "same latency, less memory per device" -- correct for one request, pessimistic
    for a saturated engine.
    """
    one = cost(pp=1, tokens=8, reqs=8, context=512)
    four = cost(pp=4, tokens=8, reqs=8, context=512)

    assert four.compute_seconds == pytest.approx(one.compute_seconds)
    assert four.memory_seconds == pytest.approx(one.memory_seconds)
    # Only the stage hand-offs are new, and they are small next to an all-reduce.
    assert four.comm_seconds > 0.0
    assert four.comm_seconds < cost(tp=4, tokens=8, reqs=8, context=512).comm_seconds


def test_pipeline_hand_offs_cost_less_than_tensor_all_reduces():
    """Which is why pipeline parallelism is what crosses a slow link."""
    tensor = cost(tp=4, tokens=64, reqs=8, context=512).comm_seconds
    pipeline = cost(pp=4, tokens=64, reqs=8, context=512).comm_seconds
    assert pipeline < tensor
