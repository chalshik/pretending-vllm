"""The cost model. R9.

R9.3 is the bar these hold the roofline to: reproduce the qualitative regimes with no
special-casing. Absolute accuracy is explicitly not claimed (R9.5) and is not asserted
here -- doing so would encode the current coefficients as if they were measurements.
"""

from __future__ import annotations

import numpy as np
import pytest

from pvllm.sim.cost_model import (
    ConstantCostModel,
    RooflineCostModel,
    StepProfile,
    build_cost_model,
)
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.model_db import load_model_card


@pytest.fixture
def roofline() -> RooflineCostModel:
    return RooflineCostModel(
        load_model_card("dense-8b"),
        load_device_card("datacenter-80gb"),
        dtype="bfloat16",
    )


def prefill(num_tokens: int) -> StepProfile:
    return StepProfile(
        num_tokens=num_tokens,
        num_reqs=1,
        query_lens=[num_tokens],
        seq_lens=[num_tokens],
    )


def decode(batch: int, context: int = 1024) -> StepProfile:
    return StepProfile(
        num_tokens=batch,
        num_reqs=batch,
        query_lens=[1] * batch,
        seq_lens=[context] * batch,
    )


# --- the qualitative regimes (R9.3) ----------------------------------------


def test_prefill_is_compute_bound(roofline):
    assert roofline.step_cost(prefill(4096)).is_compute_bound


def test_prefill_is_roughly_linear_in_tokens(roofline):
    """Slightly superlinear, because attention is quadratic in context."""
    small = roofline.step_cost(prefill(1024)).duration
    large = roofline.step_cost(prefill(2048)).duration
    assert 1.9 <= large / small <= 2.5


def test_decode_is_memory_bound(roofline):
    assert not roofline.step_cost(decode(8)).is_compute_bound


def test_decode_latency_is_nearly_flat_in_batch(roofline):
    """The fixed weight read dominates, which is what makes batching pay."""
    single = roofline.step_cost(decode(1)).duration
    batched = roofline.step_cost(decode(64)).duration
    assert batched < 2.5 * single


def test_throughput_saturates_with_batch_size(roofline):
    """Rising throughput with diminishing returns -- not unbounded scaling."""
    throughputs = [
        batch / roofline.step_cost(decode(batch)).duration for batch in (1, 8, 32, 128)
    ]
    assert throughputs == sorted(throughputs)
    early_gain = throughputs[1] / throughputs[0]
    late_gain = throughputs[3] / throughputs[2]
    assert late_gain < early_gain


def test_kv_traffic_eventually_dominates_at_long_context(roofline):
    short = roofline.step_cost(decode(32, context=1024))
    long = roofline.step_cost(decode(32, context=131072))
    assert long.duration > 5 * short.duration
    assert not long.is_compute_bound


def test_none_of_this_is_special_cased(roofline):
    """Both regimes come out of the same max(compute, memory), which is why the
    crossover appears without anyone coding one."""
    costs = [roofline.step_cost(prefill(4096)), roofline.step_cost(decode(4))]
    assert costs[0].is_compute_bound
    assert not costs[1].is_compute_bound


def test_moe_is_cheaper_than_its_parameter_count_suggests():
    """Only routed experts participate, so active parameters drive compute."""
    device = load_device_card("datacenter-80gb")
    moe = load_model_card("moe-8x7b")
    assert moe.num_active_parameters < moe.num_parameters / 3

    model = RooflineCostModel(moe, device, dtype="bfloat16")
    dense = RooflineCostModel(load_model_card("dense-70b"), device, dtype="bfloat16")
    # Comparable total size, far less compute.
    assert (
        model.step_cost(prefill(2048)).compute_seconds
        < dense.step_cost(prefill(2048)).compute_seconds
    )


# --- structure -------------------------------------------------------------


def test_cost_breakdown_is_reported_for_the_debug_surface(roofline):
    """D9: seeing that a step was memory-bound explains a latency curve in a way one
    number cannot."""
    record = roofline.step_cost(decode(8)).as_dict()
    assert record["bound_by"] == "memory"
    assert record["provenance"] == "modeled"
    assert record["duration"] > 0
    assert set(record) >= {"compute_s", "memory_s", "comm_s", "overhead_s", "flops"}


def test_tensor_parallelism_adds_a_communication_term():
    model, device = load_model_card("dense-8b"), load_device_card("datacenter-80gb")
    single = RooflineCostModel(model, device, dtype="bfloat16", tp_size=1)
    sharded = RooflineCostModel(model, device, dtype="bfloat16", tp_size=8)
    assert single.step_cost(prefill(1024)).comm_seconds == 0.0
    assert sharded.step_cost(prefill(1024)).comm_seconds > 0.0


def test_a_captured_graph_lowers_launch_overhead(roofline):
    eager = roofline.step_cost(decode(4))
    captured = roofline.step_cost(
        StepProfile(
            num_tokens=4,
            num_reqs=4,
            query_lens=[1] * 4,
            seq_lens=[1024] * 4,
            is_graph_hit=True,
        )
    )
    assert captured.overhead_seconds < eager.overhead_seconds


def test_enforce_eager_disables_graph_benefits():
    model, device = load_model_card("dense-8b"), load_device_card("datacenter-80gb")
    eager_only = RooflineCostModel(model, device, dtype="bfloat16", enforce_eager=True)
    hit = StepProfile(
        num_tokens=4,
        num_reqs=4,
        query_lens=[1] * 4,
        seq_lens=[64] * 4,
        is_graph_hit=True,
    )
    assert eager_only.step_cost(hit).overhead_seconds > 0
    assert eager_only.graph_capture_seconds(16) == 0.0


# --- jitter ----------------------------------------------------------------


def test_jitter_is_seeded_and_therefore_reproducible():
    """R19.2: a run with jitter is still reproducible from one seed."""
    model, device = load_model_card("dense-8b"), load_device_card("datacenter-80gb")

    def run() -> list[float]:
        cost_model = RooflineCostModel(
            model, device, dtype="bfloat16", jitter_sigma=0.1
        )
        rng = np.random.default_rng(7)
        return [cost_model.step_cost(decode(4), rng).duration for _ in range(10)]

    assert run() == run()


def test_jitter_never_runs_the_clock_backwards():
    """A large sigma could otherwise draw a negative multiplier."""
    model, device = load_model_card("dense-8b"), load_device_card("datacenter-80gb")
    cost_model = RooflineCostModel(model, device, dtype="bfloat16", jitter_sigma=5.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert cost_model.step_cost(decode(2), rng).duration >= 0.0


def test_no_jitter_means_identical_durations(roofline):
    rng = np.random.default_rng(0)
    costs = {roofline.step_cost(decode(4), rng).duration for _ in range(5)}
    assert len(costs) == 1


# --- the constant profile --------------------------------------------------


def test_constant_profile_is_deterministic_and_linear():
    """R9.6: a test asserting on step counts should not depend on the roofline's
    coefficients."""
    cost_model = ConstantCostModel()
    first = cost_model.step_cost(decode(4))
    second = cost_model.step_cost(decode(4))
    assert first.duration == second.duration
    assert cost_model.step_cost(decode(8)).duration > first.duration
    assert cost_model.weight_load_seconds(1 << 30) == 0.0


def test_build_cost_model_dispatches_on_profile_name():
    model, device = load_model_card("tiny-test"), load_device_card("tiny-2gb")
    assert (
        build_cost_model("constant", model, device, dtype="bfloat16").name == "constant"
    )
    assert (
        build_cost_model("roofline", model, device, dtype="bfloat16").name == "roofline"
    )
    with pytest.raises(ValueError, match="unknown cost_model_profile"):
        build_cost_model("magic", model, device, dtype="bfloat16")


def test_weight_load_time_scales_with_bandwidth(roofline):
    """R10.4: startup takes plausible time, so a readiness bug can surface."""
    assert roofline.weight_load_seconds(16 << 30) > 0
    assert roofline.weight_load_seconds(32 << 30) == pytest.approx(
        2 * roofline.weight_load_seconds(16 << 30)
    )
