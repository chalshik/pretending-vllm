"""Expert parallelism. R13.4.

Expert parallelism does not shard anything new -- it *re-cuts* the MoE weights along
the expert axis instead of the intermediate axis, and widens the group doing the
cutting from the tensor dimension to the tensor *and data* dimensions. That last part
is the surprise: turning EP on stops data-parallel replicas being independent copies
of a model and makes them shards of one.

The payoff is a memory number and only a memory number. A Mixtral-class MoE is 46.7B
parameters of which 45.1B are experts, so where those experts live decides whether the
model fits at all -- and at `data_parallel_size=1` EP is a no-op, because `ep_size ==
tp_size` and dividing the experts eight ways is the same either way.
"""

from __future__ import annotations

import pytest

from pvllm.config.parallel import ParallelConfig
from pvllm.entrypoints.llm import LLM
from pvllm.sim.cost_model import StepProfile, build_cost_model
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.memory import compute_weight_bytes
from pvllm.sim.model_db import load_model_card

GIB = 1 << 30


@pytest.fixture(scope="module")
def moe():
    return load_model_card("moe-8x7b")


@pytest.fixture(scope="module")
def dense():
    return load_model_card("dense-8b")


# --- the group --------------------------------------------------------------


def test_the_group_spans_the_tensor_and_data_dimensions():
    """Upstream derives `ep_size = tp_size` *after* flattening TP across DP, which is
    `dp * tp`. Its own docstring example is dp=2, tp=2 -> ep=4."""
    config = ParallelConfig(
        enable_expert_parallel=True, data_parallel_size=2, tensor_parallel_size=2
    )
    assert config.expert_parallel_size == 4
    # And `world_size` stays per replica -- EP changes what lives on the devices, not
    # how many a replica spans.
    assert config.world_size == 2


def test_a_single_device_is_a_no_op_rather_than_an_error():
    """Upstream's `use_ep` requires `dp * tp > 1` and quietly stays False otherwise. A
    script that runs on real vLLM has to run here (C7), so this is accepted and
    logged rather than refused."""
    assert ParallelConfig(enable_expert_parallel=True).expert_parallel_size == 1


def test_a_dense_model_with_expert_parallelism_is_refused_by_name():
    """Upstream's `_verify_with_expert_parallelism` says the same thing: there is
    nothing to spread."""
    with pytest.raises(ValueError, match="mixture-of-experts"):
        LLM(
            model="dense-8b",
            device_card="datacenter-80gb",
            max_model_len=512,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            disable_log_stats=True,
        ).shutdown()


# --- memory, which is the whole point ---------------------------------------


def test_the_experts_divide_by_the_group_and_nothing_else_does(moe):
    """Attention, the norms, the router and the embeddings are untouched by EP. Only
    the expert MLP block moves."""
    whole = compute_weight_bytes(moe, "bfloat16")
    ep8 = compute_weight_bytes(moe, "bfloat16", tp_size=1, ep_size=8)
    # 45.1 of 46.7 billion parameters are experts, so eight-way expert sharding takes
    # ~87 GiB to ~13.5, not to ~11.
    assert whole / GIB == pytest.approx(87.0, abs=0.5)
    assert ep8 / GIB == pytest.approx(13.5, abs=0.5)


def test_with_expert_parallelism_off_the_experts_still_shard_by_tensor_parallelism(moe):
    """The trap in the sentinel: `ep_size=1` would mean "divide by one" and leave the
    experts *unsharded*, so a `--tensor-parallel-size 8` MoE would report 85 GiB per
    device instead of 11 and refuse hardware that fits it. `None` means "not
    expert-parallel"."""
    assert compute_weight_bytes(moe, "bfloat16", tp_size=8) / GIB == pytest.approx(
        11.3, abs=0.3
    )


def test_at_one_replica_expert_parallelism_changes_no_memory(moe):
    """`ep_size == tp_size` when `data_parallel_size == 1`, so the experts are divided
    the same number of ways either way. Worth pinning because it is the reason EP is
    only interesting with replicas."""
    assert compute_weight_bytes(
        moe, "bfloat16", tp_size=8, ep_size=8
    ) == compute_weight_bytes(moe, "bfloat16", tp_size=8)


def test_a_rank_holds_whole_experts_so_the_count_rounds_up(moe):
    """Eight experts over three ranks is 3/3/2, and the device that has to fit the
    model is the one holding three. Flooring would report the average and promise a
    fit the busiest rank does not have."""
    # Compared on the *expert* bytes: the dense and embedding terms are the same in
    # every case, so multiplying the whole per-device figure would triple those too.
    one = compute_weight_bytes(moe, "bfloat16", tp_size=1, ep_size=8)  # 1 expert
    two = compute_weight_bytes(moe, "bfloat16", tp_size=1, ep_size=4)  # 2 experts
    three = compute_weight_bytes(moe, "bfloat16", tp_size=1, ep_size=3)  # ceil(8/3)
    per_expert = two - one
    assert per_expert > 0
    assert three - one == pytest.approx(2 * per_expert, rel=1e-6)


def test_more_ranks_than_experts_still_holds_one_expert(moe):
    """At `ep_size > num_experts` a floor would report zero expert bytes for a rank
    that still holds one whole expert."""
    assert compute_weight_bytes(
        moe, "bfloat16", tp_size=1, ep_size=16
    ) == compute_weight_bytes(moe, "bfloat16", tp_size=1, ep_size=8)


def test_a_dense_model_is_unaffected(dense):
    assert compute_weight_bytes(
        dense, "bfloat16", tp_size=8, ep_size=8
    ) == compute_weight_bytes(dense, "bfloat16", tp_size=8)


# --- latency ----------------------------------------------------------------


def _cost(model, **kw):
    return build_cost_model(
        "roofline",
        model,
        load_device_card("datacenter-80gb"),
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        **kw,
    ).step_cost(
        StepProfile(
            num_tokens=512, num_reqs=4, query_lens=[128] * 4, seq_lens=[512] * 4
        )
    )


def test_expert_parallelism_does_not_reduce_per_device_compute(moe):
    """The obvious wrong move. Under TP each rank holds every expert sliced to `I/tp`
    and runs `tokens * top_k` pairs through the slice. Under EP each rank holds
    `E/ep` whole experts and runs the pairs that route to them at full width -- and
    because the token set is the union across replicas, the two land on the same
    number. EP moves where the weights live, not how much arithmetic happens."""
    assert _cost(moe, tp_size=8).compute_seconds == pytest.approx(
        _cost(moe, tp_size=8, ep_size=8, dp_size=1).compute_seconds
    )


def test_at_one_replica_expert_parallelism_costs_no_communication(moe):
    """Upstream's `use_all2all_kernels` requires `dp_size > 1`; below that the MoE
    layer issues the same all-reduce it would without EP. So a TP-only EP run must
    report the same duration as the TP run."""
    plain = _cost(moe, tp_size=8)
    with_ep = _cost(moe, tp_size=8, ep_size=8, dp_size=1)
    assert with_ep.comm_seconds == pytest.approx(plain.comm_seconds)
    assert with_ep.duration == pytest.approx(plain.duration)


def test_replicas_make_the_collective_carry_their_union(moe):
    """The cost of EP, and the reason `--data-parallel-size 8 --enable-expert-parallel`
    is a different proposition from `--tensor-parallel-size 8`: the EP group spans the
    replicas, so its collective carries every replica's tokens rather than one's."""
    one = _cost(moe, tp_size=1, ep_size=1, dp_size=1)
    eight = _cost(moe, tp_size=1, ep_size=8, dp_size=8)
    assert eight.comm_seconds > one.comm_seconds


def test_a_dense_model_gets_no_expert_term(dense):
    assert _cost(dense, tp_size=1, ep_size=8, dp_size=8).comm_seconds == pytest.approx(
        _cost(dense, tp_size=1).comm_seconds
    )


# --- end to end -------------------------------------------------------------


def test_expert_parallelism_makes_a_model_fit_that_otherwise_cannot():
    """The whole payoff, stated as a capacity answer: 87 GiB of weights does not fit
    on an 80 GiB card, and the same model at `--data-parallel-size 8
    --enable-expert-parallel` does."""
    with pytest.raises(Exception, match="No memory left for the KV cache"):
        LLM(
            model="moe-8x7b",
            device_card="datacenter-80gb",
            max_model_len=2048,
            disable_log_stats=True,
        ).shutdown()

    llm = LLM(
        model="moe-8x7b",
        device_card="datacenter-80gb",
        max_model_len=2048,
        block_size=16,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        tensor_parallel_size=1,
        data_parallel_size=8,
        enable_expert_parallel=True,
        disable_log_stats=True,
    )
    try:
        core = llm.llm_engine.engine_core.engine_cores[0]
        profile = core.executor.driver_worker.memory_profile
        assert profile is not None
        assert profile.weight_bytes / GIB == pytest.approx(13.5, abs=0.5)
        assert profile.num_gpu_blocks > 0
    finally:
        llm.shutdown()
