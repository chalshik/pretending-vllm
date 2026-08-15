"""The token generator and the simulated device. R11.1--R11.3, R10.1."""

from __future__ import annotations

import pytest

from pvllm.sim.clock import VirtualClock
from pvllm.sim.cost_model import ConstantCostModel, StepProfile
from pvllm.sim.device import SimDevice
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.memory import SimOutOfMemoryError, compute_memory_profile
from pvllm.sim.model import SimModel
from pvllm.sim.model_db import load_model_card
from pvllm.sim.rng import RngFactory
from pvllm.tokenizers.mock import EOS_TOKEN_ID, MockTokenizer


def make_model(policy: str = "from_request", **kwargs) -> SimModel:
    return SimModel(
        model=load_model_card("dense-8b"),
        rng_factory=RngFactory(seed=42),
        output_length_policy=policy,
        **kwargs,
    )


def generate(model: SimModel, request_id: str, max_tokens: int) -> list[int]:
    """Emit until EOS, as the engine would."""
    tokens: list[int] = []
    for position in range(max_tokens):
        token = model.sample_token(request_id, position, max_tokens)
        tokens.append(token)
        if token == EOS_TOKEN_ID:
            break
    return tokens


# --- output length (R11.2) -------------------------------------------------


def test_from_request_honours_max_tokens():
    """The default, because a test double should do what the client asked."""
    model = make_model("from_request")
    assert model.planned_output_length("r0", max_tokens=64) == 64


def test_fixed_policy_ignores_the_request():
    model = make_model("fixed", output_length_fixed=10)
    assert model.planned_output_length("r0", max_tokens=1000) == 10


def test_lognormal_produces_a_spread_of_lengths():
    """The knob that makes workload experiments meaningful: real requests stop at
    varying lengths, which drives the batch composition the scheduler sees."""
    model = make_model("lognormal")
    lengths = {model.planned_output_length(f"r{i}", 4096) for i in range(50)}
    assert len(lengths) > 10


def test_uniform_stays_within_its_range():
    model = make_model("uniform", output_length_range=(5, 15))
    for i in range(50):
        assert 5 <= model.planned_output_length(f"r{i}", 4096) <= 15


def test_a_policy_never_exceeds_what_the_client_asked_for():
    """Emitting past max_tokens would be a protocol violation whatever the workload
    model says."""
    model = make_model("fixed", output_length_fixed=1000)
    assert model.planned_output_length("r0", max_tokens=8) == 8


def test_planned_length_is_decided_once():
    """Redrawing every step would let the target wander, and the request would stop
    when a draw happened to be small rather than at a planned length."""
    model = make_model("lognormal")
    first = model.planned_output_length("r0", 4096)
    assert all(model.planned_output_length("r0", 4096) == first for _ in range(5))


def test_unimplemented_policies_name_themselves():
    with pytest.raises(NotImplementedError, match="from_fixture"):
        make_model("from_fixture").planned_output_length("r0", 16)


# --- generation ------------------------------------------------------------


def test_generation_stops_by_emitting_eos():
    """Through the real stop-detection path (R11.5), not out of band -- the finish
    accounting, finish_reason, and metrics all key off that path."""
    model = make_model("fixed", output_length_fixed=5)
    tokens = generate(model, "r0", max_tokens=100)
    assert tokens[-1] == EOS_TOKEN_ID
    assert len(tokens) == 5


def test_output_is_reproducible_from_the_seed():
    assert generate(make_model(), "r0", 32) == generate(make_model(), "r0", 32)


def test_output_is_independent_of_interleaving():
    """R19.2, at the model layer: a request's tokens must not depend on who else was
    running."""
    solo = generate(make_model(), "target", 24)

    busy = make_model()
    for other in ("a", "b", "c"):
        generate(busy, other, 24)
    assert generate(busy, "target", 24) == solo


def test_generated_content_detokenizes_to_stable_text():
    """R11.3: the property HTTP golden tests depend on."""
    tokenizer = MockTokenizer(vocab_size=load_model_card("dense-8b").vocab_size)
    first = tokenizer.decode(generate(make_model(), "r0", 20), skip_special_tokens=True)
    second = tokenizer.decode(
        generate(make_model(), "r0", 20), skip_special_tokens=True
    )
    assert first == second
    assert first  # non-empty


def test_sampled_ids_are_within_the_vocabulary():
    model = make_model()
    vocab_size = load_model_card("dense-8b").vocab_size
    for token in generate(model, "r0", 64):
        assert 0 <= token < vocab_size


def test_batch_sampling_matches_per_request_sampling():
    batched = make_model().sample_tokens(["a", "b"], [0, 0], [16, 16])
    individual = make_model()
    assert batched == [
        individual.sample_token("a", 0, 16),
        individual.sample_token("b", 0, 16),
    ]


# --- logprobs (R11.1, NG3) -------------------------------------------------


def test_logprobs_are_schema_correct_without_a_vocab_sized_array():
    """R11.1: on a 128k vocabulary at batch 256 that array would be 128 MiB a step,
    which would make the simulator slower than the thing it simulates."""
    model = make_model()
    token_ids, logprobs, rank = model.sample_logprobs("r0", sampled_token_id=500, k=5)

    assert len(token_ids) == len(logprobs) == 5
    assert token_ids[0] == 500
    assert rank == 0
    assert logprobs == sorted(logprobs, reverse=True)
    assert all(value <= 0.0 for value in logprobs)


def test_zero_logprobs_returns_just_the_sampled_token():
    token_ids, logprobs, rank = make_model().sample_logprobs("r0", 7, k=0)
    assert token_ids == [7] and logprobs == [0.0] and rank == 0


# --- the device (R10.1) ----------------------------------------------------


def make_device() -> SimDevice:
    return SimDevice(
        card=load_device_card("datacenter-80gb"),
        clock=VirtualClock(),
        cost_model=ConstantCostModel(),
    )


def test_executing_a_step_advances_the_clock():
    """The single place the clock moves during inference."""
    device = make_device()
    before = device.clock.elapsed
    cost = device.execute(StepProfile(num_tokens=8, num_reqs=2))
    assert cost.duration > 0
    assert device.clock.elapsed == pytest.approx(before + cost.duration)
    assert device.num_steps == 1


async def test_async_execution_advances_the_clock_too():
    """Needed under a real clock: blocking the loop would stall the HTTP server that
    is meant to be streaming during the step."""
    device = make_device()
    cost = await device.execute_async(StepProfile(num_tokens=4, num_reqs=1))
    assert device.clock.elapsed == pytest.approx(cost.duration)


def test_applying_a_memory_profile_claims_the_pools():
    device = make_device()
    profile = compute_memory_profile(
        load_model_card("dense-8b"),
        load_device_card("datacenter-80gb"),
        dtype="bfloat16",
        kv_cache_dtype=None,
        block_size=16,
        gpu_memory_utilization=0.92,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
    )
    device.apply_memory_profile(profile)

    pools = device.ledger.pools()
    assert set(pools) >= {"weights", "activation_peak", "kv_cache"}
    assert pools["weights"] == profile.weight_bytes


def test_the_device_refuses_to_pretend_it_is_bigger_than_its_card():
    device = SimDevice(
        card=load_device_card("tiny-2gb"),
        clock=VirtualClock(),
        cost_model=ConstantCostModel(),
    )
    with pytest.raises(SimOutOfMemoryError):
        device.allocate("weights", 8 << 30)


def test_last_step_cost_is_available_for_introspection():
    """D9: the debug surface reports why a step took what it took."""
    device = make_device()
    assert device.last_step_cost is None
    device.execute(StepProfile(num_tokens=4, num_reqs=1))
    assert device.last_step_cost is not None
    assert device.last_step_cost.as_dict()["provenance"] == "modeled"
