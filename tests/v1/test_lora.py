"""LoRA adapters. R16.1.

Three things about LoRA are observable from outside, and they are the three that are
tested here. The rest of upstream's LoRA machinery -- loading weights, the punica
kernels, slicing by rank -- has no simulated counterpart and no observable effect.

1. **Adapter identity partitions the prefix cache.** Two requests with byte-identical
   prompts and different adapters produce different KV, so they must not share blocks.
   Getting this wrong reports cache hit rates far above a real deployment's, and C3
   calls hit rate exact.
2. **Adapters cost device memory**, which comes out of the KV pool. Serving eight
   adapters at rank 64 is not free, and a capacity answer that ignored it is wrong in
   the optimistic direction.
3. **`max_loras` bounds concurrency**, independently of KV capacity and `max_num_seqs`.
   A fifth adapter waits for a slot. That queueing is invisible unless modeled, and it
   is the reason a multi-tenant deployment behaves differently from a single-tenant
   one at the same request rate.
"""

from __future__ import annotations

import pytest

from pvllm.config.lora import SUPPORTED_LORA_RANKS, LoRAConfig
from pvllm.engine.arg_utils import EngineArgs
from pvllm.lora.request import LoRARequest
from pvllm.sampling_params import SamplingParams
from pvllm.v1.engine.llm_engine import LLMEngine

BASE = {
    "model": "dense-0.6b",
    "max_model_len": 512,
    "block_size": 16,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 8,
    "device_card": "workstation-24gb",
    "disable_log_stats": True,
}


def adapter(index: int, rank: int = 16) -> LoRARequest:
    return LoRARequest(
        lora_name=f"adapter-{index}",
        lora_int_id=index,
        lora_path=f"/adapters/{index}",
        rank=rank,
    )


def make_engine(**overrides) -> LLMEngine:
    return LLMEngine.from_engine_args(EngineArgs(**{**BASE, **overrides}))


# --- the request type ------------------------------------------------------


def test_adapters_are_identified_by_id_alone():
    """The id is the adapter's identity. Hashing the path too would make two
    references to one adapter look like two adapters to the scheduler's slot set,
    which would then admit more adapters than there are slots."""
    assert adapter(1) == LoRARequest(lora_name="other", lora_int_id=1, lora_path="/z")
    assert len({adapter(1), LoRARequest(lora_name="x", lora_int_id=1)}) == 1
    assert adapter(1) != adapter(2)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_adapter_id_is_refused(bad):
    with pytest.raises(ValueError, match="lora_int_id must be positive"):
        LoRARequest(lora_name="x", lora_int_id=bad, lora_path="/x")


def test_an_unsupported_rank_is_refused_rather_than_rounded():
    """Rounding would silently change how much memory the adapter occupies, which is
    the number this config exists to get right."""
    with pytest.raises(ValueError, match="max_lora_rank must be one of"):
        LoRAConfig(max_lora_rank=17)
    for rank in SUPPORTED_LORA_RANKS:
        assert LoRAConfig(max_lora_rank=rank).max_lora_rank == rank


# --- the prefix cache (R6.3, C3) -------------------------------------------


def test_the_same_prompt_under_different_adapters_does_not_share_blocks():
    """The cache-poisoning case. Sharing here would mean one tenant reading
    another's KV, and would inflate the reported hit rate."""
    from pvllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
    from pvllm.v1.request import Request

    def request_with(lora: LoRARequest | None) -> Request:
        return Request(
            request_id="r",
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=SamplingParams(max_tokens=4),
            arrival_time=0.0,
            lora_request=lora,
        )

    keys_one = generate_block_hash_extra_keys(request_with(adapter(1)))
    keys_two = generate_block_hash_extra_keys(request_with(adapter(2)))
    keys_none = generate_block_hash_extra_keys(request_with(None))

    assert keys_one != keys_two
    assert keys_one != keys_none
    assert keys_none is None


def test_two_requests_on_one_adapter_do_share():
    """The other half: partitioning by adapter must not partition by request, or
    the cache would never hit for a multi-tenant workload at all."""
    from pvllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
    from pvllm.v1.request import Request

    def request_named(name: str) -> Request:
        return Request(
            request_id=name,
            prompt_token_ids=[1, 2, 3, 4],
            sampling_params=SamplingParams(max_tokens=4),
            arrival_time=0.0,
            # Same adapter id, different name and path -- one adapter referred to
            # two ways.
            lora_request=LoRARequest(
                lora_name=name, lora_int_id=7, lora_path=f"/{name}"
            ),
        )

    assert generate_block_hash_extra_keys(
        request_named("a")
    ) == generate_block_hash_extra_keys(request_named("b"))


def test_a_shared_prefix_hits_within_an_adapter_and_misses_across():
    """End to end, through the real cache rather than the key function."""
    prompt = "a shared prefix long enough to fill more than one block of KV cache"

    engine = make_engine(enable_lora=True, max_loras=4, enable_prefix_caching=True)
    try:
        for index, lora in enumerate([adapter(1), adapter(1), adapter(2)]):
            engine.add_request(
                f"r{index}", prompt, SamplingParams(max_tokens=4), lora_request=lora
            )
        while engine.has_unfinished_requests():
            engine.step()

        stats = engine.make_stats()
        # r1 reuses r0's blocks; r2 cannot, because its adapter differs.
        assert stats["prefix_cache_hits"] > 0
        assert stats["prefix_cache_hits"] < stats["prefix_cache_queries"]
    finally:
        engine.shutdown()


# --- memory (R16.1) --------------------------------------------------------


def test_adapters_take_memory_from_the_kv_pool():
    def blocks(**overrides) -> int:
        engine = make_engine(**overrides)
        profile = engine.engine_core.engine_core.executor.driver_worker.memory_profile
        engine.shutdown()
        assert profile is not None
        return profile.num_gpu_blocks

    plain = blocks()
    one = blocks(enable_lora=True, max_loras=1, max_lora_rank=16)
    many = blocks(enable_lora=True, max_loras=8, max_lora_rank=64)

    assert plain > one > many, (plain, one, many)


def test_lora_memory_scales_with_adapters_and_rank():
    from pvllm.sim.memory import compute_lora_bytes
    from pvllm.sim.model_db import load_model_card

    card = load_model_card("dense-8b")
    one = compute_lora_bytes(card, "bfloat16", max_loras=1, max_lora_rank=16)
    four = compute_lora_bytes(card, "bfloat16", max_loras=4, max_lora_rank=16)
    wide = compute_lora_bytes(card, "bfloat16", max_loras=1, max_lora_rank=64)

    assert four == 4 * one
    assert wide == 4 * one
    assert compute_lora_bytes(card, "bfloat16", max_loras=0, max_lora_rank=16) == 0


# --- the admission constraint (R16.1) --------------------------------------


def test_more_adapters_than_slots_queue_rather_than_run():
    """The queueing that makes a multi-tenant deployment behave differently from a
    single-tenant one at the same rate, and that is invisible unless modeled."""
    engine = make_engine(enable_lora=True, max_loras=2, max_num_seqs=8)
    try:
        for index in range(4):
            engine.add_request(
                f"r{index}",
                f"prompt {index}",
                SamplingParams(max_tokens=6),
                lora_request=adapter(index + 1),
            )
        engine.step()

        scheduler = engine.engine_core.engine_core.scheduler
        resident = {
            request.lora_request.lora_int_id
            for request in scheduler.running
            if request.lora_request is not None
        }
        assert len(resident) <= 2
        # Everything else is waiting on a slot, not on capacity.
        assert len(scheduler.running) < 4
    finally:
        engine.shutdown()


def test_every_request_still_completes_when_slots_are_contended():
    """The constraint must throttle, not deadlock: an adapter that cannot be
    admitted now has to be admitted once a slot frees."""
    engine = make_engine(enable_lora=True, max_loras=2, max_num_seqs=8)
    try:
        for index in range(6):
            engine.add_request(
                f"r{index}",
                f"prompt {index}",
                SamplingParams(max_tokens=4),
                lora_request=adapter(index + 1),
            )
        finished = set()
        for _ in range(500):
            if not engine.has_unfinished_requests():
                break
            for output in engine.step():
                if output.finished:
                    finished.add(output.request_id)
        assert finished == {f"r{i}" for i in range(6)}
    finally:
        engine.shutdown()


def test_requests_sharing_one_adapter_are_not_throttled():
    """`max_loras` bounds distinct adapters, not requests. Ten requests on one
    adapter need one slot."""
    engine = make_engine(enable_lora=True, max_loras=1, max_num_seqs=8)
    try:
        for index in range(6):
            engine.add_request(
                f"r{index}",
                f"prompt {index}",
                SamplingParams(max_tokens=4),
                lora_request=adapter(1),
            )
        engine.step()
        assert len(engine.engine_core.engine_core.scheduler.running) > 1
    finally:
        engine.shutdown()


# --- the config surface ----------------------------------------------------


def test_lora_is_off_unless_enabled():
    engine = make_engine()
    assert engine.vllm_config.lora_config is None
    engine.shutdown()


def test_lora_modules_without_enable_lora_is_refused():
    """Serving an adapter changes both the memory budget and the admission
    constraint, so it is not inferred from the presence of a module."""
    with pytest.raises(ValueError, match="--lora-modules was given without"):
        EngineArgs(**BASE, lora_modules=["sql=/adapters/sql"]).create_engine_config()
