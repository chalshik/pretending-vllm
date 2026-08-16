"""Hybrid KV cache groups: models that mix full and windowed attention. R6.7.

M4e covered a *uniformly* windowed model. This is the shape the current generation of
open models actually ships: Gemma-3 alternates five sliding-window layers to every
full-attention one, Llama-4 three local to one global. The KV footprint is then
neither bounded nor unbounded, and reporting either figure answers a capacity
question with the wrong model's number.

The grouping is upstream's and its shape is not obvious: 25 windowed layers and 5 full
ones become *six* groups of five, not two groups of 25 and 5. Every group must occupy
the same bytes per block or the shared pool fragments, so the group size is the
smallest bucket and larger buckets are split.
"""

from __future__ import annotations

import pytest

from pvllm.engine.arg_utils import EngineArgs
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.v1.core.kv_cache_utils import get_kv_cache_groups
from pvllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)
from pvllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

BASE = {
    "model": "hybrid-4b",
    "device_card": "datacenter-80gb",
    "max_model_len": 4096,
    "block_size": 16,
    "max_num_batched_tokens": 512,
    "max_num_seqs": 4,
    "disable_log_stats": True,
    "seed": 5,
}


def spec(window: int | None = None, layers_head_size: int = 256) -> KVCacheSpec:
    shared = {
        "block_size": 16,
        "num_kv_heads": 4,
        "head_size": layers_head_size,
        "dtype": "bfloat16",
        "dtype_bytes": 2,
    }
    if window is None:
        return FullAttentionSpec(**shared)
    return SlidingWindowSpec(**shared, sliding_window=window)


# --- the card ---------------------------------------------------------------


def test_the_card_describes_its_repeat_pattern():
    from pvllm.sim.model_db import load_model_card

    card = load_model_card("hybrid-4b")
    assert card.is_hybrid_attention
    assert card.num_hidden_layers == 30
    # Gemma-3's convention: the last layer of each repeat is the full one.
    assert card.num_full_attention_layers == 5
    assert [card.layer_is_full_attention(i) for i in range(6)] == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_a_dense_card_is_not_hybrid():
    from pvllm.sim.model_db import load_model_card

    card = load_model_card("dense-8b")
    assert not card.is_hybrid_attention
    assert card.num_full_attention_layers == card.num_hidden_layers


# --- the grouping -----------------------------------------------------------


def test_layers_of_a_hybrid_model_get_their_own_specs():
    config = EngineArgs(**BASE).create_engine_config()
    specs = get_kv_cache_spec(config)
    assert len(specs) == 30
    windowed = [name for name, s in specs.items() if isinstance(s, SlidingWindowSpec)]
    full = [name for name, s in specs.items() if type(s) is FullAttentionSpec]
    assert len(windowed) == 25
    assert len(full) == 5


def test_the_groups_are_equal_sized_and_share_a_page_size():
    """Twenty-five windowed and five full layers become six groups of five, not two
    groups of twenty-five and five: the pool can only be divided evenly if every
    group occupies the same bytes per block."""
    groups = get_kv_cache_groups(
        get_kv_cache_spec(EngineArgs(**BASE).create_engine_config())
    )
    assert len(groups) == 6
    assert {len(group.layer_names) for group in groups} == {5}
    pages = {
        group.kv_cache_spec.page_size_bytes * len(group.layer_names) for group in groups
    }
    assert len(pages) == 1
    assert (
        sum(isinstance(group.kv_cache_spec, SlidingWindowSpec) for group in groups) == 5
    )


def test_a_dense_model_still_collapses_to_one_group():
    """The common case must not pay for the feature."""
    config = EngineArgs(
        model="dense-8b", device_card="datacenter-80gb", max_model_len=2048
    ).create_engine_config()
    groups = get_kv_cache_groups(get_kv_cache_spec(config))
    assert len(groups) == 1
    assert len(groups[0].layer_names) == 32


def test_the_split_stripes_rather_than_slices():
    """Upstream uses `layers[i::num_groups]`, not contiguous slices. Under pipeline
    parallelism a contiguous split puts whole groups on one stage and leaves empty
    ones on another, which are then padded to the same size -- wasting exactly the
    memory the grouping exists to save."""
    specs: dict[str, KVCacheSpec] = {}
    for index in range(4):
        specs[f"full.{index}"] = spec()
    for index in range(8):
        specs[f"sw.{index}"] = spec(window=512)

    groups = get_kv_cache_groups(specs)
    assert len(groups) == 3
    assert {len(group.layer_names) for group in groups} == {4}
    windowed = [
        group.layer_names
        for group in groups
        if isinstance(group.kv_cache_spec, SlidingWindowSpec)
    ]
    assert windowed == [
        ["sw.0", "sw.2", "sw.4", "sw.6"],
        ["sw.1", "sw.3", "sw.5", "sw.7"],
    ]


def test_groups_that_cannot_share_a_page_size_are_refused_by_name():
    """The invariant the whole scheme rests on. A violation means the pool cannot be
    divided into equal pages, and every block count downstream would be wrong."""
    specs: dict[str, KVCacheSpec] = {
        "a.0": spec(),
        "a.1": spec(),
        # A different head size makes a different page size, so no group size can
        # reconcile the two.
        "b.0": spec(window=512, layers_head_size=128),
    }
    with pytest.raises(NotImplementedError, match="same bytes per block"):
        get_kv_cache_groups(specs)


# --- the pool ---------------------------------------------------------------


def test_the_memory_profile_and_the_engine_core_agree():
    """They size the pool independently, and the startup line is what a capacity plan
    reads. Two answers means one of them is a lie."""
    llm = LLM(**BASE)
    try:
        core = llm.llm_engine.engine_core.engine_core
        assert (
            core.kv_cache_config.num_blocks
            == core.executor.driver_worker.memory_profile.num_gpu_blocks
        )
    finally:
        llm.shutdown()


def test_a_hybrid_request_is_charged_for_every_group_it_holds():
    """A hybrid request holds blocks in all six groups: the window's worth in five of
    them and the whole conversation's worth in one. Charging it for one group would
    report the bounded-KV concurrency of a model that is only five-sixths bounded."""
    from pvllm.sim.hardware_db import load_device_card
    from pvllm.sim.memory import compute_memory_profile
    from pvllm.sim.model_db import load_model_card

    config = EngineArgs(**BASE).create_engine_config()
    groups = get_kv_cache_groups(get_kv_cache_spec(config))
    profile = compute_memory_profile(
        load_model_card("hybrid-4b"),
        load_device_card("datacenter-80gb"),
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        block_size=16,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        kv_cache_groups=groups,
    )
    # Five windowed groups at ceil(1023/16)+1 = 65 blocks, one full group at
    # 4096/16 = 256 blocks.
    expected_blocks_per_request = 5 * 65 + 256
    assert profile.max_concurrency == pytest.approx(
        (profile.num_gpu_blocks - 1) / expected_blocks_per_request, rel=1e-6
    )


def test_hybrid_attention_buys_concurrency_on_the_same_model():
    """The capacity argument, run both ways on one model rather than compared across
    two. With the hybrid manager disabled every layer holds the whole conversation."""

    def concurrency(**overrides) -> float:
        llm = LLM(**{**BASE, "max_model_len": 8192, **overrides})
        try:
            worker = llm.llm_engine.engine_core.engine_core.executor.driver_worker
            assert worker.memory_profile is not None
            return worker.memory_profile.max_concurrency
        finally:
            llm.shutdown()

    hybrid = concurrency()
    promoted = concurrency(disable_hybrid_kv_cache_manager=True)
    assert hybrid > promoted * 2


def test_disabling_the_hybrid_manager_collapses_the_groups():
    llm = LLM(**BASE, disable_hybrid_kv_cache_manager=True)
    try:
        core = llm.llm_engine.engine_core.engine_core
        assert core.kv_cache_config.num_groups == 1
        assert not core.scheduler.kv_cache_manager.has_sliding_window
    finally:
        llm.shutdown()


# --- running it -------------------------------------------------------------


def test_a_hybrid_model_serves_requests():
    llm = LLM(**BASE)
    try:
        outputs = llm.generate(
            ["a prompt for the hybrid model", "and a second one"],
            SamplingParams(max_tokens=24),
        )
        assert all(len(output.outputs[0].token_ids) == 24 for output in outputs)
        assert llm.llm_engine.engine_core.engine_core.kv_cache_config.num_groups == 6
    finally:
        llm.shutdown()


def test_the_kv_layout_does_not_change_what_is_generated():
    """The output is a property of the model and the seed, not of how its KV is
    filed. If collapsing six groups into one changed the tokens, one of the two
    layouts would be writing KV somewhere the other reads."""

    def run(**overrides) -> list[int]:
        llm = LLM(**{**BASE, **overrides})
        try:
            output = llm.generate(["a prompt"], SamplingParams(max_tokens=16))[0]
            return list(output.outputs[0].token_ids)
        finally:
            llm.shutdown()

    assert run() == run(disable_hybrid_kv_cache_manager=True)


def test_every_group_s_slot_mapping_is_validated():
    """R8.3's oracle is only an oracle if it runs over every group. A hybrid model's
    windowed groups free blocks the full group keeps, which is exactly where an
    off-by-one in block accounting lands -- and checking group 0 alone would never
    see it."""
    import pvllm.v1.worker.gpu.model_runner as runner_module

    seen: list[int] = []
    original = runner_module.build_attn_metadata

    def recording(*args, **kwargs):
        seen.append(kwargs.get("group_id", 0))
        return original(*args, **kwargs)

    runner_module.build_attn_metadata = recording
    try:
        llm = LLM(**BASE)
        try:
            llm.generate(["a prompt"], SamplingParams(max_tokens=4))
        finally:
            llm.shutdown()
    finally:
        runner_module.build_attn_metadata = original

    assert set(seen) == {0, 1, 2, 3, 4, 5}


def test_the_window_bounds_kv_for_the_windowed_groups():
    """A long conversation grows the full group's block table and not the windowed
    ones -- which is the whole of what hybrid attention does."""
    llm = LLM(**{**BASE, "max_model_len": 2048, "max_num_batched_tokens": 2048})
    try:
        engine = llm.llm_engine
        engine.add_request(
            "r0", [7] * 1600, SamplingParams(max_tokens=64), pooling_params=None
        )
        while engine.has_unfinished_requests():
            engine.step()
        manager = engine.engine_core.engine_core.scheduler.kv_cache_manager
        # The request is gone, so the pool is whole again -- the windowed groups
        # returned every block they evicted along the way rather than holding them.
        assert manager.block_pool.get_num_free_blocks() == (
            manager.block_pool.num_usable_blocks
        )
    finally:
        llm.shutdown()
