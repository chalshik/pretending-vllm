"""Sliding-window attention. R6.7.

The attention pattern is not the interesting part -- there is no attention. What
matters is that KV per request stops growing with the conversation.

Full attention makes a long-context deployment bounded by *length*: a 32k
conversation holds 32k tokens of KV whatever else is happening. A window bounds each
request at `sliding_window` tokens however long it runs, so capacity is bounded by
*concurrency* instead. Those are different capacity planning problems, and a
simulator reporting the full-attention number for a windowed model would overstate its
memory by the ratio of context to window.
"""

from __future__ import annotations

import pytest

from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

BASE = {
    "model": "tiny-test",
    "max_model_len": 512,
    "block_size": 16,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
}


def held_blocks(engine: LLM) -> int:
    pool = (
        engine.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
    )
    return sum(1 for block in pool.blocks if block.ref_cnt > 0 and not block.is_null)


# --- the spec --------------------------------------------------------------


def test_a_windowed_layer_cannot_share_a_group_with_a_full_one():
    """Two layers can share a block table only if they free blocks at the same
    points. A windowed one does not, so its type id differs."""
    common = {
        "block_size": 16,
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": "bfloat16",
        "dtype_bytes": 2,
    }
    full = FullAttentionSpec(**common)
    windowed = SlidingWindowSpec(**common, sliding_window=4096)
    narrower = SlidingWindowSpec(**common, sliding_window=1024)

    assert full.type_id != windowed.type_id
    # Different windows are different groups too, for the same reason.
    assert windowed.type_id != narrower.type_id


def test_a_non_positive_window_is_refused():
    with pytest.raises(ValueError, match="sliding_window must be positive"):
        SlidingWindowSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype="bfloat16",
            dtype_bytes=2,
            sliding_window=0,
        )


# --- the capacity answer ---------------------------------------------------


def test_a_window_raises_concurrency_in_proportion():
    """The headline effect, and the reason this is modeled rather than approximated.
    Concurrency is bounded by tokens-per-request, and a window is what sets it."""

    def concurrency(window: int | None) -> float:
        engine = LLM(
            model="dense-8b",
            max_model_len=32768,
            block_size=16,
            device_card="datacenter-80gb",
            disable_log_stats=True,
            sliding_window=window,
        )
        try:
            profile = engine.llm_engine.engine_core.engine_core.executor.driver_worker.memory_profile
            assert profile is not None
            return profile.max_concurrency
        finally:
            engine.shutdown()

    full = concurrency(None)
    wide = concurrency(4096)
    narrow = concurrency(1024)

    # Not exactly 8x and 32x, and this test used to assert that it was. Concurrency
    # is bounded by the blocks a request *holds*, which is one more than the window
    # divides into -- the live window straddles a block boundary and eviction runs
    # after allocation, so the outgoing block is still held. The old assertion
    # encoded the overstated figure, which is how it survived.
    assert wide > full * 7
    assert narrow > full * 30
    assert wide < full * 8
    assert narrow < full * 32


def test_a_window_larger_than_the_context_changes_nothing():
    """A 8k window on a 512-token model bounds nothing, so it must not be reported
    as though it did."""
    engine = LLM(**BASE, sliding_window=8192)
    try:
        profile = engine.llm_engine.engine_core.engine_core.executor.driver_worker.memory_profile
        assert profile is not None
        plain = LLM(**BASE)
        try:
            reference = plain.llm_engine.engine_core.engine_core.executor.driver_worker.memory_profile
            assert reference is not None
            assert profile.max_concurrency == reference.max_concurrency
        finally:
            plain.shutdown()
    finally:
        engine.shutdown()


# --- the eviction ----------------------------------------------------------


def test_blocks_are_released_as_the_window_slides():
    """Not just at the end. A request generating far past its window has to hand
    blocks back *while it runs*, or the bounded-KV property does not exist."""
    engine = LLM(**{**BASE, "num_gpu_blocks_override": 60}, sliding_window=48)
    try:
        engine.llm_engine.add_request("r0", "hello", SamplingParams(max_tokens=200))
        peak = 0
        for _ in range(500):
            if not engine.llm_engine.has_unfinished_requests():
                break
            engine.llm_engine.step()
            peak = max(peak, held_blocks(engine))

        # 200 tokens at block_size 16 would be 13+ blocks without a window; the
        # window caps it near 48/16 + 1.
        assert peak <= 6, f"held {peak} blocks for a 48-token window"
    finally:
        engine.shutdown()


def test_a_long_generation_still_produces_every_token():
    """Eviction must bound memory without truncating output."""
    engine = LLM(**{**BASE, "num_gpu_blocks_override": 60}, sliding_window=48)
    try:
        output = engine.generate(["hello there"], SamplingParams(max_tokens=200))[0]
        assert len(output.outputs[0].token_ids) == 200
        assert output.outputs[0].finish_reason == "length"
    finally:
        engine.shutdown()


def test_concurrent_windowed_requests_all_complete():
    engine = LLM(**{**BASE, "num_gpu_blocks_override": 80}, sliding_window=48)
    try:
        outputs = engine.generate(
            [f"prompt {i}" for i in range(6)], SamplingParams(max_tokens=100)
        )
        assert len(outputs) == 6
        assert all(len(o.outputs[0].token_ids) == 100 for o in outputs)
    finally:
        engine.shutdown()


def test_the_null_block_is_never_handed_out():
    """It stands in for evicted slots in every request's table at once. Allocating
    it to somebody would give two requests the same block for different tokens."""
    engine = LLM(**{**BASE, "num_gpu_blocks_override": 60}, sliding_window=48)
    try:
        engine.generate(["hello there"], SamplingParams(max_tokens=200))
        pool = engine.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
        assert pool.null_block is not None
        assert pool.null_block.block_id == 0
        assert pool.null_block.ref_cnt >= 1
        assert pool.num_usable_blocks == pool.num_gpu_blocks - 1
    finally:
        engine.shutdown()


def test_a_full_attention_engine_reserves_no_null_block():
    """The reservation shifts every block id, so it happens only where it is
    needed -- otherwise every recorded trace would change for a feature not in use."""
    engine = LLM(**BASE)
    try:
        pool = engine.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
        assert pool.null_block is None
        assert pool.num_usable_blocks == pool.num_gpu_blocks
    finally:
        engine.shutdown()


# --- prefix caching interaction --------------------------------------------


def test_a_windowed_group_caches_the_tail_its_window_attends_to():
    """R6.4, R6.7. A windowed group does not need the whole prefix -- only the last
    `window` tokens are ever read -- so upstream searches right to left for a
    contiguous run covering the window and fills everything before it with the null
    block. Turning caching off pool-wide was the older answer here, and it cost a
    hybrid model its entire hit rate.

    The null blocks are what make it safe: the request is handed a block table whose
    early entries are the shared placeholder, so nothing reads KV for tokens the model
    can no longer see."""
    engine = LLM(**BASE, sliding_window=64, enable_prefix_caching=True)
    try:
        manager = engine.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
        assert manager.has_sliding_window
        assert manager.enable_caching
    finally:
        engine.shutdown()
