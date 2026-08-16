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

    pool: list[int] = []

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
            pool.append(profile.num_gpu_blocks)
            return profile.max_concurrency
        finally:
            engine.shutdown()

    full = concurrency(None)
    wide = concurrency(4096)
    narrow = concurrency(1024)
    # A window changes what a request holds, not how big the pool is.
    assert len(set(pool)) == 1
    blocks = pool[0]

    # Not 8x and 32x, and this test has twice encoded an overstated figure. What a
    # request *holds* is the window plus one step's token budget: eviction runs after
    # the step has allocated slots for everything it scheduled, so the whole prefill
    # chunk is resident alongside the window. That term dominates a narrow window, so
    # the gain is real and far below the ratio of the windows themselves.
    #
    # Pinned to the arithmetic rather than to a band, because a band is what let this
    # pass for two wrong values in a row: the pre-fix steady-state figures were 7.97x
    # and 31.51x, which sit inside any bound loose enough to admit the true 2.66x and
    # 3.55x. Every term below is independently derivable -- 32768 tokens over a
    # 16-token block for the unbounded case, `windowed_blocks_for_one_request` for the
    # other two, less the reserved null block -- so an edit that re-breaks the peak
    # moves one side and not the other.
    from pvllm.sim.memory import windowed_blocks_for_one_request

    budget = 8192  # dense-8b's default max_num_batched_tokens at this max_model_len
    assert full == pytest.approx(blocks / (32768 // 16), rel=1e-3)
    for window, measured in ((4096, wide), (1024, narrow)):
        held = windowed_blocks_for_one_request(window, 16, 32768, budget)
        assert measured == pytest.approx((blocks - 1) / held, rel=1e-3), window
    assert 2.0 < wide / full < 3.5
    assert 1.0 < narrow / wide < 1.6


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


def test_a_window_smaller_than_a_step_budget_is_refused_not_hung():
    """R10.6. The peak a windowed request passes through is its window *plus* one
    step's token budget: eviction runs in `update_from_output`, after the step has
    already allocated slots for everything it scheduled.

    Under-counting that is not a small error in a reported figure -- it is a silent
    hang. The pool cleared the steady state, so startup passed and reported a
    concurrency figure; then `allocate_slots` returned `None` every step forever, with
    no error and no log line. The request computed zero tokens and the engine never
    stopped asking.
    """
    from pvllm.sim.memory import SimOutOfMemoryError, windowed_blocks_for_one_request

    # A 64-token window against a 1024-token step budget needs 69 blocks, not 5.
    assert windowed_blocks_for_one_request(64, 16, 2048, 1024) == 69
    assert windowed_blocks_for_one_request(64, 16, 2048, 0) == 5

    with pytest.raises(SimOutOfMemoryError, match="needs 69"):
        LLM(
            model="tiny-test",
            device_card="tiny-2gb",
            max_model_len=2048,
            block_size=16,
            max_num_batched_tokens=1024,
            max_num_seqs=1,
            sliding_window=64,
            num_gpu_blocks_override=16,
            disable_log_stats=True,
        ).shutdown()

    # And a pool that does clear the peak serves the request rather than stalling.
    llm = LLM(
        model="tiny-test",
        device_card="tiny-2gb",
        max_model_len=2048,
        block_size=16,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        sliding_window=64,
        num_gpu_blocks_override=128,
        disable_log_stats=True,
    )
    try:
        engine = llm.llm_engine
        engine.add_request(
            "r0", [7] * 1500, SamplingParams(max_tokens=8), pooling_params=None
        )
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            engine.step()
        assert not engine.has_unfinished_requests()
    finally:
        llm.shutdown()
