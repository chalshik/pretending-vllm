"""KV cache manager allocation and release. R6.4--R6.7."""

from __future__ import annotations

import pytest

from pvllm.sampling_params import SamplingParams
from pvllm.v1.core.kv_cache_manager import KVCacheManager
from pvllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from pvllm.v1.request import Request


def make_config(num_blocks: int = 16, block_size: int = 4) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype="bfloat16",
        dtype_bytes=2,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
    )


def make_manager(num_blocks: int = 16, block_size: int = 4, max_model_len: int = 256):
    return KVCacheManager(
        make_config(num_blocks, block_size), max_model_len=max_model_len
    )


def make_request(request_id: str = "r0", prompt_len: int = 8) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=16),
        arrival_time=0.0,
    )


def test_allocation_rounds_up_to_whole_blocks():
    manager = make_manager(block_size=4)
    request = make_request(prompt_len=9)
    blocks = manager.allocate_slots(request, num_new_tokens=9)
    assert blocks is not None
    # 9 tokens over a block size of 4 needs 3 blocks.
    assert blocks.get_block_ids() == ([0, 1, 2],)


def test_allocation_is_exact_on_a_block_boundary():
    manager = make_manager(block_size=4)
    blocks = manager.allocate_slots(make_request(prompt_len=8), num_new_tokens=8)
    assert blocks is not None
    assert blocks.get_block_ids() == ([0, 1],)


def test_incremental_allocation_only_returns_the_new_blocks():
    """Decode steps append one token at a time; the scheduler needs to know which
    blocks are new so it can tell the worker (`CachedRequestData.new_block_ids`)."""
    manager = make_manager(block_size=4)
    request = make_request(prompt_len=4)

    first = manager.allocate_slots(request, num_new_tokens=4)
    assert first is not None and first.get_block_ids() == ([0],)

    request.num_computed_tokens = 4
    second = manager.allocate_slots(request, num_new_tokens=1)
    assert second is not None and second.get_block_ids() == ([1],)

    # Still inside block 1, so nothing new is needed.
    request.num_computed_tokens = 5
    third = manager.allocate_slots(request, num_new_tokens=1)
    assert third is not None and third.get_block_ids() == ([],)


def test_allocation_that_does_not_fit_returns_none_and_changes_nothing():
    """R6.5: fail early. A partial allocation would leave the request holding blocks
    it cannot use while starving the request that could have used them."""
    manager = make_manager(num_blocks=4, block_size=4)
    free_before = manager.block_pool.get_num_free_blocks()

    request = make_request(prompt_len=64)
    assert manager.allocate_slots(request, num_new_tokens=64) is None
    assert manager.block_pool.get_num_free_blocks() == free_before


def test_not_fitting_is_an_outcome_not_an_error():
    """The scheduler reads None as 'does not fit right now' and preempts (R5.5).
    Raising would turn a scheduling decision into a failed request."""
    manager = make_manager(num_blocks=2, block_size=4)
    assert manager.allocate_slots(make_request(prompt_len=100), 100) is None


def test_allocation_never_reserves_past_max_model_len():
    """A request at the length cap needs no further slots; rounding past it would
    allocate a block that can never be written."""
    manager = make_manager(num_blocks=64, block_size=4, max_model_len=8)
    request = make_request(prompt_len=8)
    blocks = manager.allocate_slots(request, num_new_tokens=8)
    assert blocks is not None
    assert blocks.get_block_ids() == ([0, 1],)

    request.num_computed_tokens = 8
    more = manager.allocate_slots(request, num_new_tokens=4)
    assert more is not None
    assert more.get_block_ids() == ([],)


def test_zero_token_allocation_is_rejected():
    manager = make_manager()
    with pytest.raises(ValueError, match="num_new_tokens=0"):
        manager.allocate_slots(make_request(), num_new_tokens=0)


def test_free_returns_every_block_the_request_held():
    manager = make_manager(num_blocks=8, block_size=4)
    request = make_request(prompt_len=8)
    manager.allocate_slots(request, num_new_tokens=8)
    assert manager.block_pool.get_num_free_blocks() == 6

    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == 8


def test_free_releases_tail_first():
    """R6.6/C2. The head of a sequence is what a later request is most likely to
    share, so it must outlive the tail in the free queue."""
    manager = make_manager(num_blocks=4, block_size=4)
    request = make_request(prompt_len=16)
    manager.allocate_slots(request, num_new_tokens=16)
    manager.free(request)

    order = [
        b.block_id for b in manager.block_pool.free_block_queue.get_all_free_blocks()
    ]
    assert order == [3, 2, 1, 0]


def test_freed_blocks_are_reused_in_eviction_order():
    manager = make_manager(num_blocks=4, block_size=4)
    first = make_request("a", prompt_len=16)
    manager.allocate_slots(first, num_new_tokens=16)
    manager.free(first)

    second = make_request("b", prompt_len=8)
    blocks = manager.allocate_slots(second, num_new_tokens=8)
    assert blocks is not None
    # Front of the queue after a reverse free is the old tail.
    assert blocks.get_block_ids() == ([3, 2],)


def test_two_requests_get_disjoint_blocks_without_caching():
    """No prefix cache means no sharing, even for identical prompts."""
    manager = make_manager(num_blocks=8, block_size=4)
    a = manager.allocate_slots(make_request("a", prompt_len=8), 8)
    b = manager.allocate_slots(make_request("b", prompt_len=8), 8)
    assert a is not None and b is not None
    assert set(a.get_block_ids()[0]).isdisjoint(b.get_block_ids()[0])


def test_prefix_cache_lookup_is_empty_until_m2():
    manager = make_manager()
    blocks, num_tokens = manager.get_computed_blocks(make_request())
    assert num_tokens == 0
    assert blocks.num_blocks == 0


def test_common_prefix_blocks_are_zero_without_a_cache():
    """Honest: without sharing nothing is common, which correctly disables cascade
    attention (R5.9)."""
    manager = make_manager()
    assert manager.get_num_common_prefix_blocks("r0", 4) == [0]


def test_block_ids_are_grouped_per_kv_cache_group():
    """R6.7: the tuple shape exists from the start even with one group."""
    manager = make_manager()
    blocks = manager.allocate_slots(make_request(prompt_len=8), 8)
    assert blocks is not None
    block_ids = blocks.get_block_ids()
    assert isinstance(block_ids, tuple)
    assert len(block_ids) == manager.num_kv_cache_groups == 1


def test_usage_reflects_the_pool():
    manager = make_manager(num_blocks=8, block_size=4)
    assert manager.usage == 0.0
    manager.allocate_slots(make_request(prompt_len=16), 16)
    assert manager.usage == pytest.approx(0.5)


def test_full_pool_then_free_then_reallocate_is_stable():
    """R21.1: every admitted request terminates, and the pool returns to empty."""
    manager = make_manager(num_blocks=8, block_size=4)
    requests = [make_request(f"r{i}", prompt_len=8) for i in range(4)]

    allocated = [r for r in requests if manager.allocate_slots(r, 8) is not None]
    assert len(allocated) == 4
    assert manager.usage == 1.0
    assert manager.allocate_slots(make_request("overflow", prompt_len=4), 4) is None

    for request in allocated:
        manager.free(request)
    assert manager.usage == 0.0
    assert manager.allocate_slots(make_request("after", prompt_len=4), 4) is not None
