"""Block pool and free-queue ordering. R6.1, R6.2, R6.5, R6.6, R21.1."""

from __future__ import annotations

import pytest

from pvllm.v1.core.block_pool import BlockPool
from pvllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


def ids(blocks) -> list[int]:
    return [b.block_id for b in blocks]


# --- the free queue --------------------------------------------------------


def test_queue_starts_ordered_by_block_id():
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(5)])
    assert ids(queue.get_all_free_blocks()) == [0, 1, 2, 3, 4]
    assert queue.num_free_blocks == 5


def test_empty_queue_refuses_to_pop():
    queue = FreeKVCacheBlockQueue([])
    with pytest.raises(ValueError, match="No free blocks"):
        queue.popleft()


def test_popleft_takes_from_the_front():
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(4)])
    assert queue.popleft().block_id == 0
    assert queue.popleft().block_id == 1
    assert ids(queue.get_all_free_blocks()) == [2, 3]
    assert queue.num_free_blocks == 2


def test_popleft_n_matches_repeated_popleft():
    batch = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(6)])
    one_at_a_time = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(6)])

    assert ids(batch.popleft_n(3)) == ids([one_at_a_time.popleft() for _ in range(3)])
    assert ids(batch.get_all_free_blocks()) == ids(one_at_a_time.get_all_free_blocks())


def test_popleft_n_of_zero_is_a_no_op():
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(3)])
    assert queue.popleft_n(0) == []
    assert queue.num_free_blocks == 3


def test_remove_from_the_middle_is_supported():
    """The reason this is a linked list and not a deque.

    `touch` must pull a cached block out of the middle of the free queue in O(1)
    when a second request hits it.
    """
    blocks = [KVCacheBlock(i) for i in range(5)]
    queue = FreeKVCacheBlockQueue(blocks)
    queue.remove(blocks[2])
    assert ids(queue.get_all_free_blocks()) == [0, 1, 3, 4]
    assert queue.num_free_blocks == 4


def test_removing_a_block_not_in_the_queue_is_an_error():
    blocks = [KVCacheBlock(i) for i in range(3)]
    queue = FreeKVCacheBlockQueue(blocks)
    popped = queue.popleft()
    with pytest.raises(RuntimeError, match="not in the free list"):
        queue.remove(popped)


def test_append_goes_to_the_back_and_prepend_to_the_front():
    blocks = [KVCacheBlock(i) for i in range(4)]
    queue = FreeKVCacheBlockQueue(blocks)
    taken = queue.popleft_n(2)

    queue.append_n(taken)
    assert ids(queue.get_all_free_blocks()) == [2, 3, 0, 1]

    more = queue.popleft_n(2)  # [2, 3]
    queue.prepend_n(more)
    assert ids(queue.get_all_free_blocks()) == [2, 3, 0, 1]


def test_queue_survives_being_emptied_and_refilled():
    blocks = [KVCacheBlock(i) for i in range(3)]
    queue = FreeKVCacheBlockQueue(blocks)
    taken = queue.popleft_n(3)
    assert queue.num_free_blocks == 0
    assert queue.get_all_free_blocks() == []

    queue.append_n(taken)
    assert ids(queue.get_all_free_blocks()) == [0, 1, 2]
    assert queue.num_free_blocks == 3


# --- the pool --------------------------------------------------------------


def test_pool_refuses_a_nonpositive_size():
    """R10.5: a KV pool that did not fit is a startup error, not a runtime surprise."""
    with pytest.raises(ValueError, match="num_gpu_blocks must be positive"):
        BlockPool(0)


def test_caching_a_full_block_makes_it_findable():
    """R6.5: registered on allocation, not at request end -- a full block can be
    shared now, and waiting would miss every hit from a concurrent request."""
    from pvllm.v1.core.kv_cache_utils import BlockHash

    pool = BlockPool(8, enable_caching=True)
    blocks = pool.get_new_blocks(2)
    hashes = [BlockHash(b"aaa"), BlockHash(b"bbb")]

    pool.cache_full_blocks(hashes, blocks, 0, 2, group_id=0)
    assert pool.get_cached_block(hashes[0], group_id=0) is blocks[0]
    assert pool.get_cached_block(hashes[1], group_id=0) is blocks[1]
    assert pool.get_cached_block(BlockHash(b"zzz"), group_id=0) is None


def test_a_block_hash_is_scoped_to_its_kv_cache_group():
    """R6.7: two groups can produce the same digest for different content."""
    from pvllm.v1.core.kv_cache_utils import BlockHash

    pool = BlockPool(8, enable_caching=True)
    blocks = pool.get_new_blocks(1)
    pool.cache_full_blocks([BlockHash(b"aaa")], blocks, 0, 1, group_id=0)

    assert pool.get_cached_block(BlockHash(b"aaa"), group_id=0) is blocks[0]
    assert pool.get_cached_block(BlockHash(b"aaa"), group_id=1) is None


def test_reallocating_a_cached_block_evicts_its_entry():
    """R6.2: the hash is cleared on eviction, so no entry ever points at reused
    memory."""
    from pvllm.v1.core.kv_cache_utils import BlockHash

    pool = BlockPool(2, enable_caching=True)
    blocks = pool.get_new_blocks(1)
    block_hash = BlockHash(b"aaa")
    pool.cache_full_blocks([block_hash], blocks, 0, 1, group_id=0)
    pool.free_blocks(blocks)

    # Drain the pool so the cached block must be reused.
    pool.get_new_blocks(2)
    assert pool.get_cached_block(block_hash, group_id=0) is None
    assert pool.num_evicted_blocks >= 1


def test_allocation_takes_a_reference_and_shrinks_the_free_pool():
    pool = BlockPool(8)
    blocks = pool.get_new_blocks(3)
    assert ids(blocks) == [0, 1, 2]
    assert all(b.ref_cnt == 1 for b in blocks)
    assert pool.get_num_free_blocks() == 5


def test_over_allocation_is_refused():
    pool = BlockPool(4)
    with pytest.raises(ValueError, match="cannot get 5 free blocks"):
        pool.get_new_blocks(5)


def test_freeing_returns_blocks_and_restores_the_count():
    pool = BlockPool(4)
    blocks = pool.get_new_blocks(2)
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 4
    assert all(b.ref_cnt == 0 for b in blocks)


def test_reverse_free_puts_the_tail_at_the_front_of_the_queue():
    """R6.6, and the reason `free` reverses.

    A request's blocks are freed tail-first so the *head* of its sequence -- the part
    a later request is most likely to share -- survives longest in the free queue.
    """
    pool = BlockPool(4)
    blocks = pool.get_new_blocks(4)  # ids 0..3, head to tail
    pool.free_blocks(reversed(blocks))
    assert ids(pool.free_block_queue.get_all_free_blocks()) == [3, 2, 1, 0]


def test_shared_blocks_are_only_returned_when_the_last_holder_frees():
    pool = BlockPool(4)
    blocks = pool.get_new_blocks(2)
    pool.touch(blocks)  # a second request hits the same prefix
    assert all(b.ref_cnt == 2 for b in blocks)

    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 2  # still held by the first request
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 4


def test_touch_pulls_a_free_block_back_out_of_the_queue():
    pool = BlockPool(4)
    blocks = pool.get_new_blocks(2)
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 4

    pool.touch(blocks)
    assert pool.get_num_free_blocks() == 2
    assert all(b.ref_cnt == 1 for b in blocks)


def test_double_free_is_caught():
    """R21.1: no negative ref_cnt. Silently going negative would let a block be
    handed to two requests at once."""
    pool = BlockPool(4)
    blocks = pool.get_new_blocks(1)
    pool.free_blocks(blocks)
    with pytest.raises(AssertionError, match="freed more times than it was allocated"):
        pool.free_blocks(blocks)


def test_usage_tracks_allocation():
    pool = BlockPool(10)
    assert pool.get_usage() == 0.0
    pool.get_new_blocks(5)
    assert pool.get_usage() == pytest.approx(0.5)
    pool.get_new_blocks(5)
    assert pool.get_usage() == pytest.approx(1.0)


def test_invariants_hold_across_a_churn_of_allocations():
    """R21.1: total == free + allocated, no negative ref_cnt, usage in [0, 1].

    Asserted inside the pool on every mutation when PVLLM_DEBUG_INVARIANTS is set,
    which conftest does for the whole suite.
    """
    pool = BlockPool(32)
    held: list[list] = []
    for i in range(1, 9):
        held.append(pool.get_new_blocks(i % 4 + 1))
        if len(held) > 2:
            pool.free_blocks(reversed(held.pop(0)))
    for blocks in held:
        pool.free_blocks(reversed(blocks))
    assert pool.get_num_free_blocks() == 32
    assert pool.get_usage() == 0.0


def test_reset_prefix_cache_refuses_while_blocks_are_in_use():
    """Clearing the cache under a running request would let its blocks be handed
    to someone else."""
    pool = BlockPool(4)
    blocks = pool.get_new_blocks(1)
    assert pool.reset_prefix_cache() is False
    pool.free_blocks(blocks)
    assert pool.reset_prefix_cache() is True
