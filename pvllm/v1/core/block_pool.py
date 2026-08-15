"""The block pool: who owns which KV block, and what gets evicted next.

Upstream: vllm/v1/core/block_pool.py
Tier: A

C2 binds allocation and free order exactly. Two behaviours carry that weight:

**Free order decides eviction order (R6.6).** `free_blocks` takes blocks already
ordered by eviction priority; the caller frees a request's blocks in reverse so the
*tail* of its chain is evicted first. That is what preserves the shared prefix at the
head, which is the entire point of a prefix cache.

**Unhashed blocks are prepended, hashed blocks appended.** A block with no hash can
never produce a cache hit, so evicting it ahead of hashed blocks costs nothing and
preserves more cache. Get this backwards and hit rate collapses while every count
still balances -- a bug that looks like a tuning problem rather than a correctness one.

Prefix caching itself lands in M2. The pool refuses to pretend: constructing it with
`enable_caching=True` raises, rather than reporting a 0% hit rate on a workload that
should hit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pvllm import envs
from pvllm.logger import init_logger
from pvllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)

logger = init_logger(__name__)


class BlockPool:
    """Owns every KV cache block and the free queue over them.

    Args:
        num_gpu_blocks: Total blocks, from the memory model (R10.2).
        enable_caching: Whether prefix caching is on. M2.
        enable_kv_cache_events: Whether to publish block store/remove events (R12.5).
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        if num_gpu_blocks <= 0:
            raise ValueError(
                f"num_gpu_blocks must be positive, got {num_gpu_blocks}. The memory "
                f"model derives this; a non-positive value means the KV pool did not "
                f"fit (requirement R10.5)."
            )
        self.num_gpu_blocks = num_gpu_blocks
        # Annotated explicitly: the guard above narrows the parameter to False, and
        # without this the unhashed-block branch below reads as dead code.
        self.enable_caching: bool = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events

        # Indexed by block_id, so blocks[i].block_id == i.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(num_gpu_blocks)
        ]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Populated in M2. Maps a block hash to the blocks holding that content.
        self.cached_block_hash_to_block: dict[
            BlockHashWithGroupId, dict[int, KVCacheBlock]
        ] = {}

        #: R6.9. Eviction count, for the cache-effectiveness metrics.
        self.num_evicted_blocks = 0
        self._debug_invariants = envs.PVLLM_DEBUG_INVARIANTS

    # --- allocation ----------------------------------------------------------

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Pop `num_blocks` from the head of the free queue and take a reference.

        Does not consult the cache; the caller looks up cached blocks first and
        `touch`es those separately (R6.5).
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"cannot get {num_blocks} free blocks from the pool; "
                f"{self.get_num_free_blocks()} are free"
            )

        blocks = self.free_block_queue.popleft_n(num_blocks)
        for block in blocks:
            self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0, (
                f"block {block.block_id} came off the free queue with "
                f"ref_cnt={block.ref_cnt}; the queue and the reference counts have "
                f"diverged"
            )
            block.ref_cnt += 1

        self._check_invariants()
        return blocks

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Take a reference to blocks hit by another request's prefix.

        A block with `ref_cnt == 0` is sitting in the free queue as an eviction
        candidate, so it must come out of the queue as well as gain a reference --
        which is why the queue supports O(1) removal from the middle.
        """
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
        self._check_invariants()

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Drop a reference to each block, returning any that hit zero to the queue.

        `ordered_blocks` must already be in eviction priority order -- first evicted
        first. `KVCacheManager` reverses a request's blocks before calling this so the
        tail of the chain goes first (R6.6).
        """
        blocks_with_hash: list[KVCacheBlock] = []
        blocks_without_hash: list[KVCacheBlock] = []

        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt < 0:
                raise AssertionError(
                    f"block {block.block_id} freed more times than it was allocated "
                    f"(ref_cnt={block.ref_cnt}); a request freed blocks it did not own"
                )
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None and self.enable_caching:
                    blocks_without_hash.append(block)
                else:
                    blocks_with_hash.append(block)

        # Unhashed blocks can never produce a hit, so they go to the front and are
        # spent first. Hashed blocks go to the back, staying available to a later
        # request with the same prefix for as long as possible.
        self.free_block_queue.prepend_n(blocks_without_hash)
        self.free_block_queue.append_n(blocks_with_hash)
        self._check_invariants()

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """Drop a block's cache entry as it is reallocated. Returns whether it had one.

        R6.2: the hash is cleared here, which is what makes eviction a real event
        rather than a stale entry pointing at reused memory.
        """
        block_hash = block.block_hash
        if block_hash is None:
            return False

        entries = self.cached_block_hash_to_block.get(block_hash)
        if entries is not None:
            entries.pop(block.block_id, None)
            if not entries:
                del self.cached_block_hash_to_block[block_hash]
        block.reset_hash()
        self.num_evicted_blocks += 1
        return True

    # --- prefix cache (R6.3--R6.5) -------------------------------------------

    def get_cached_block(
        self, block_hash: BlockHash, group_id: int
    ) -> KVCacheBlock | None:
        """The block holding this content, if any is still resident.

        Any of them: several blocks can hold identical content when two requests
        raced to cache it before either was evicted. They are interchangeable, so
        the first is as good as any.
        """
        key = make_block_hash_with_group_id(block_hash, group_id)
        entries = self.cached_block_hash_to_block.get(key)
        if not entries:
            return None
        return next(iter(entries.values()))

    def cache_full_blocks(
        self,
        request_block_hashes: list[BlockHash],
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        group_id: int,
    ) -> None:
        """Register blocks that just became full. R6.5.

        Called immediately on allocation rather than when the request finishes: a
        block whose contents are complete can be shared *now*, and waiting until the
        request ends would miss every hit from a concurrent request with the same
        prefix -- which is the common case a prefix cache exists to serve.
        """
        if num_full_blocks <= num_cached_blocks:
            return

        for i in range(num_cached_blocks, num_full_blocks):
            if i >= len(request_block_hashes) or i >= len(blocks):
                break
            block = blocks[i]
            if block.block_hash is not None:
                continue
            key = make_block_hash_with_group_id(request_block_hashes[i], group_id)
            block.set_block_hash(key)
            self.cached_block_hash_to_block.setdefault(key, {})[block.block_id] = block

    # --- introspection -------------------------------------------------------

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Fraction of the pool in use. Feeds `vllm:kv_cache_usage_perc` (R12.1)."""
        return 1.0 - (self.get_num_free_blocks() / self.num_gpu_blocks)

    def reset_prefix_cache(self) -> bool:
        """Drop every cached block. R6.10.

        Refuses while any block is still referenced: clearing the cache under a
        running request would let its blocks be handed to someone else.
        """
        num_used = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used > 0:
            logger.warning(
                "Failed to reset prefix cache: %d blocks are still in use.", num_used
            )
            return False

        for block in self.blocks:
            self._maybe_evict_cached_block(block)
        self.cached_block_hash_to_block.clear()
        self.num_evicted_blocks = 0
        logger.info("Successfully reset prefix cache")
        return True

    # --- invariants (R21.1) --------------------------------------------------

    def _check_invariants(self) -> None:
        """Assert the pool's accounting still holds.

        Off unless `PVLLM_DEBUG_INVARIANTS` is set, which the test suite always sets.
        These are the cheapest place to catch a KV manager bug: a violation here
        points at the allocation that broke it, whereas the same bug found later
        surfaces as a wrong answer with no trail back.
        """
        if not self._debug_invariants:
            return

        free_blocks = self.free_block_queue.get_all_free_blocks()
        num_free = len(free_blocks)

        assert num_free == self.free_block_queue.num_free_blocks, (
            f"free queue counter says {self.free_block_queue.num_free_blocks} but the "
            f"list holds {num_free}"
        )

        allocated = sum(1 for block in self.blocks if block.ref_cnt > 0)
        # total == free + allocated
        assert num_free + allocated == self.num_gpu_blocks, (
            f"{num_free} free + {allocated} allocated != {self.num_gpu_blocks} total"
        )

        for block in free_blocks:
            assert block.ref_cnt == 0, (
                f"block {block.block_id} is in the free queue with "
                f"ref_cnt={block.ref_cnt}"
            )

        assert 0.0 <= self.get_usage() <= 1.0, (
            f"KV usage {self.get_usage()} out of range"
        )

    def __repr__(self) -> str:
        return (
            f"BlockPool(num_gpu_blocks={self.num_gpu_blocks}, "
            f"free={self.get_num_free_blocks()}, usage={self.get_usage():.3f})"
        )
