"""Per-group block bookkeeping.

Upstream: vllm/v1/core/single_type_kv_cache_manager.py
Tier: A

One manager per KV cache group (R6.7). Only `FullAttentionManager` exists until hybrid
models land in M4, but the split is here from the start: retrofitting it would mean
touching every block-id call site in the scheduler and the runner, because the
tuple-of-lists shape in `SchedulerOutput.block_ids` only makes sense if the manager was
built around groups.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from pvllm.v1.core.block_pool import BlockPool
from pvllm.v1.core.kv_cache_utils import KVCacheBlock
from pvllm.v1.kv_cache_interface import KVCacheSpec


class SingleTypeKVCacheManager(ABC):
    """Block bookkeeping for one KV cache group."""

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
    ) -> None:
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_group_id = kv_cache_group_id

        #: request_id -> its blocks, in allocation order. The head of this list is
        #: the start of the sequence, which is why freeing in reverse evicts the
        #: tail first (R6.6).
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

    @abstractmethod
    def get_num_blocks_to_allocate(self, request_id: str, num_tokens: int) -> int:
        """How many *new* blocks holding `num_tokens` would need."""

    @abstractmethod
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        """Allocate and record the new blocks. Returns only the newly added ones."""

    def get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self.req_to_blocks.get(request_id, [])

    def adopt_cached_blocks(self, request_id: str, blocks: list[KVCacheBlock]) -> None:
        """Install prefix-cache hits at the head of a request's block table. R6.5.

        At the head, and only when the request holds nothing yet: a cache hit is by
        definition a *prefix*, so appending it after existing blocks would place the
        shared beginning of a sequence after its own middle.
        """
        held = self.req_to_blocks[request_id]
        if held:
            raise AssertionError(
                f"request {request_id} already holds {len(held)} blocks; cached "
                f"prefix blocks must be adopted before any are allocated"
            )
        held.extend(blocks)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """Remove the request's bookkeeping and hand back its blocks.

        Returns them in *allocation* order. The caller reverses before freeing, so
        the tail is evicted first (R6.6) -- doing the reverse here would hide that
        decision from the place it matters.
        """
        return self.req_to_blocks.pop(request_id, [])

    def free(self, request_id: str) -> None:
        """Release a request's blocks, tail first."""
        blocks = self.pop_blocks_for_free(request_id)
        if blocks:
            self.block_pool.free_blocks(reversed(blocks))

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """Blocks shared by every request holding KV cache, for cascade attention.

        R5.9. Takes any running request's id and walks its blocks; it does not take a
        request count, because the comparison upstream makes is against the number of
        requests *holding blocks*, which is not the same set as the ones scheduled
        this step.
        """


class FullAttentionManager(SingleTypeKVCacheManager):
    """Standard causal attention: every token needs a slot, forever.

    Sliding-window and state-space variants can drop blocks that fall out of scope;
    this one never can, which is what makes its accounting the simple case.
    """

    def get_num_blocks_to_allocate(self, request_id: str, num_tokens: int) -> int:
        num_required = (num_tokens + self.block_size - 1) // self.block_size
        num_held = len(self.req_to_blocks[request_id])
        return max(0, num_required - num_held)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        blocks = self.req_to_blocks[request_id]
        num_required = (num_tokens + self.block_size - 1) // self.block_size
        num_new = max(0, num_required - len(blocks))
        if num_new == 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new)
        blocks.extend(new_blocks)
        return new_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """The leading run of blocks that every KV-holding request shares. R5.9.

        A block is common when its `ref_cnt` equals the number of requests holding
        blocks at all. The walk stops at the first block that is not -- these are
        *prefix* blocks, so a shared block after an unshared one is not part of the
        common prefix and counting it would let a cascade kernel read across
        sequences.

        The count is a lower bound, and upstream says so too: a request that holds
        blocks but was not scheduled this step still counts in the denominator, so
        every scheduled request can share a prefix and this can still return 0. There
        is no cheap way to detect that, and under-reporting is the safe direction --
        it only forgoes an optimization.
        """
        num_common_blocks = 0
        num_holders = len(self.req_to_blocks)
        for block in self.req_to_blocks[running_request_id]:
            if block.ref_cnt != num_holders:
                break
            num_common_blocks += 1
        return num_common_blocks


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    block_pool: BlockPool,
    kv_cache_group_id: int,
) -> SingleTypeKVCacheManager:
    """Pick the manager for a group's spec."""
    from pvllm.v1.kv_cache_interface import FullAttentionSpec

    if isinstance(kv_cache_spec, FullAttentionSpec):
        return FullAttentionManager(kv_cache_spec, block_pool, kv_cache_group_id)
    raise NotImplementedError(
        f"no KV cache manager for {type(kv_cache_spec).__name__}; sliding-window and "
        f"state-space groups (requirement R6.7) land in M4"
    )
