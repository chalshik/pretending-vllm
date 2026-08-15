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
from typing import Any

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
        # `.get`, not `[...]`. `req_to_blocks` is a defaultdict, so indexing it here
        # would *insert* an empty entry for a request that is merely being considered
        # -- and `allocate_slots` calls this before deciding whether the request
        # fits. Every failed admission would leave a phantom holder behind, and
        # `get_num_common_prefix_blocks` counts holders, so the common-prefix count
        # would collapse to zero for the rest of the run. Upstream reads it the same
        # non-mutating way, for the same reason.
        num_held = len(self.req_to_blocks.get(request_id, ()))
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
        # Only requests that actually hold blocks count toward the denominator. An
        # empty entry means a request that asked and did not fit; including it would
        # make no block's ref_cnt ever equal the holder count.
        num_holders = sum(1 for blocks in self.req_to_blocks.values() if blocks)
        for block in self.req_to_blocks.get(running_request_id, ()):
            if block.ref_cnt != num_holders:
                break
            num_common_blocks += 1
        return num_common_blocks


class SlidingWindowManager(FullAttentionManager):
    """Attention over a bounded window. R6.7.

    Inherits the allocation arithmetic and adds the one thing that differs: blocks
    holding tokens that have fallen out of the window are freed, and their slots in
    the block table are replaced by the null block rather than removed. The table has
    to keep its length because positions index into it.

    What this buys is a different capacity story. Full attention grows KV linearly
    with conversation length, so a long-context deployment is bounded by *length*. A
    sliding window bounds each request at `sliding_window` tokens however long the
    conversation runs, so capacity is bounded by *concurrency* instead. A simulator
    that reported the full-attention number for a windowed model would overstate its
    memory by the ratio of context to window -- 32x on a 128k model with a 4k window.

    **Prefix caching is disabled for windowed groups**, as upstream does: a cached
    block is only reusable if the whole prefix leading to it is still attended to,
    and inside a window it usually is not. Reusing one would hand a request KV for
    tokens the model can no longer see.
    """

    def __init__(self, kv_cache_spec: KVCacheSpec, *args: Any, **kwargs: Any) -> None:
        super().__init__(kv_cache_spec, *args, **kwargs)
        from pvllm.v1.kv_cache_interface import SlidingWindowSpec

        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        self.sliding_window = kv_cache_spec.sliding_window

    def num_blocks_in_window(self) -> int:
        """Blocks a request needs once its context exceeds the window.

        `window - 1` because the token being generated attends to the `window - 1`
        before it plus itself, and a block boundary can fall anywhere in between --
        so one extra block is needed to cover a window that straddles two.
        """
        return (self.sliding_window - 1 + self.block_size - 1) // self.block_size + 1

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        """Free blocks that have fallen out of the window. R6.7.

        Called after each step's allocation, so a long-running request returns
        capacity as it goes rather than holding everything until it finishes -- which
        is the entire point of a window.
        """
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return

        # The first token still attended to, and therefore the first block still
        # needed. Everything before it is dead.
        first_live_token = max(0, num_computed_tokens - self.sliding_window)
        first_live_block = first_live_token // self.block_size

        null_block = self.block_pool.null_block
        assert null_block is not None, (
            "a sliding-window group needs the pool's null block; BlockPool must be "
            "constructed with reserve_null_block=True"
        )

        removed: list[KVCacheBlock] = []
        for index in range(min(first_live_block, len(blocks))):
            block = blocks[index]
            if block is null_block:
                # Released on an earlier step. Skipped, *not* broken on: nulls
                # accumulate from index 0 upward, so stopping at the first one would
                # mean nothing is ever evicted after the first eviction -- the bound
                # would hold for one step and then quietly stop holding.
                continue
            removed.append(block)
            blocks[index] = null_block

        if removed:
            # Reversed, like every other free path (R6.6): the most recently
            # allocated of the dead blocks goes back to the front of the queue.
            self.block_pool.free_blocks(reversed(removed))

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """Zero, always.

        A windowed group has no shared prefix to run cascade attention over: the
        blocks at the head of every request's table are null once the conversation
        passes the window, and a null block is shared by everyone for a reason that
        has nothing to do with content.
        """
        return 0


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    block_pool: BlockPool,
    kv_cache_group_id: int,
) -> SingleTypeKVCacheManager:
    """Pick the manager for a group's spec."""
    from pvllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    # Order matters if the specs ever share a base; checked most specific first.
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return SlidingWindowManager(kv_cache_spec, block_pool, kv_cache_group_id)
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return FullAttentionManager(kv_cache_spec, block_pool, kv_cache_group_id)
    raise NotImplementedError(
        f"no KV cache manager for {type(kv_cache_spec).__name__}; state-space "
        f"(Mamba) groups are not implemented"
    )
