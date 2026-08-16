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
from collections.abc import Sequence
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

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[Any],
        max_length: int,
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        group_id: int,
    ) -> tuple[list[KVCacheBlock], int]:
        """The longest prefix of `block_hashes` this group already holds. R6.4, C3.

        Per *type*, because the two types answer differently: full attention wants
        the longest run from the start, while a windowed group only needs the tail
        that its window still attends to. Returns `(blocks, hit_tokens)`.
        """
        raise NotImplementedError

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock] = (),
        total_computed_tokens: int = 0,
    ) -> int:
        """Blocks the pool must hand over for this request to hold `num_tokens`.

        Ported from upstream's `SingleTypeKVCacheManager.get_num_blocks_to_allocate`
        rather than derived, because the two things it does beyond the obvious
        subtraction are exactly the two that are wrong when you derive it.

        First, `new_computed_blocks` counts in *full*, null placeholders included. A
        windowed or state-space group's cache hit is mostly the shared null block,
        and `adopt_cached_blocks` installs the whole list -- so those slots are
        already filled and `allocate_new_blocks` will not ask for them. Charging them
        was a silent hang: a request whose prefix *hit* could be refused forever on a
        pool that was entirely free, which is the failure a cache hit should be least
        able to cause.

        Second, `num_skipped_blocks` discounts the head a group no longer reads --
        the prefix outside a window, every state but the newest. Without it a group
        that sheds blocks is still charged for the ones it shed.

        `total_computed_tokens` is the prefix length after this step's hits are
        counted; it is what decides how much of the head is skipped.
        """
        block_size = self.block_size
        num_required_blocks = -(-num_tokens // block_size)
        # `.get`, not `[...]`. `req_to_blocks` is a defaultdict, so indexing it here
        # would *insert* an empty entry for a request that is merely being considered
        # -- and `allocate_slots` calls this before deciding whether the request
        # fits. Every failed admission would leave a phantom holder behind, and
        # `get_num_common_prefix_blocks` counts holders, so the common-prefix count
        # would collapse to zero for the rest of the run. Upstream reads it the same
        # non-mutating way, for the same reason.
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        num_skipped_blocks = (
            self.get_num_skipped_tokens(total_computed_tokens) // block_size
        )
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0
        )

        # A cached block with no other holder is *in* the free queue, and `touch`
        # takes it out. So it draws on the same free pool the new blocks do, and
        # counting it as still available would let an allocation pass the check and
        # then run the queue dry partway through -- a crash rather than the `None`
        # the scheduler knows how to handle. The ones the head skips are not touched
        # and so are not charged.
        num_skipped_new_computed = max(0, num_skipped_blocks - num_req_blocks)
        num_evictable_blocks = sum(
            1
            for block in new_computed_blocks[num_skipped_new_computed:]
            if block.ref_cnt == 0 and not block.is_null
        )
        return num_new_blocks + num_evictable_blocks

    @abstractmethod
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        """Allocate and record the new blocks. Returns only the newly added ones."""

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """How much of the computed prefix this group no longer reads.

        Zero for full attention, which is what makes its accounting the simple case:
        every token it ever computed is still attended to. A window skips everything
        older than itself; a recurrent state skips everything but the last token.
        `remove_skipped_blocks` and `get_num_blocks_to_allocate` are both written
        against this one number, so a new group type gets both behaviours by
        answering it.
        """
        return 0

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        """Free the head this group no longer reads, replacing it with the null block.

        A no-op wherever `get_num_skipped_tokens` returns 0, which is why full
        attention pays nothing for it. For the groups that do shed, it is the whole
        mechanism: without it a long-running request holds every block it ever
        touched, and the bounded-KV property that makes windows and recurrent states
        worth having would not exist.

        The table keeps its length -- freed slots become the null block rather than
        disappearing -- because positions index into it.
        """
        num_skipped_tokens = self.get_num_skipped_tokens(num_computed_tokens)
        if num_skipped_tokens <= 0:
            return
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return

        null_block = self.block_pool.null_block
        assert null_block is not None, (
            f"{type(self).__name__} sheds blocks and so needs the pool's null block; "
            f"BlockPool must be constructed with reserve_null_block=True"
        )

        num_skipped_blocks = min(num_skipped_tokens // self.block_size, len(blocks))
        removed: list[KVCacheBlock] = []
        for index in range(num_skipped_blocks):
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

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[Any],
        max_length: int,
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        group_id: int,
    ) -> tuple[list[KVCacheBlock], int]:
        """The longest run of cached blocks from the start. R6.4.

        A miss ends the search: hashes are chained through their parent, so a block
        that is not cached guarantees every block after it is not either.
        """
        block_size = kv_cache_spec.block_size
        computed: list[KVCacheBlock] = []
        for block_hash in block_hashes[: max_length // block_size]:
            cached = block_pool.get_cached_block(block_hash, group_id=group_id)
            if cached is None:
                break
            computed.append(cached)
        return computed, len(computed) * block_size


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

    A windowed group *does* cache -- this paragraph used to say the opposite, and it
    outlived the change that made it wrong. What it caches is not a prefix but the
    tail its window still attends to, which is why `find_longest_cache_hit` searches
    right to left and fills everything before the run with the null block. Refusing
    to cache would have been the simple answer and it cost a hybrid model its whole
    hit rate, because the coordinator reconciles across every group.
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

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[Any],
        max_length: int,
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        group_id: int,
    ) -> tuple[list[KVCacheBlock], int]:
        """The tail this window still attends to. R6.4, R6.7.

        A windowed group does not need the whole prefix -- only the last `window`
        tokens are ever read -- so upstream searches *right to left* for a contiguous
        run long enough to cover the window, and fills everything before it with the
        null block. That is why a hybrid model can hit at all: the full-attention
        groups need the prefix from token zero, the windowed ones need only a window's
        worth ending at the same place.
        """
        from pvllm.v1.kv_cache_interface import SlidingWindowSpec

        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        block_size = kv_cache_spec.block_size
        # `window - 1` for the same reason `num_blocks_in_window` uses it: the token
        # being generated attends to the `window - 1` before it plus itself.
        needed = -(-(kv_cache_spec.sliding_window - 1) // block_size)
        max_num_blocks = max_length // block_size
        null_block = block_pool.null_block
        assert null_block is not None, (
            "a sliding-window group needs the reserved null block to stand in for "
            "the prefix its window no longer attends to (R6.7)"
        )
        computed: list[KVCacheBlock] = [null_block] * max_num_blocks

        contiguous = 0
        for index in range(max_num_blocks - 1, -1, -1):
            cached = block_pool.get_cached_block(block_hashes[index], group_id=group_id)
            if cached is None:
                contiguous = 0
                continue
            computed[index] = cached
            contiguous += 1
            if contiguous >= needed:
                # Trim whatever followed the run; the hit ends where it ends.
                del computed[index + contiguous :]
                return computed, len(computed) * block_size
        # No run long enough. Whatever contiguous prefix exists is still a hit.
        del computed[contiguous:]
        return computed, len(computed) * block_size

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Everything older than the window. R6.7.

        The base `remove_skipped_blocks` floors this to a block boundary, so the
        first block still needed is `max(0, n - window) // block_size` -- which is
        what this used to compute inline, and is unchanged by moving it here.
        """
        return max(0, num_computed_tokens - self.sliding_window)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """Zero, always.

        A windowed group has no shared prefix to run cascade attention over: the
        blocks at the head of every request's table are null once the conversation
        passes the window, and a null block is shared by everyone for a reason that
        has nothing to do with content.
        """
        return 0


class MambaManager(FullAttentionManager):
    """Blocks for a state-space group. R6.7.

    A recurrent state is *one* page, not a page per block boundary. The block table
    still has an entry per boundary -- positions index into it -- but every entry
    below the newest is the null block, so what a request actually holds is a single
    live state however long its context runs. That is the entire capacity argument
    for a state-space layer, and getting it wrong charges a Mamba group as if it were
    linear in context, which is the shape it exists not to have.

    Two things make that true, and both are answers to questions the base class asks:
    `get_num_skipped_tokens` (everything but the last token) and, through it,
    `remove_skipped_blocks` and `get_num_blocks_to_allocate`. The third,
    `find_longest_cache_hit`, differs for a reason of its own.
    """

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[Any],
        max_length: int,
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        group_id: int,
    ) -> tuple[list[KVCacheBlock], int]:
        """The newest cached state, and nulls for everything before it. R6.4, R6.7.

        A recurrent state is not a prefix -- position N's state depends on every token
        before it, sequentially, so there is nothing to concatenate. The tempting
        answer is therefore "no hit": return `([], 0)`.

        That answer is wrong, and wrong in the direction that silently destroys a
        number. `KVCacheCoordinator.find_longest_cache_hit` reconciles a hit across
        *every* group, so a group reporting zero drives the whole reconciled length to
        zero and throws away the attention groups' hits -- a hybrid model's prefix
        cache hit rate would read exactly 0.0 for every request, and C3 calls that rate
        exact.

        What a state-space group can serve is a *snapshot*: if the state at some block
        boundary was cached, the request resumes from there. So this scans right to
        left for the newest cached block and returns it with null placeholders before
        it, which is upstream's answer and which reconciles cleanly with an attention
        group's prefix.
        """
        block_size = kv_cache_spec.block_size
        null_block = block_pool.null_block
        assert null_block is not None, (
            "a state-space group needs the reserved null block to stand in for the "
            "states it did not snapshot (R6.7)"
        )
        for index in range(max_length // block_size - 1, -1, -1):
            cached = block_pool.get_cached_block(block_hashes[index], group_id=group_id)
            if cached is not None:
                return [null_block] * index + [cached], (index + 1) * block_size
        return [], 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """All but the last computed token. R6.7.

        Upstream says it in one line -- "Mamba only need to keep the state of the
        last computed token" -- and the consequence is the whole point of the layer:
        the state at position N already summarises every token before it, so the
        snapshots taken at earlier boundaries are dead the moment a newer one exists.

        Inheriting full attention's zero here was the defect this replaces. It left
        a request holding `ceil(tokens / block_size)` state pages instead of one, so
        a 4 MiB-per-page state-space group was charged 8x its true cost at 8k context
        and grew from there -- and the model card, the README and the commit message
        all said the opposite. Blocks below the newest are still *freed*, not
        forgotten: they keep their hash and stay in the free queue, so a later
        request can still resume from a snapshot they hold.
        """
        return num_computed_tokens - 1

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """Zero. A recurrent state is per request; there is no shared prefix to
        run cascade attention over, and reporting one would invite an optimization
        that cannot apply."""
        return 0


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    block_pool: BlockPool,
    kv_cache_group_id: int,
) -> SingleTypeKVCacheManager:
    """Pick the manager for a group's spec."""
    from pvllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    # Order matters if the specs ever share a base; checked most specific first.
    if isinstance(kv_cache_spec, MambaSpec):
        return MambaManager(kv_cache_spec, block_pool, kv_cache_group_id)
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return SlidingWindowManager(kv_cache_spec, block_pool, kv_cache_group_id)
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return FullAttentionManager(kv_cache_spec, block_pool, kv_cache_group_id)
    raise NotImplementedError(
        f"no KV cache manager for {type(kv_cache_spec).__name__}; state-space "
        f"(Mamba) groups are not implemented"
    )
