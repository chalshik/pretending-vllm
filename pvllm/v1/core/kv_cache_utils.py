"""KV cache blocks, the free-block queue, and hashing.

Upstream: vllm/v1/core/kv_cache_utils.py
Tier: A

C2 and C3 bind this module exactly: block allocation and free order, and block hash
values, must match upstream for the same inputs.

`FreeKVCacheBlockQueue` is an intrusive doubly linked list rather than a `deque`, and
that is not an optimization detail -- it is what makes the eviction *order* reproducible.
A block hit by a second request must be removable from the middle of the free queue in
O(1) (`touch`), and a `deque` cannot do that. Change the data structure and the
allocation trace changes with it.

Hashing (R6.3) lands in M2 with prefix caching. The types and the seam exist now.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple


class BlockHash(NamedTuple):
    """The hash of a full block's contents, plus the tokens that produced it.

    A NamedTuple rather than a bare int so a hash can never be confused with a block
    id -- both are ints, both flow through the KV manager, and mixing them would
    produce a cache that silently returns the wrong blocks.

    Carrying the token ids alongside the digest lets the pool verify a match rather
    than trust it, which makes a hash collision a detectable event instead of silent
    data corruption. Upstream does the same.
    """

    hash_value: int
    token_ids: tuple[int, ...]


class BlockHashWithGroupId(NamedTuple):
    """A block hash scoped to one KV cache group.

    Two groups with different block sizes can produce the same digest for different
    content, so the group id is part of the cache key (R6.7).
    """

    block_hash: BlockHash
    group_id: int


@dataclass
class KVCacheBlock:
    """KV cache block metadata. R6.2."""

    #: 0 to num_gpu_blocks - 1.
    block_id: int
    #: How many requests currently hold this block. Zero means it is in the free
    #: queue and may be evicted.
    ref_cnt: int = 0
    #: Set only when the block is full and cached; cleared on eviction (R6.2).
    _block_hash: BlockHashWithGroupId | None = None

    # Manipulated *only* by FreeKVCacheBlockQueue. Touching these anywhere else
    # corrupts the list silently -- the block count stays right while the ordering
    # goes wrong, which shows up much later as an unreproducible eviction trace.
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None

    #: The null block is a shared placeholder that must never be cached or freed.
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        return self._block_hash

    def set_block_hash(self, block_hash: BlockHashWithGroupId) -> None:
        assert self._block_hash is None, (
            f"block {self.block_id} already has a hash; re-hashing a cached block "
            f"would orphan its entry in the cache map"
        )
        self._block_hash = block_hash

    def reset_hash(self) -> None:
        """Clear the hash when the block is evicted."""
        self._block_hash = None

    def __repr__(self) -> str:
        # Deliberately does not follow prev/next: a linked-list node whose repr walks
        # its neighbours recurses through the whole queue.
        return (
            f"KVCacheBlock(block_id={self.block_id}, ref_cnt={self.ref_cnt}, "
            f"hash={self._block_hash is not None})"
        )


class FreeKVCacheBlockQueue:
    """Free blocks as an intrusive doubly linked list. R6.1.

    Ordered by block id initially. Once blocks have been allocated and freed the
    order becomes the eviction order:

    1. Least recently freed at the front.
    2. Among blocks freed together, the tail of a request's block chain comes first --
       maintained by the caller freeing a request's blocks in reverse (R6.6).

    Sentinel head and tail nodes remove the boundary branches, so every real block is
    guaranteed to have both neighbours. They are never popped.

    No Python objects are allocated while manipulating the list; the links live on the
    blocks themselves. This matters because the queue is touched on every allocation
    and every free of every step.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """Pop the least recently freed block."""
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync with the "
                f"free list"
            )
            raise ValueError("No free blocks available")

        first_block = self.fake_free_list_head.next_free_block
        if first_block.next_free_block is None:
            raise RuntimeError(
                f"block {first_block.block_id} is in the free list without a "
                f"next link; the list has been corrupted"
            )

        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first `n` free blocks, walking the list once."""
        if n == 0:
            return []
        assert self.num_free_blocks >= n, (
            f"asked for {n} free blocks but only {self.num_free_blocks} are free"
        )
        self.num_free_blocks -= n

        curr_block = self.fake_free_list_head.next_free_block
        popped: list[KVCacheBlock] = []
        for _ in range(n):
            assert curr_block is not None
            popped.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return popped

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block from the middle of the queue, in O(1).

        This is what a `deque` cannot do, and it is exactly what `touch` needs when a
        second request hits a cached block that was sitting in the free list.
        """
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(
                f"remove() called on block {block.block_id}, which is not in the free "
                f"list; its ref_cnt accounting is wrong"
            )

        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back at the tail -- the most-recently-freed end."""
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError("free list tail sentinel lost its prev link")
        last_block = self.fake_free_list_tail.prev_free_block

        last_block.next_free_block = block
        block.prev_free_block = last_block
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Append several blocks at the tail, in order."""
        if not blocks:
            return
        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, "free list tail sentinel lost its prev link"

        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block
        self.num_free_blocks += len(blocks)

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put blocks at the *front*, so they are evicted first.

        Used for blocks that carry no hash: they can never produce a cache hit, so
        evicting them ahead of hashed blocks costs nothing and preserves more of the
        prefix cache.
        """
        if not blocks:
            return
        first_block = self.fake_free_list_head.next_free_block
        assert first_block is not None, "free list head sentinel lost its next link"

        prev_block: KVCacheBlock = self.fake_free_list_head
        for block in blocks:
            block.prev_free_block = prev_block
            prev_block.next_free_block = block
            prev_block = block

        prev_block.next_free_block = first_block
        first_block.prev_free_block = prev_block
        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Walk the queue front to back. For tests and invariant checks."""
        blocks: list[KVCacheBlock] = []
        curr_block = self.fake_free_list_head.next_free_block
        while curr_block is not None and curr_block is not self.fake_free_list_tail:
            blocks.append(curr_block)
            curr_block = curr_block.next_free_block
        return blocks


def get_request_block_hasher(
    block_size: int, hash_algo: str
) -> None:  # pragma: no cover - M2
    """Build the per-request block hasher injected into `Request` (F8, R6.3).

    Lands in M2 with prefix caching. Named here so the seam is visible: `Request`
    already accepts the callable this will return.
    """
    raise NotImplementedError(
        "block hashing (requirement R6.3) lands in M2 with prefix caching"
    )


def need_extra_keys(request: object) -> bool:  # pragma: no cover - M2
    """Whether a request contributes extra keys (LoRA id, mm hash, cache salt)."""
    raise NotImplementedError("prefix cache extra keys (requirement R6.3) land in M2")


def free_block_ids(blocks: Iterable[KVCacheBlock]) -> list[int]:
    """Block ids, for tracing and assertions."""
    return [block.block_id for block in blocks]
