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

import hashlib
import pickle
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NewType

from pvllm.logger import init_logger
from pvllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec

if TYPE_CHECKING:
    from pvllm.v1.request import Request

logger = init_logger(__name__)


#: The hash of a full block's contents, including every block before it.
#:
#: `NewType` over `bytes`, matching upstream: a digest rather than a structure, so
#: two blocks with identical content and identical history hash identically no matter
#: which request produced them -- which is the entire mechanism of prefix caching.
BlockHash = NewType("BlockHash", bytes)

#: A block hash scoped to one KV cache group (R6.7).
#:
#: Two groups with different block sizes can produce the same digest for different
#: content, so the group id is appended to the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def sha256_hash(value: Any) -> bytes:
    """SHA-256 of a pickled object. Upstream's default at the pin."""
    return hashlib.sha256(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).digest()


def builtin_hash(value: Any) -> bytes:
    """Python's builtin `hash`, as 8 bytes.

    Faster and far more collision-prone than SHA-256. Upstream offers it as an
    option; it is only safe because a hit is verified by the block still being
    resident, not by the digest alone.

    `hash()` is salted per process by `PYTHONHASHSEED`, so this is *not* reproducible
    across runs unless that variable is set. `sha256` is the default here for that
    reason.
    """
    return hash(value).to_bytes(8, "big", signed=True)


_HASH_FUNCTIONS: dict[str, Callable[[Any], bytes]] = {
    "sha256": sha256_hash,
    "builtin": builtin_hash,
}


def get_hash_fn_by_name(name: str) -> Callable[[Any], bytes]:
    try:
        return _HASH_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown prefix_caching_hash_algo {name!r}; expected one of "
            f"{sorted(_HASH_FUNCTIONS)}"
        ) from None


def compute_none_hash(hash_fn: Callable[[Any], bytes], seed: int) -> BlockHash:
    """The sentinel standing in for "no parent block".

    **A deliberate divergence from upstream, and the one caveat on C3.** Upstream
    uses `os.urandom(32)` unless `PYTHONHASHSEED` is set, so that cache keys are not
    predictable across processes. That would make block hashes differ on every run
    here, which breaks B4 (identical output from the same seed) and makes a recorded
    conformance trace incomparable to the next one.

    So this derives the sentinel from the run seed instead: deterministic by
    construction, and still distinct per seed.

    The consequence for C3 is worth stating precisely. Hit *rates* and block
    allocation order are reproducible either way and can be compared to a real vLLM
    run directly. Hash *values* can only be compared if that run had `PYTHONHASHSEED`
    set and the same value fed this derivation -- upstream's own warning says as
    much. Until golden traces exist this is theoretical, but it is a real limit on
    what C3 can check, not a detail.
    """
    return BlockHash(hash_fn(("pvllm-none-hash", seed)))


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


def free_block_ids(blocks: Iterable[KVCacheBlock]) -> list[int]:
    """Block ids, for tracing and assertions."""
    return [block.block_id for block in blocks]


# --- hashing (R6.3, C3) ----------------------------------------------------


def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int = 0, end_token_idx: int | None = None
) -> tuple[Any, ...] | None:
    """Everything beyond token ids that must distinguish one block from another.

    Two requests with identical tokens must *not* share blocks when anything else
    about them differs: a different LoRA adapter produces different KV for the same
    tokens, and `cache_salt` exists precisely so a caller can partition the cache
    between tenants. Omitting a key here is a cache-poisoning bug -- one request
    silently reading another's KV -- which is why the unimplemented cases raise
    rather than being skipped.
    """
    # Upstream's order, exactly: LoRA, then multimodal, then the salt. The tuple is
    # hashed, so the order is part of the value, and C3 makes hash values themselves
    # part of the contract. An order of our own choosing would have been a second
    # divergence on top of `compute_none_hash`'s -- and unlike that one, an
    # unnecessary and undocumented one.
    keys: list[Any] = []
    if request.lora_request is not None:
        # R16.1. The adapter's *name*, as upstream keys it. The id would be the more
        # natural identity -- two names for one adapter would then share its cached
        # prefixes -- but upstream treats them as distinct, and a simulator that
        # improves on the engine it stands in for is telling its user something the
        # real engine will not do.
        keys.append(request.lora_request.lora_name)
    # R18. Only the images this *block* overlaps, which is what upstream does and
    # matters more than it looks. Folding every one of a request's images into every
    # block's key would partition the text *before* the first image too -- so two
    # prompts sharing a long system prompt and differing only in a later image would
    # share nothing, and the reported hit rate would be far below a real
    # deployment's. C3 calls hit rate exact, so over-partitioning is as wrong as
    # under-partitioning; it is just wrong in the safe direction.
    if request.mm_features:
        end = end_token_idx if end_token_idx is not None else request.num_tokens
        # The identifier *and* the item's offset within this block, which is what
        # upstream appends. The offset is what keeps two different tilings of the
        # same images apart: blocks of pure placeholder tokens are byte-identical
        # whatever produced them, so without it a block covering image A's tail and
        # image B's head hashes the same as one covering a different split of the
        # same pair -- and the second request reads KV computed for a different
        # layout. C3 makes hash values themselves part of the contract, so this
        # matches upstream exactly rather than approximately.
        keys.extend(
            (feature.identifier, feature.position - start_token_idx)
            for feature in request.mm_features
            if feature.position < end
            and feature.position + feature.length > start_token_idx
        )
    # Only on the first block, as upstream does. Every later block chains through
    # block 0's hash, which already carries the salt, so repeating it partitions
    # nothing further -- it only made every block's hash value differ from
    # upstream's.
    if start_token_idx == 0 and request.cache_salt:
        keys.append(request.cache_salt)
    return tuple(keys) if keys else None


def hash_block_tokens(
    hash_fn: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    none_hash: BlockHash,
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Hash one full block, chained to every block before it. R6.3, C3.

    The chain through `parent_block_hash` is what makes this a *prefix* cache rather
    than a block cache: block 3 of one sequence matches block 3 of another only if
    blocks 0 through 2 matched too. Hash a block's tokens alone and two requests
    sharing a middle passage but not a beginning would wrongly collide -- and the
    second would read KV computed under a different preceding context.
    """
    if not parent_block_hash:
        parent_block_hash = none_hash
    return BlockHash(
        hash_fn((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )


def get_request_block_hasher(
    block_size: int,
    hash_fn: Callable[[Any], bytes],
    none_hash: BlockHash,
) -> Callable[[Request], list[BlockHash]]:
    """Build the per-request hasher injected into `Request` (F8).

    Returns hashes for blocks that became *full* since the last call. A partial tail
    is never hashed: a later token would change its contents, so caching it would
    publish a block whose identity is about to change underneath any request that
    matched it.
    """

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start = len(request.block_hashes) * block_size
        num_tokens = request.num_tokens
        parent_hash = request.block_hashes[-1] if request.block_hashes else None

        new_hashes: list[BlockHash] = []
        while start + block_size <= num_tokens:
            block_tokens = request.all_token_ids[start : start + block_size]
            # Per block, because the multimodal keys depend on which images this
            # block's token span touches (R18).
            extra_keys = generate_block_hash_extra_keys(
                request, start, start + block_size
            )
            block_hash = hash_block_tokens(
                hash_fn, parent_hash, block_tokens, none_hash, extra_keys
            )
            new_hashes.append(block_hash)
            parent_hash = block_hash
            start += block_size
        return new_hashes

    return request_block_hasher


def get_kv_cache_groups(
    specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Split a model's layers into groups that can share a block table. R6.7.

    Upstream's algorithm (`_get_kv_cache_groups_uniform_page_size`), and its shape is
    not obvious, so it is worth stating. A model with 25 windowed layers and 5 full
    ones is not two groups of 25 and 5. It is *six* groups of five: the layers repeat
    with a pattern, and the block pool can only be divided evenly if every group
    occupies the same bytes per block. Groups of unequal size would need pages of
    unequal size, and a pool of mixed-size pages fragments.

    So the group size is the smallest bucket, and each larger bucket is split into
    `ceil(len / group_size)` groups. Padding is added if a bucket does not divide
    evenly, and upstream's `layers[i::num_groups]` striping is used rather than
    contiguous slicing -- under pipeline parallelism a contiguous split puts whole
    groups on one stage and leaves empty ones on another, which then get padded to
    the same size and waste the memory the grouping exists to save.
    """
    if not specs:
        raise ValueError("the model reported no attention layers")

    buckets: dict[str, list[str]] = {}
    for layer_name, spec in specs.items():
        buckets.setdefault(spec.type_id, []).append(layer_name)

    if len(buckets) == 1:
        layer_names = next(iter(buckets.values()))
        return [
            KVCacheGroupSpec(
                layer_names=layer_names, kv_cache_spec=specs[layer_names[0]]
            )
        ]

    sizes = [len(layers) for layers in buckets.values()]
    group_size = min(sizes)
    if max(sizes) < min(sizes) * 1.5:
        # Upstream's heuristic, and the reason is worth keeping: padding one bucket
        # up to the larger size wastes less than splitting the larger one into two
        # groups that are each mostly padding.
        group_size = max(sizes)

    grouped: list[list[str]] = []
    for layers in buckets.values():
        num_groups = -(-len(layers) // group_size)
        padding = num_groups * group_size - len(layers)
        if padding:
            # Not literally added: the short group holds fewer layers, and the pool
            # is sized from the *longest* group, so those slots are paid for and
            # unused. Reported as waste rather than as an addition -- the earlier
            # wording named an operation that never happened, and the code then
            # refused the very pattern it had just claimed to pad.
            logger.warning(
                "Hybrid KV cache: %d layer slot(s) in a group of %d go unused, "
                "wasting at most %.1f%% of the KV pool [modeled]",
                padding,
                group_size,
                100.0 * padding / (num_groups * group_size),
            )
        for index in range(num_groups):
            grouped.append(layers[index::num_groups])

    groups = [
        KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=specs[layer_names[0]])
        for layer_names in grouped
    ]

    # The invariant the whole scheme rests on, stated the way upstream states it:
    # every *layer* must occupy the same bytes per block. Group lengths need not
    # match. A bucket that does not divide evenly leaves one short group, and
    # upstream allocates `max(len(layer_names))` tensors and simply lets the short
    # group index fewer of them -- which is what "padding" means there. Requiring
    # equal group *lengths* refused patterns upstream accepts: `hybrid-4b` at
    # `pipeline_parallel_size=2` is 2 full and 13 windowed layers per stage, so the
    # windowed bucket stripes into six groups of 2 and one of 1, and the engine died
    # at startup immediately after logging that it had added padding it never added.
    page_sizes = {group.kv_cache_spec.page_size_bytes for group in groups}
    if len(page_sizes) != 1:
        raise NotImplementedError(
            f"this model's attention layers would need pages of {sorted(page_sizes)} "
            f"bytes each. Every layer must occupy the same bytes per block, or the "
            f"pool fragments; upstream assumes the same."
        )
    return groups
