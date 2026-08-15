"""The KV cache manager: the scheduler's view of block allocation.

Upstream: vllm/v1/core/kv_cache_manager.py
Tier: A

C2 binds this exactly. `allocate_slots` follows upstream's order (R6.5), and the order
is the point:

1. Work out how many new blocks are needed, across every group.
2. **Fail early** if the pool cannot cover it -- return `None` before touching
   anything. A partial allocation would leave the request holding blocks it cannot
   use while starving the request that could have used them.
3. Touch cached blocks (M2), then pop new ones from the free queue head.

Returning `None` rather than raising is what lets the scheduler treat "does not fit"
as a scheduling outcome -- it preempts and retries (R5.5) instead of failing the
request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pvllm.logger import init_logger
from pvllm.v1.core.block_pool import BlockPool
from pvllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from pvllm.v1.core.kv_cache_metrics import PrefixCacheStats
from pvllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    compute_none_hash,
    get_hash_fn_by_name,
    get_request_block_hasher,
)
from pvllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from pvllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """An allocation result.

    The interface between the scheduler and the KV manager: the scheduler sees block
    *ids*, never `KVCacheBlock` objects, so the pool's internals stay out of the
    scheduling logic.

    `blocks[i][j]` is the j-th block of the i-th KV cache group. Groups are the outer
    dimension because different groups may hold different numbers of blocks once
    hybrid models land (R6.7).
    """

    blocks: tuple[list[KVCacheBlock], ...]

    def __add__(self, other: KVCacheBlocks) -> KVCacheBlocks:
        return KVCacheBlocks(
            tuple(
                own + theirs
                for own, theirs in zip(self.blocks, other.blocks, strict=True)
            )
        )

    def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...] | None:
        """Block ids per group -- what crosses into `SchedulerOutput`."""
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([block.block_id for block in group] for group in self.blocks)

    def new_empty(self) -> KVCacheBlocks:
        return KVCacheBlocks(tuple([] for _ in self.blocks))

    @property
    def num_blocks(self) -> int:
        return sum(len(group) for group in self.blocks)


class KVCacheManager:
    """Owns block allocation for every request the scheduler is tracking."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = False,
        enable_kv_cache_events: bool = False,
        log_stats: bool = False,
        hash_algo: str = "sha256",
        seed: int = 0,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.log_stats = log_stats

        self.block_size = kv_cache_config.block_size
        from pvllm.v1.kv_cache_interface import SlidingWindowSpec

        # R6.7. A windowed group needs the null block to stand in for evicted slots,
        # and prefix caching is off for the whole pool when any group slides: a
        # cached block is only reusable if the prefix leading to it is still
        # attended to, and inside a window it usually is not.
        self.has_sliding_window = any(
            isinstance(group.kv_cache_spec, SlidingWindowSpec)
            for group in kv_cache_config.kv_cache_groups
        )
        if self.has_sliding_window and enable_caching:
            logger.info(
                "Prefix caching disabled: this model has sliding-window attention, "
                "and a cached block is only reusable while the prefix leading to it "
                "is still inside the window (R6.7)."
            )
            enable_caching = False

        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            reserve_null_block=self.has_sliding_window,
        )
        self.enable_caching = enable_caching
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config, self.block_pool, enable_caching
        )
        self.num_kv_cache_groups = self.coordinator.num_groups

        #: Reused for requests that got nothing, so the common path allocates no
        #: tuple. Upstream does the same.
        self._empty_blocks = KVCacheBlocks(
            tuple([] for _ in range(self.num_kv_cache_groups))
        )

        # R6.9.
        self.prefix_cache_stats = PrefixCacheStats()

        # F8: the hasher is handed to each Request at construction, so the manager
        # owns hashing policy (algorithm, salt, extra keys) and Request only stores
        # the result.
        self.hash_fn = get_hash_fn_by_name(hash_algo)
        self.none_hash = compute_none_hash(self.hash_fn, seed)
        self.block_hasher = (
            get_request_block_hasher(self.block_size, self.hash_fn, self.none_hash)
            if enable_caching
            else None
        )

    @property
    def usage(self) -> float:
        """Fraction of the block pool in use. Feeds `vllm:kv_cache_usage_perc`."""
        return self.block_pool.get_usage()

    # --- lookup --------------------------------------------------------------

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """The longest cached prefix for a request, and its token count.

        Walks the request's block hashes from the start and stops at the first miss:
        a prefix cache is only usable contiguously, since the KV for a gap does not
        exist and a hit beyond it cannot be read.

        **At least one token is always recomputed.** A request whose every block is
        cached would otherwise be scheduled with zero new tokens -- nothing to run,
        no logits, no sampled token, and it would never progress. The rule only bites
        on an exact full-prompt hit, which makes it the easiest thing here to get
        wrong and not notice.
        """
        if not self.enable_caching or request.num_computed_tokens > 0:
            return self._empty_blocks, 0

        self.prefix_cache_stats.queries += request.num_tokens

        computed: list[KVCacheBlock] = []
        for block_hash in request.block_hashes:
            block = self.block_pool.get_cached_block(block_hash, group_id=0)
            if block is None:
                break
            computed.append(block)

        num_computed_tokens = len(computed) * self.block_size
        if computed and num_computed_tokens == request.num_tokens:
            computed.pop()
            num_computed_tokens -= self.block_size

        self.prefix_cache_stats.hits += num_computed_tokens
        if not computed:
            return self._empty_blocks, 0
        return KVCacheBlocks((computed,)), num_computed_tokens

    # --- allocation ----------------------------------------------------------

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> KVCacheBlocks | None:
        """Reserve slots for `num_new_tokens` more tokens. R6.5.

        Returns the newly allocated blocks, or `None` if the pool cannot cover the
        request -- which the scheduler reads as "does not fit right now", not as an
        error.
        """
        if num_new_tokens == 0:
            raise ValueError(
                "allocate_slots called with num_new_tokens=0; a step that schedules a "
                "request must give it at least one token"
            )

        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        # Never reserve past the model's length cap: a request at max_model_len needs
        # no further slots, and rounding up past it would allocate a block that can
        # never be written.
        num_tokens_needing_slots = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )

        # Blocks hit in the cache are adopted before counting what is still needed,
        # so a request with a long shared prefix asks the pool for almost nothing.
        cached_blocks = (
            new_computed_blocks.blocks[0]
            if new_computed_blocks is not None and new_computed_blocks.num_blocks
            else []
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request.request_id, num_tokens_needing_slots
        ) - len(cached_blocks)

        # A cached block with no other holder is *in* the free queue, and `touch`
        # takes it out. So it draws on the same free pool the new blocks do, and
        # counting it as still available would let an allocation pass the check and
        # then run the queue dry partway through -- which is a crash rather than the
        # `None` the scheduler knows how to handle.
        num_free_cached_blocks = sum(1 for block in cached_blocks if block.ref_cnt == 0)

        # Fail early, before anything is mutated (R6.5).
        if (
            num_blocks_to_allocate + num_free_cached_blocks
            > self.block_pool.get_num_free_blocks()
        ):
            return None

        if cached_blocks:
            # Take a reference and pull them out of the free queue before anything
            # else can claim them (R6.5).
            self.block_pool.touch(cached_blocks)
            self.coordinator.adopt_cached_blocks(request.request_id, (cached_blocks,))

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_needing_slots
        )

        if self.enable_caching and not delay_cache_blocks:
            self._cache_full_blocks(request, num_computed_tokens + num_new_tokens)

        return KVCacheBlocks(new_blocks)

    def _cache_full_blocks(self, request: Request, num_tokens: int) -> None:
        """Register every block that is now complete. R6.5.

        Immediately, not at request end: a full block can be shared *now*, and
        waiting would miss every hit from a request running concurrently with this
        one -- which is the case a prefix cache mostly exists to serve.
        """
        blocks = self.coordinator.get_blocks(request.request_id)[0]
        num_full_blocks = min(num_tokens // self.block_size, len(request.block_hashes))
        num_already_cached = sum(1 for b in blocks if b.block_hash is not None)
        self.block_pool.cache_full_blocks(
            list(request.block_hashes),
            list(blocks),
            num_already_cached,
            num_full_blocks,
            group_id=0,
        )

    # --- release -------------------------------------------------------------

    def remove_skipped_blocks(self, request: Request) -> None:
        """Release blocks that have fallen out of a sliding window. R6.7.

        A no-op for a full-attention model, which is why it costs one flag check on
        the common path. For a windowed one it is the whole mechanism: without it a
        long conversation holds every block it ever touched, and the bounded-KV
        property that makes windows worth having would not exist.
        """
        if not self.has_sliding_window:
            return
        self.coordinator.remove_skipped_blocks(
            request.request_id, request.num_computed_tokens
        )

    def free(self, request: Request) -> None:
        """Release everything a request holds.

        Blocks go back tail-first (R6.6), so the head of the sequence -- the part a
        later request is most likely to share -- survives longest in the free queue.
        """
        self.coordinator.free(request.request_id)

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """Take a request's blocks without returning them to the pool.

        The caller must free them, in reverse. Used where freeing has to be deferred
        past the point the request leaves the scheduler.
        """
        return self.coordinator.pop_blocks_for_free(request.request_id)

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        return KVCacheBlocks(
            tuple(list(group) for group in self.coordinator.get_blocks(request_id))
        )

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Per group, the leading blocks every KV-holding request shares. R5.9.

        Pass any running request's id; the answer is a property of the pool, not of
        that request. A real backend uses it to run cascade attention over the shared
        prefix once instead of once per request.

        It is computed and carried into `SchedulerOutput` and the attention metadata
        because that is upstream's contract, but **the cost model does not read it**:
        cascade attention is not modeled, so a step over a large shared prefix costs
        the same here as one over none. Latencies on shared-prefix workloads are
        therefore pessimistic relative to a real backend that takes the optimization.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def reset_prefix_cache(self) -> bool:
        """R6.10. Also resets the hit-rate counters, matching upstream."""
        if not self.block_pool.reset_prefix_cache():
            return False
        self.prefix_cache_stats.reset()
        return True

    def make_prefix_cache_stats(self) -> PrefixCacheStats:
        """Cache effectiveness since the last reset. R6.9."""
        self.prefix_cache_stats.evictions = self.block_pool.num_evicted_blocks
        self.prefix_cache_stats.cached_blocks = len(
            self.block_pool.cached_block_hash_to_block
        )
        return self.prefix_cache_stats

    def __repr__(self) -> str:
        return (
            f"KVCacheManager(num_blocks={self.kv_cache_config.num_blocks}, "
            f"block_size={self.block_size}, groups={self.num_kv_cache_groups}, "
            f"usage={self.usage:.3f})"
        )
