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
from pvllm.v1.core.kv_cache_utils import KVCacheBlock
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
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        self.log_stats = log_stats

        self.block_size = kv_cache_config.block_size
        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config, self.block_pool, enable_caching
        )
        self.num_kv_cache_groups = self.coordinator.num_groups

        #: Reused for requests that got nothing, so the common path allocates no
        #: tuple. Upstream does the same.
        self._empty_blocks = KVCacheBlocks(
            tuple([] for _ in range(self.num_kv_cache_groups))
        )

        # R6.9. Real counters even in M1, where the hit rate is always zero -- a
        # metric that only appears once a feature lands is a metric nobody wired up.
        self.prefix_cache_queries = 0
        self.prefix_cache_hits = 0

    @property
    def usage(self) -> float:
        """Fraction of the block pool in use. Feeds `vllm:kv_cache_usage_perc`."""
        return self.block_pool.get_usage()

    # --- lookup --------------------------------------------------------------

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """The longest cached prefix for a request, and its token count.

        Always empty until prefix caching lands in M2. The call site exists now so
        the scheduler's admission path is the real one (R6.4).
        """
        if not self.enable_caching:
            return self._empty_blocks, 0
        raise NotImplementedError("prefix cache lookup (requirement R6.4) lands in M2")

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

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request.request_id, num_tokens_needing_slots
        )

        # Fail early, before anything is mutated (R6.5).
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_needing_slots
        )

        if self.enable_caching and not delay_cache_blocks:
            raise NotImplementedError(
                "caching full blocks on allocation (requirement R6.5) lands in M2"
            )

        return KVCacheBlocks(new_blocks)

    # --- release -------------------------------------------------------------

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

    def get_num_common_prefix_blocks(
        self, request_id: str, num_running_requests: int
    ) -> list[int]:
        """Per group, blocks shared by every running request (R5.9)."""
        return self.coordinator.get_num_common_prefix_blocks(
            request_id, num_running_requests
        )

    def reset_prefix_cache(self) -> bool:
        """R6.10. Also resets the hit-rate counters, matching upstream."""
        if not self.block_pool.reset_prefix_cache():
            return False
        self.prefix_cache_queries = 0
        self.prefix_cache_hits = 0
        return True

    def make_prefix_cache_stats(self) -> tuple[int, int]:
        """`(queries, hits)` since the last reset. R6.9."""
        return self.prefix_cache_queries, self.prefix_cache_hits

    def __repr__(self) -> str:
        return (
            f"KVCacheManager(num_blocks={self.kv_cache_config.num_blocks}, "
            f"block_size={self.block_size}, groups={self.num_kv_cache_groups}, "
            f"usage={self.usage:.3f})"
        )
