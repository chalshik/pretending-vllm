"""Coordinates the per-group managers.

Upstream: vllm/v1/core/kv_cache_coordinator.py
Tier: A

R6.7. A request needs blocks in *every* group or none: allocating in group 0 and then
failing in group 1 would leave the request half-resident, so the coordinator asks every
group what it needs, checks the total against the pool once, and only then allocates.
"""

from __future__ import annotations

from typing import Any

from pvllm.v1.core.block_pool import BlockPool
from pvllm.v1.core.kv_cache_utils import KVCacheBlock
from pvllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from pvllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
)


class KVCacheCoordinator:
    """Fans allocation and freeing across KV cache groups."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        block_pool: BlockPool,
        enable_caching: bool = False,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.block_pool = block_pool
        self.enable_caching = enable_caching

        self.single_type_managers: tuple[SingleTypeKVCacheManager, ...] = tuple(
            get_manager_for_kv_cache_spec(group.kv_cache_spec, block_pool, group_id)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        )

    @property
    def num_groups(self) -> int:
        return len(self.single_type_managers)

    def find_longest_cache_hit(
        self, block_hashes: list[Any], max_length: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """The longest prefix *every* group can serve. R6.4, R6.7, C3.

        Upstream's fixed-point: each attention type either accepts the candidate
        length or reduces it, and any reduction restarts the pass. It converges
        because the length only ever decreases.

        The reason it has to iterate rather than take a minimum: the two types answer
        different questions. A full-attention group needs the prefix from token zero,
        so its hit is downward-closed -- shortening the candidate only trims it. A
        windowed group needs a contiguous run covering its window *ending at the
        candidate*, so moving the candidate can invalidate the run it just found and
        force a different one. Asking each type once and taking the smallest answer
        would report a hit the windowed group cannot actually serve.

        Returns `(blocks per group, hit tokens)`.
        """
        num_groups = len(self.single_type_managers)
        hit_length = max_length
        blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        length_by_group: list[int] = [0] * num_groups

        while True:
            current = hit_length
            for group_id, manager in enumerate(self.single_type_managers):
                spec = self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
                # A *positive* test. Written as "not sliding window" it would sweep
                # a state-space group into the full-attention shortcut, looking it up
                # once and then only min-ing -- leaving a block list from a longer
                # candidate attached to a shorter reconciled length. Only full
                # attention is downward-closed. `MLAAttentionSpec` subclasses
                # `FullAttentionSpec` and belongs here; `SlidingWindowSpec` and
                # `MambaSpec` do not.
                if isinstance(spec, FullAttentionSpec) and (
                    blocks_by_group[group_id] is not None
                ):
                    # Full attention is downward-closed: look it up once, then trim.
                    current = min(current, length_by_group[group_id])
                    continue
                blocks, found = type(manager).find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=current,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    group_id=group_id,
                )
                current = found
                blocks_by_group[group_id] = blocks
                length_by_group[group_id] = found
            if current >= hit_length:
                break
            hit_length = current

        # Trim the full-attention groups to the reconciled length; a windowed group's
        # list already ends where its run does.
        num_blocks = -(-hit_length // self.kv_cache_config.block_size)
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            group_blocks = blocks_by_group[group_id]
            if group_blocks is not None and isinstance(
                group.kv_cache_spec, FullAttentionSpec
            ):
                del group_blocks[num_blocks:]
        return tuple(blocks or [] for blocks in blocks_by_group), hit_length

    def get_num_blocks_to_allocate(self, request_id: str, num_tokens: int) -> int:
        """Total new blocks across every group.

        Summed before any allocation happens so the caller can fail early (R6.5)
        rather than discovering a shortfall partway through.
        """
        return sum(
            manager.get_num_blocks_to_allocate(request_id, num_tokens)
            for manager in self.single_type_managers
        )

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> tuple[list[KVCacheBlock], ...]:
        """Allocate in every group. Returns the new blocks per group."""
        return tuple(
            manager.allocate_new_blocks(request_id, num_tokens)
            for manager in self.single_type_managers
        )

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        return tuple(
            manager.get_blocks(request_id) for manager in self.single_type_managers
        )

    def adopt_cached_blocks(
        self, request_id: str, blocks: tuple[list[KVCacheBlock], ...]
    ) -> None:
        """Install prefix-cache hits in every group. R6.5."""
        for manager, group_blocks in zip(
            self.single_type_managers, blocks, strict=True
        ):
            manager.adopt_cached_blocks(request_id, group_blocks)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """Every group's blocks for a request, without returning them to the pool."""
        blocks: list[KVCacheBlock] = []
        for manager in self.single_type_managers:
            blocks.extend(manager.pop_blocks_for_free(request_id))
        return blocks

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        """Fan the window eviction out to every group that has one. R6.7."""
        for manager in self.single_type_managers:
            remove = getattr(manager, "remove_skipped_blocks", None)
            if remove is not None:
                remove(request_id, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """Release a request's blocks in every group, tail first (R6.6)."""
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Per group, blocks shared by every request holding KV cache (R5.9)."""
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    block_pool: BlockPool,
    enable_caching: bool = False,
) -> KVCacheCoordinator:
    """Build the coordinator for a resolved KV cache layout."""
    if not kv_cache_config.kv_cache_groups:
        raise ValueError(
            "KVCacheConfig has no groups; the memory model must resolve at least one "
            "before the coordinator can be built"
        )
    return KVCacheCoordinator(kv_cache_config, block_pool, enable_caching)
