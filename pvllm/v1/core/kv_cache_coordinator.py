"""Coordinates the per-group managers.

Upstream: vllm/v1/core/kv_cache_coordinator.py
Tier: A

R6.7. A request needs blocks in *every* group or none: allocating in group 0 and then
failing in group 1 would leave the request half-resident, so the coordinator asks every
group what it needs, checks the total against the pool once, and only then allocates.
"""

from __future__ import annotations

from pvllm.v1.core.block_pool import BlockPool
from pvllm.v1.core.kv_cache_utils import KVCacheBlock
from pvllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from pvllm.v1.kv_cache_interface import KVCacheConfig


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

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """Every group's blocks for a request, without returning them to the pool."""
        blocks: list[KVCacheBlock] = []
        for manager in self.single_type_managers:
            blocks.extend(manager.pop_blocks_for_free(request_id))
        return blocks

    def free(self, request_id: str) -> None:
        """Release a request's blocks in every group, tail first (R6.6)."""
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(
        self, request_id: str, num_running: int
    ) -> list[int]:
        """Per group, blocks shared by every running request (R5.9)."""
        return [
            manager.get_num_common_prefix_blocks(request_id, num_running)
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
