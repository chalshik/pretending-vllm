"""KV cache shape and layout.

Upstream: vllm/v1/kv_cache_interface.py
Tier: A

R6.7: the group abstraction exists from the start even though only one group is ever
built until hybrid models land in M4. Retrofitting groups later would mean touching
every block-id call site in the scheduler and the runner -- the tuple-of-lists shape in
`SchedulerOutput.block_ids` only makes sense if the manager was built around groups
from the beginning.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KVCacheSpec:
    """How one kind of attention layer consumes KV cache."""

    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: str
    #: Bytes per element of `dtype`.
    dtype_bytes: int

    @property
    def page_size_bytes(self) -> int:
        """Bytes one block occupies, for one layer.

        `2 *` is key plus value.
        """
        return (
            2 * self.block_size * self.num_kv_heads * self.head_size * self.dtype_bytes
        )

    @property
    def type_id(self) -> str:
        """Layers sharing a type id can share a KV cache group."""
        return f"full_attention_{self.block_size}_{self.page_size_bytes}"


@dataclass(frozen=True)
class FullAttentionSpec(KVCacheSpec):
    """A standard causal attention layer. The only spec until M4."""


@dataclass
class KVCacheGroupSpec:
    """Layers that share a block table.

    One group per distinct `KVCacheSpec`. A model with sliding-window or state-space
    layers has more than one, which is why the scheduler passes block ids as a tuple
    indexed by group.
    """

    layer_names: list[str]
    kv_cache_spec: KVCacheSpec


@dataclass
class KVCacheTensor:
    """A slab of device memory backing one or more layers' KV cache."""

    size: int
    shared_by: list[str]


@dataclass
class KVCacheConfig:
    """The resolved KV cache layout for this engine."""

    num_blocks: int
    kv_cache_tensors: list[KVCacheTensor] = field(default_factory=list)
    kv_cache_groups: list[KVCacheGroupSpec] = field(default_factory=list)

    @property
    def num_groups(self) -> int:
        return len(self.kv_cache_groups)

    @property
    def block_size(self) -> int:
        if not self.kv_cache_groups:
            raise ValueError("KVCacheConfig has no groups; block size is undefined")
        return self.kv_cache_groups[0].kv_cache_spec.block_size

    @property
    def page_size_bytes(self) -> int:
        """Bytes one block costs across every layer of every group."""
        return sum(
            len(group.layer_names) * group.kv_cache_spec.page_size_bytes
            for group in self.kv_cache_groups
        )
