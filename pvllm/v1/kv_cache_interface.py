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
    """How one kind of layer consumes the KV cache.

    Split from `AttentionSpec` because a state-space layer is not an attention layer:
    it has no KV heads and no head size, and its page does not scale with
    `block_size`. Upstream splits at the same seam for the same reason.
    """

    block_size: int

    @property
    def page_size_bytes(self) -> int:
        """Bytes one block occupies, for one layer."""
        raise NotImplementedError

    @property
    def type_id(self) -> str:
        """Layers sharing a type id can share a KV cache group."""
        return f"{type(self).__name__}_{self.block_size}_{self.page_size_bytes}"


@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    """A layer that caches keys and values per token."""

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


@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    """A state-space layer: a fixed recurrent state per request. R6.7.

    The shape that makes it different from every other spec here is that its page does
    **not** scale with `block_size`. An attention page is `block_size` tokens of KV; a
    Mamba page is a conv ring buffer plus an SSM temporal state, both sized from the
    model config alone and constant however long the context gets.

    One pool cannot hold pages of two sizes, so upstream reconciles them *before* any
    spec exists: it grows `cache_config.block_size` until an attention page is at least
    as large as the state, then pads the state page up to match exactly. For a
    Nemotron-H-class model that moves the block size from 16 tokens to about 1040 --
    which changes block counts, prefix-cache granularity and every block hash value,
    so it lands on C2 and C3 far harder than the state bytes land on the memory model.
    `page_size_padded` carries the result of that reconciliation.
    """

    #: Bytes of recurrent state one layer holds for one request, before padding.
    state_bytes: int
    #: Set by the page-size unification to the common page. Never smaller than
    #: `state_bytes`.
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self) -> int:
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.state_bytes
            return self.page_size_padded
        return self.state_bytes


@dataclass(frozen=True)
class FullAttentionSpec(AttentionSpec):
    """A standard causal attention layer: every token is attended to, forever."""


@dataclass(frozen=True)
class MLAAttentionSpec(FullAttentionSpec):
    """Multi-head latent attention: one compressed latent per token. R6.7.

    Upstream models it as full attention with a different page size, and the
    difference is a single factor of two: `AttentionSpec.real_page_size_bytes` is
    `2 * block_size * num_kv_heads * head_size * dtype`, and
    `MLAAttentionSpec.real_page_size_bytes` drops the `2` because there is one latent
    rather than a key and a value.

    `num_kv_heads` is 1 by construction -- upstream's `get_num_kv_heads` returns 1 for
    MLA before dividing by the tensor-parallel size, so the latent is replicated on
    every rank. `head_size` is `kv_lora_rank + qk_rope_head_dim`.
    """

    @property
    def page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.head_size * self.dtype_bytes


@dataclass(frozen=True)
class SlidingWindowSpec(AttentionSpec):
    """An attention layer that only attends to the last `sliding_window` tokens. R6.7.

    The consequential difference is not the attention pattern -- it is that KV per
    request stops growing with context. A model with a 128k context and a 4k window
    holds 4k tokens of KV per request however long the conversation gets, so capacity
    is bounded by concurrency rather than by conversation length. That is a different
    capacity planning problem, and it is the reason this is modeled rather than
    approximated.
    """

    sliding_window: int = 0

    def __post_init__(self) -> None:
        if self.sliding_window < 1:
            raise ValueError(
                f"sliding_window must be positive, got {self.sliding_window}"
            )

    @property
    def type_id(self) -> str:
        # The window is part of the identity: two layers with different windows
        # cannot share a block table, because they free blocks at different points.
        return f"{super().type_id}_sw{self.sliding_window}"


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
