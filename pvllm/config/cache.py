"""KV cache configuration.

Upstream: vllm/config/cache.py
Tier: C
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Upstream's default at the pin. Note it is sha256, not the builtin hash (R6.3).
PrefixCachingHashAlgo = str

DEFAULT_BLOCK_SIZE = 16


@dataclass
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: int = DEFAULT_BLOCK_SIZE
    gpu_memory_utilization: float = 0.92
    cache_dtype: str = "auto"
    num_gpu_blocks_override: int | None = None
    sliding_window: int | None = None
    # R1.4: both default to enabled upstream.
    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "sha256"
    #: Hard cap on KV bytes, bypassing the utilization-derived budget.
    kv_cache_memory_bytes: int | None = None

    # Derived during `determine_available_memory` (R10.2/R10.3), not user-set.
    num_gpu_blocks: int | None = field(default=None, init=False)
    num_cpu_blocks: int | None = field(default=None, init=False)
    kv_cache_size_tokens: int | None = field(default=None, init=False)
    kv_cache_max_concurrency: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got "
                f"{self.gpu_memory_utilization}"
            )
        if self.cache_dtype not in ("auto", "fp8", "float8_e4m3", "float8_e5m2"):
            raise ValueError(
                f"unsupported cache_dtype {self.cache_dtype!r}; expected 'auto' or an "
                f"fp8 variant"
            )
        if self.prefix_caching_hash_algo not in ("sha256", "builtin"):
            raise ValueError(
                f"unsupported prefix_caching_hash_algo "
                f"{self.prefix_caching_hash_algo!r}; expected 'sha256' or 'builtin'"
            )
        if self.sliding_window is not None:
            raise NotImplementedError(
                "sliding-window attention needs multiple KV cache groups "
                "(requirement R6.7), which lands in M4"
            )

    @property
    def resolved_cache_dtype(self) -> str | None:
        """`None` means "same as the model dtype", matching upstream's 'auto'."""
        if self.cache_dtype == "auto":
            return None
        if self.cache_dtype == "fp8":
            return "float8_e4m3"
        return self.cache_dtype
