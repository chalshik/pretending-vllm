"""Prefix cache effectiveness. R6.9.

Upstream: vllm/v1/core/kv_cache_metrics.py
Tier: B

F5: upstream split these out of the KV manager into their own module at the pin, and
added block-residency histograms (`vllm:kv_block_lifetime_seconds`,
`vllm:kv_block_reuse_gap_seconds`, `vllm:kv_block_idle_before_evict_seconds`) that the
draft spec's R12.1 list predates.

**Counted in tokens, not blocks.** A hit rate over blocks would read differently from
the same workload at a different block size, which makes it useless for comparing
configurations -- and comparing configurations cheaply is the point of the project
(G5). Upstream counts tokens for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrefixCacheStats:
    """Cache effectiveness since the last reset.

    Cumulative rather than per-interval: Prometheus computes rates from monotonic
    counters, and a per-interval value would give a different answer depending on
    scrape timing.
    """

    #: Prompt tokens looked up, whether or not they hit.
    queries: int = 0
    #: Prompt tokens served from cache.
    hits: int = 0
    #: Blocks evicted to make room for new allocations.
    evictions: int = 0
    #: Blocks currently holding cached content.
    cached_blocks: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of queried tokens served from cache.

        Zero rather than undefined when nothing has been queried: a dashboard
        dividing by zero at startup is a worse answer than an honest zero.
        """
        return self.hits / self.queries if self.queries else 0.0

    def reset(self) -> None:
        """R6.10. Called when the prefix cache is cleared."""
        self.queries = 0
        self.hits = 0
        self.evictions = 0
        self.cached_blocks = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "prefix_cache_queries": self.queries,
            "prefix_cache_hits": self.hits,
            "prefix_cache_hit_rate": self.hit_rate,
            "prefix_cache_evictions": self.evictions,
            "prefix_cache_cached_blocks": self.cached_blocks,
        }
