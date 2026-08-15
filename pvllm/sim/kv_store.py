"""A simulated external KV store. R17.2.

Upstream: (none -- simulator)
Tier: D

The store a `SimSharedStoreConnector` talks to. It holds block *hashes*, never bytes,
and answers two questions: is this prefix here, and how long would moving it take.

That is the whole content of KV disaggregation from the engine's point of view. A
prefill node writes KV for a prompt; a decode node discovers it already exists and
pulls it instead of recomputing. Whether the pull is worth it is a question about
*bandwidth against recompute cost*, and both sides of that comparison are numbers this
can supply without a byte moving.

**The numbers are modeled (R9.5).** `bandwidth_bytes_per_second` and
`latency_seconds` are what you tell it. Set them from a measurement of your real store
-- an NVMe-backed LMCache and an S3 bucket differ by four orders of magnitude, and
which one you have decides whether disaggregation helps or hurts.

Shared between instances through a module-level registry, so two `pvllm` engines in
one process can hand KV to each other -- which is what makes R17.2's end-to-end
prefill/decode exercise possible without a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Named stores, so two engines constructed independently can share one. Keyed by
#: name rather than passed by reference because the two engines are configured
#: separately -- that is the whole point of disaggregation.
_STORES: dict[str, SimKVStore] = {}


@dataclass
class SimKVStore:
    """An external KV store, as a set of resident block hashes and two rates."""

    name: str = "default"
    #: How fast KV moves in or out. The number that decides whether pulling a
    #: prefix beats recomputing it.
    bandwidth_bytes_per_second: float = 10e9
    #: Fixed cost per transfer, whatever its size. A round trip to a remote store
    #: dominates for short prefixes, which is why a store can be *slower* than
    #: recompute for small prompts and much faster for long ones.
    latency_seconds: float = 0.001
    #: Capacity in blocks. `None` means unbounded, which is the useful default for
    #: exercising the mechanism; set it to see eviction.
    capacity_blocks: int | None = None

    #: block hash -> nothing. A set with insertion order, for eviction.
    resident: dict[bytes, None] = field(default_factory=dict)
    num_lookups: int = 0
    num_hits: int = 0
    bytes_read: int = 0
    bytes_written: int = 0

    def __post_init__(self) -> None:
        if self.bandwidth_bytes_per_second <= 0:
            raise ValueError(
                f"bandwidth_bytes_per_second must be positive, got "
                f"{self.bandwidth_bytes_per_second}"
            )
        if self.latency_seconds < 0:
            raise ValueError(
                f"latency_seconds must be non-negative, got {self.latency_seconds}"
            )

    # --- lookup --------------------------------------------------------------

    def longest_prefix(self, block_hashes: list[bytes]) -> int:
        """How many leading blocks are resident.

        A *prefix*, stopping at the first miss, for the same reason the local prefix
        cache does: KV for a gap does not exist, and a hit beyond one cannot be read.
        """
        matched = 0
        for block_hash in block_hashes:
            if block_hash not in self.resident:
                break
            matched += 1

        self.num_lookups += len(block_hashes)
        self.num_hits += matched
        return matched

    # --- transfer ------------------------------------------------------------

    def transfer_seconds(self, num_bytes: int) -> float:
        """What moving `num_bytes` costs. Modeled, not measured (R9.5)."""
        if num_bytes <= 0:
            return 0.0
        return self.latency_seconds + num_bytes / self.bandwidth_bytes_per_second

    def read(self, num_bytes: int) -> float:
        self.bytes_read += num_bytes
        return self.transfer_seconds(num_bytes)

    def write(self, block_hashes: list[bytes], num_bytes: int) -> float:
        """Store blocks, evicting oldest-first if the capacity bites."""
        for block_hash in block_hashes:
            # Re-inserted rather than skipped, so a re-written block becomes the
            # most recent -- which is what makes the eviction order an LRU.
            self.resident.pop(block_hash, None)
            self.resident[block_hash] = None

        if self.capacity_blocks is not None:
            while len(self.resident) > self.capacity_blocks:
                self.resident.pop(next(iter(self.resident)))

        self.bytes_written += num_bytes
        return self.transfer_seconds(num_bytes)

    @property
    def hit_rate(self) -> float:
        return self.num_hits / self.num_lookups if self.num_lookups else 0.0

    def clear(self) -> None:
        self.resident.clear()
        self.num_lookups = self.num_hits = 0
        self.bytes_read = self.bytes_written = 0


def get_store(name: str = "default", **kwargs: float | int | None) -> SimKVStore:
    """The named store, creating it on first use.

    A registry rather than an argument because the two engines in a disaggregated
    pair are configured separately -- neither can be handed a reference to the
    other's store, which is exactly the situation the real deployment is in.
    """
    store = _STORES.get(name)
    if store is None:
        store = SimKVStore(name=name, **kwargs)  # type: ignore[arg-type]
        _STORES[name] = store
    return store


def reset_stores() -> None:
    """Drop every store. For tests, which must not leak state between cases."""
    _STORES.clear()
