"""Fake weight materialization, and the startup timeline. R10.4.

Upstream: (none -- simulator)
Tier: D

No weights are ever read (NG1). What is reproduced is the *shape of startup*: weight
load, profiling run, KV allocation, graph capture, and the total, reported in the same
log line upstream emits.

That matters more than it sounds. R2.7 requires `/health` to report ready only after
load and profiling complete, and a product under test that polls readiness -- or times
out waiting for it -- exercises real behaviour only if startup takes plausible time.
A simulator that is ready instantly cannot surface a readiness bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pvllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StartupTimeline:
    """What startup spent its time on. R10.4."""

    load_weights_seconds: float = 0.0
    profile_run_seconds: float = 0.0
    kv_cache_seconds: float = 0.0
    graph_capture_seconds: float = 0.0
    phases: list[tuple[str, float]] = field(default_factory=list)

    def record(self, phase: str, seconds: float) -> None:
        self.phases.append((phase, seconds))

    @property
    def total_seconds(self) -> float:
        return (
            self.load_weights_seconds
            + self.profile_run_seconds
            + self.kv_cache_seconds
            + self.graph_capture_seconds
        )

    def summary(self, kv_cache_gib: float) -> str:
        """The startup line, shaped like upstream's."""
        return (
            f"init engine (profile, create kv cache, warmup model) took "
            f"{self.total_seconds:.2f} seconds "
            f"(load={self.load_weights_seconds:.2f}s, "
            f"profile={self.profile_run_seconds:.2f}s, "
            f"kv_cache={self.kv_cache_seconds:.2f}s [{kv_cache_gib:.2f}GiB], "
            f"graph_capture={self.graph_capture_seconds:.2f}s) [modeled]"
        )


def materialize_weights(weight_bytes: int, load_bandwidth: float) -> float:
    """ "Load" the weights and return how long it modeled taking.

    Nothing is allocated: a faithful simulator of an 80 GiB device must not need
    80 GiB of host RAM to run, which is the entire reason this project can run on a
    laptop. The ledger tracks the *claim* on device memory; this returns the time.
    """
    if load_bandwidth <= 0:
        raise ValueError(f"load_bandwidth must be positive, got {load_bandwidth}")
    return weight_bytes / load_bandwidth
