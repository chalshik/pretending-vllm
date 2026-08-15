"""Benchmark result shapes. R20.

Upstream: vllm/benchmarks/serve.py (the `BenchmarkMetrics` half)
Tier: B

Field names match upstream's `BenchmarkMetrics` so a script that parses one parses the
other. That is the point of mirroring rather than inventing: somebody's comparison
notebook should not need a pvllm branch.

**Every duration here is modeled, and every result says so.** Upstream measures wall
clock. Doing that here would measure how fast Python simulates a scheduler, which is
not a number anybody wants. So the inputs are `FinishedRequestStats` -- the engine's
own instrumentation, stamped from the engine clock (R19.1) -- rather than a second
timing path built for benchmarks. One source, so a benchmark and a `/metrics` scrape
of the same run cannot disagree about how long a request took.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pvllm.v1.metrics.stats import FinishedRequestStats

#: What upstream reports, so a comparison against its output lines up.
PERCENTILES = (25.0, 50.0, 75.0, 90.0, 95.0, 99.0)


def percentiles(values: list[float], scale: float = 1.0) -> list[list[float]]:
    """`[percentile, value]` pairs, in upstream's shape.

    Lists rather than tuples because these go straight to JSON, where a tuple would
    become a list anyway -- and a golden written from tuples would then not compare
    equal to one read back.
    """
    if not values:
        return [[p, 0.0] for p in PERCENTILES]
    computed = np.percentile(values, PERCENTILES)
    return [[p, float(v) * scale] for p, v in zip(PERCENTILES, computed, strict=True)]


def _stats(values: list[float], scale: float = 1.0) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "percentiles": percentiles([])}
    array = np.asarray(values, dtype=float) * scale
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "percentiles": percentiles(values, scale),
    }


@dataclass
class BenchmarkMetrics:
    """Aggregated results, in upstream's field names."""

    completed: int
    failed: int
    total_input: int
    total_output: int
    duration: float
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    ttft: dict[str, Any] = field(default_factory=dict)
    tpot: dict[str, Any] = field(default_factory=dict)
    e2el: dict[str, Any] = field(default_factory=dict)
    queue: dict[str, Any] = field(default_factory=dict)
    #: R9.5. Never omitted, and never anything but "modeled" from this engine.
    provenance: str = "modeled"

    @classmethod
    def from_finished(
        cls,
        finished: list[FinishedRequestStats],
        duration: float,
        failed: int = 0,
    ) -> BenchmarkMetrics:
        total_input = sum(f.num_prompt_tokens for f in finished)
        total_output = sum(f.num_generation_tokens for f in finished)
        # A zero-duration run would divide by zero. It happens with the constant cost
        # model at a tiny scale, and reporting 0.0 is less wrong than crashing a
        # sweep on its first cell.
        rate = (1.0 / duration) if duration > 0 else 0.0
        return cls(
            completed=len(finished),
            failed=failed,
            total_input=total_input,
            total_output=total_output,
            duration=duration,
            request_throughput=len(finished) * rate,
            output_throughput=total_output * rate,
            total_token_throughput=(total_input + total_output) * rate,
            ttft=_stats([f.time_to_first_token for f in finished], scale=1000.0),
            tpot=_stats([f.time_per_output_token for f in finished], scale=1000.0),
            e2el=_stats([f.e2e_latency for f in finished], scale=1000.0),
            queue=_stats([f.queue_time for f in finished], scale=1000.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "failed": self.failed,
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "duration_s": self.duration,
            "request_throughput": self.request_throughput,
            "output_throughput": self.output_throughput,
            "total_token_throughput": self.total_token_throughput,
            "mean_ttft_ms": self.ttft["mean"],
            "median_ttft_ms": self.ttft["median"],
            "std_ttft_ms": self.ttft["std"],
            "percentiles_ttft_ms": self.ttft["percentiles"],
            "mean_tpot_ms": self.tpot["mean"],
            "median_tpot_ms": self.tpot["median"],
            "std_tpot_ms": self.tpot["std"],
            "percentiles_tpot_ms": self.tpot["percentiles"],
            "mean_e2el_ms": self.e2el["mean"],
            "median_e2el_ms": self.e2el["median"],
            "std_e2el_ms": self.e2el["std"],
            "percentiles_e2el_ms": self.e2el["percentiles"],
            "mean_queue_ms": self.queue["mean"],
            "provenance": self.provenance,
        }

    def render(self, title: str) -> str:
        """Upstream's summary block, plus the disclaimer it does not need."""
        width = 62
        lines = [
            "=" * width,
            f"{title:^{width}}",
            "=" * width,
            f"{'Successful requests:':<42} {self.completed:>19}",
            f"{'Benchmark duration (s, modeled):':<42} {self.duration:>19.3f}",
            f"{'Total input tokens:':<42} {self.total_input:>19}",
            f"{'Total generated tokens:':<42} {self.total_output:>19}",
            f"{'Request throughput (req/s):':<42} {self.request_throughput:>19.2f}",
            f"{'Output token throughput (tok/s):':<42} {self.output_throughput:>19.2f}",
            f"{'Total token throughput (tok/s):':<42} "
            f"{self.total_token_throughput:>19.2f}",
        ]
        sections = (
            ("Time to First Token", self.ttft),
            ("Time per Output Token", self.tpot),
            ("End-to-end Latency", self.e2el),
            ("Queue Time", self.queue),
        )
        for label, stats in sections:
            lines.append(f"{'-' * 4} {label} {'-' * (width - 6 - len(label))}")
            lines.append(f"{'Mean (ms):':<42} {stats['mean']:>19.2f}")
            lines.append(f"{'Median (ms):':<42} {stats['median']:>19.2f}")
            for percentile, value in stats["percentiles"]:
                if percentile in (90.0, 99.0):
                    lines.append(f"{f'P{percentile:g} (ms):':<42} {value:>19.2f}")
        lines.append("=" * width)
        lines.append(
            "Durations are MODELED by the simulated cost model, not measured. "
            "Treat them as shape, not truth."
        )
        return "\n".join(lines)
