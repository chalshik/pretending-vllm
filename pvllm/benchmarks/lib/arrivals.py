"""Synthetic arrival processes. R20.

Upstream: vllm/benchmarks/serve.py (the `get_request` half)
Tier: B

Requests do not arrive in a batch in production; they arrive as a stochastic process,
and the queueing behavior a benchmark is trying to measure is a function of *that*.
Upstream samples inter-arrival gaps from a gamma distribution parameterized by a
request rate and a burstiness factor. This is the same arithmetic.

The one change: the generator comes from `RngFactory`, so a benchmark run is
reproducible from its seed. Upstream draws from the global `np.random`, which is fine
when you are measuring hardware -- the noise averages out over a real run. Here it
would not: a simulated run is otherwise exactly reproducible (B4), and an arrival
process that varied between runs would be the only source of variance in the whole
system, which would make every A/B comparison of two configs partly a comparison of
two different workloads.

The purity lint (`tests/unit/test_purity.py`) is what enforces that -- `np.random` is
unreachable from here, including in a type annotation, so there is no version of this
file that quietly regresses. Hence `GammaSource`: the generator arrives as a
structural type, the same way `Clock` and `TraceSink` cross their boundaries.
"""

from __future__ import annotations

from typing import Any, Protocol


class GammaSource(Protocol):
    """A seeded source of gamma-distributed samples.

    Narrower than `numpy.random.Generator` on purpose -- this module needs exactly
    one distribution, and naming only that makes the dependency legible and keeps
    `numpy.random` out of a file that must not reach for it.
    """

    def gamma(self, shape: float, scale: float, size: int) -> Any: ...


def arrival_times(
    num_requests: int,
    request_rate: float,
    rng: GammaSource,
    burstiness: float = 1.0,
) -> list[float]:
    """Absolute arrival times, in seconds from the start of the run.

    Args:
        num_requests: How many requests arrive.
        request_rate: Requests per second. `inf` means all at once, which is the
            offline batch case and the default upstream uses too.
        rng: A seeded generator, from `RngFactory.stream(...)`.
        burstiness: Shape of the gamma the gaps are drawn from. `1.0` is a Poisson
            process (exponential gaps). Below 1 is burstier -- long quiet stretches
            punctuated by clumps, which is what actually breaks a scheduler. Above 1
            tends toward uniform spacing, and `inf` is exactly uniform.

    Returns:
        Non-decreasing arrival times, starting at 0.0.
    """
    if num_requests < 0:
        raise ValueError(f"num_requests must be non-negative, got {num_requests}")
    if burstiness <= 0:
        raise ValueError(f"burstiness must be positive, got {burstiness}")
    if request_rate <= 0:
        raise ValueError(
            f"request_rate must be positive (or inf for an instant batch), got "
            f"{request_rate}"
        )

    if request_rate == float("inf"):
        return [0.0] * num_requests

    if burstiness == float("inf"):
        # The gamma's variance goes to zero, so gaps become exactly 1/rate. Special
        # cased because np.random.gamma(inf, 0) is not defined.
        gaps = [1.0 / request_rate] * num_requests
    else:
        theta = 1.0 / (request_rate * burstiness)
        gaps = rng.gamma(shape=burstiness, scale=theta, size=num_requests).tolist()

    times: list[float] = []
    elapsed = 0.0
    # The first request arrives at 0, not after a gap: a benchmark's clock starts
    # when its first request does, and an initial idle gap would inflate every
    # duration by a random amount.
    for gap in gaps:
        times.append(elapsed)
        elapsed += gap
    return times
