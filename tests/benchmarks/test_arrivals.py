"""Synthetic arrival processes. R20.

The arrival process is the independent variable of a serving benchmark: everything
`bench serve` reports about queueing is downstream of it. A subtly wrong one produces
believable numbers for a workload nobody asked for.
"""

from __future__ import annotations

import itertools
import math

import pytest

from pvllm.benchmarks.lib.arrivals import arrival_times
from pvllm.sim.rng import RngFactory


def stream(seed: int = 0):
    return RngFactory(seed).stream("benchmark-arrivals")


def test_an_infinite_rate_is_an_instant_batch():
    """The offline case: everything arrives before the first step, which is what
    makes a throughput benchmark a throughput benchmark."""
    assert arrival_times(5, float("inf"), stream()) == [0.0] * 5


def test_the_first_request_arrives_at_zero():
    """A leading idle gap would inflate every reported duration by a random amount,
    since the benchmark clock starts with the run and not with the first arrival."""
    times = arrival_times(10, 5.0, stream())
    assert times[0] == 0.0


def test_arrivals_are_non_decreasing():
    times = arrival_times(50, 10.0, stream())
    assert times == sorted(times)


def test_the_mean_gap_tracks_the_requested_rate():
    """Statistical, so it is asserted loosely -- but a rate that is wrong by a
    factor (a 1/rate inverted, say) fails this by a mile."""
    rate = 20.0
    times = arrival_times(2000, rate, stream())
    mean_gap = times[-1] / (len(times) - 1)
    assert math.isclose(mean_gap, 1.0 / rate, rel_tol=0.15)


def test_infinite_burstiness_spaces_requests_uniformly():
    """The gamma's variance goes to zero. Special-cased in the implementation
    because `gamma(inf, 0)` is undefined, so it is worth pinning."""
    times = arrival_times(6, 2.0, stream(), burstiness=float("inf"))
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert all(math.isclose(gap, 0.5) for gap in gaps)


def test_low_burstiness_is_burstier_than_high():
    """The knob has to actually move the thing it names. Variance of the gaps is
    what 'bursty' means, and a scheduler's tail latency is far more sensitive to it
    than to the mean rate."""

    def gap_variance(burstiness: float) -> float:
        times = arrival_times(2000, 10.0, stream(), burstiness=burstiness)
        gaps = [b - a for a, b in itertools.pairwise(times)]
        mean = sum(gaps) / len(gaps)
        return sum((g - mean) ** 2 for g in gaps) / len(gaps)

    assert gap_variance(0.2) > gap_variance(1.0) > gap_variance(5.0)


def test_the_same_seed_gives_the_same_arrivals():
    """B4. Two runs of a benchmark must differ only in what was configured, or an
    A/B comparison of two configs is partly a comparison of two workloads."""
    assert arrival_times(30, 8.0, stream(7)) == arrival_times(30, 8.0, stream(7))


def test_different_seeds_give_different_arrivals():
    assert arrival_times(30, 8.0, stream(1)) != arrival_times(30, 8.0, stream(2))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"request_rate": 0.0}, "request_rate must be positive"),
        ({"request_rate": -1.0}, "request_rate must be positive"),
        ({"request_rate": 5.0, "burstiness": 0.0}, "burstiness must be positive"),
    ],
)
def test_nonsense_parameters_are_refused(kwargs, match):
    """Rather than silently producing an empty or degenerate schedule -- which would
    make a benchmark report a real-looking number for a run that never happened."""
    with pytest.raises(ValueError, match=match):
        arrival_times(10, rng=stream(), **kwargs)


def test_zero_requests_is_an_empty_schedule():
    assert arrival_times(0, 10.0, stream()) == []
