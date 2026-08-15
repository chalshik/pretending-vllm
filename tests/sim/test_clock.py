"""Clock semantics. R19.1."""

from __future__ import annotations

import pytest

from pvllm.sim.clock import (
    DEFAULT_EPOCH,
    RealClock,
    ScaledClock,
    VirtualClock,
    build_clock,
)


def test_virtual_clock_advances_without_sleeping():
    clock = VirtualClock()
    assert clock.time() == DEFAULT_EPOCH
    clock.advance(1800.0)  # half an hour of modeled time
    assert clock.elapsed == 1800.0
    assert clock.time() == DEFAULT_EPOCH + 1800.0


@pytest.mark.parametrize(
    "clock",
    [VirtualClock(), RealClock(), ScaledClock(1000.0)],
    ids=["virtual", "real", "scaled"],
)
def test_all_modes_share_one_modeled_timeline(clock):
    """The modes differ in how they sleep, never in what time they report.

    This is what lets a virtual-clock CI run and a real-clock demo produce identical
    metrics, and what makes B4 hold outside virtual mode.
    """
    for _ in range(4):
        clock.advance(0.001)
    assert clock.elapsed == pytest.approx(0.004)
    assert clock.time() == pytest.approx(DEFAULT_EPOCH + 0.004)


def test_scaled_clock_sleeps_less_than_real_time():
    scaled = ScaledClock(100.0)
    scaled.advance(0.5)
    # The modeled timeline still moved the full half second.
    assert scaled.elapsed == pytest.approx(0.5)


def test_is_virtual_distinguishes_modeled_from_slept():
    """R12.4: metric help strings need to say whether durations were modeled."""
    assert VirtualClock().is_virtual is True
    assert RealClock().is_virtual is False
    assert ScaledClock(2.0).is_virtual is False


def test_clock_never_runs_backwards():
    clock = VirtualClock()
    with pytest.raises(ValueError, match="cannot advance"):
        clock.advance(-0.001)


async def test_async_advance_matches_sync():
    clock = VirtualClock()
    await clock.advance_async(0.25)
    assert clock.elapsed == pytest.approx(0.25)
    with pytest.raises(ValueError, match="cannot advance"):
        await clock.advance_async(-1.0)


def test_build_clock_dispatches_on_mode():
    assert isinstance(build_clock("virtual"), VirtualClock)
    assert isinstance(build_clock("real"), RealClock)
    assert isinstance(build_clock("scaled", time_scale=4.0), ScaledClock)


def test_build_clock_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown clock_mode"):
        build_clock("wallclock")  # type: ignore[arg-type]


def test_scaled_clock_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="time_scale must be positive"):
        ScaledClock(0.0)
