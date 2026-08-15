"""The clock. R19.1.

Upstream: (none -- simulator)
Tier: D

Three modes:

* ``virtual`` -- advances only by modeled durations. Never sleeps. A 30 minute load
  test runs in seconds.
* ``real`` -- sleeps the modeled duration, so a product under test observes true
  latency.
* ``scaled`` -- sleeps ``duration / time_scale``.

**All three share one modeled timeline.** ``time()`` returns accumulated modeled time
in every mode; the modes differ only in whether, and for how long, they sleep. Two
consequences worth being explicit about:

* Metrics and output timestamps are byte-identical across all three modes, so a
  virtual-clock CI run and a real-clock demo produce the same numbers. This is what
  makes B4 ("identical elapsed virtual time") hold in real mode too.
* In ``real`` mode the interpreter's own execution time is *not* added to the
  timeline. If a step models 100 ms but Python takes 130 ms, the engine reports
  100 ms while a client measuring with its own wall clock sees 130 ms. The modeled
  number is the honest one to report -- the extra 30 ms is simulator overhead that a
  real vLLM would not have -- but a consumer comparing the two will see the gap.

The timeline starts at a fixed epoch rather than at "now" so that a run is
reproducible down to its timestamps (R19.2). Timestamps still look like ordinary Unix
times, which matters because they surface in the OpenAI ``created`` field.

**Ownership (R19.1, enforced from the first commit).** The engine core owns the clock
and is the only component that advances it. Nothing else reads wall-clock time --
``tests/unit/test_purity.py`` fails the build if any module outside ``pvllm/sim/``
touches ``time.time``, ``time.monotonic``, or ``time.perf_counter``. This is the part
that cannot be retrofitted cheaply (D2), because once a dozen call sites read the
clock directly, determinism is gone and getting it back means auditing all of them.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Literal

ClockMode = Literal["virtual", "real", "scaled"]

#: 2026-01-01T00:00:00Z. A fixed, arbitrary origin. Fixed so runs are reproducible;
#: plausible so timestamps that reach an API consumer do not look absurd.
DEFAULT_EPOCH = 1767225600.0


class Clock(ABC):
    """Base clock. Subclasses differ only in how ``advance`` spends real time."""

    mode: ClockMode

    def __init__(self, *, epoch: float = DEFAULT_EPOCH) -> None:
        self._epoch = epoch
        self._elapsed = 0.0

    def time(self) -> float:
        """Current time on the modeled timeline, as a Unix timestamp."""
        return self._epoch + self._elapsed

    @property
    def elapsed(self) -> float:
        """Modeled time elapsed since the run started."""
        return self._elapsed

    @property
    def is_virtual(self) -> bool:
        """Whether durations are modeled rather than measured.

        Surfaced in metric help strings so a dashboard can tell (R12.4).
        """
        return True

    def advance(self, duration: float) -> float:
        """Advance the modeled timeline by ``duration`` seconds, sleeping if the mode
        calls for it. Returns the new time."""
        if duration < 0.0:
            raise ValueError(f"cannot advance the clock by {duration}s")
        self._sleep(duration)
        self._elapsed += duration
        return self.time()

    async def advance_async(self, duration: float) -> float:
        """As ``advance``, but yields to the event loop instead of blocking it."""
        if duration < 0.0:
            raise ValueError(f"cannot advance the clock by {duration}s")
        await self._sleep_async(duration)
        self._elapsed += duration
        return self.time()

    @abstractmethod
    def _sleep(self, duration: float) -> None: ...

    @abstractmethod
    async def _sleep_async(self, duration: float) -> None: ...

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(t={self.time():.6f}, elapsed={self._elapsed:.6f})"
        )


class VirtualClock(Clock):
    """Advances instantly. The default, and what makes cheap experiments possible (G5)."""

    mode: ClockMode = "virtual"

    def _sleep(self, duration: float) -> None:
        return None

    async def _sleep_async(self, duration: float) -> None:
        return None


class RealClock(Clock):
    """Sleeps the modeled duration, so a product under test observes true latency."""

    mode: ClockMode = "real"

    @property
    def is_virtual(self) -> bool:
        return False

    def _sleep(self, duration: float) -> None:
        if duration > 0.0:
            time.sleep(duration)

    async def _sleep_async(self, duration: float) -> None:
        if duration > 0.0:
            await asyncio.sleep(duration)


class ScaledClock(Clock):
    """Sleeps ``duration / time_scale``.

    ``time_scale > 1`` compresses a run; ``time_scale < 1`` stretches it. The modeled
    timeline still advances by the full duration, so reported latencies are unchanged.
    """

    mode: ClockMode = "scaled"

    def __init__(self, time_scale: float, *, epoch: float = DEFAULT_EPOCH) -> None:
        super().__init__(epoch=epoch)
        if time_scale <= 0.0:
            raise ValueError(f"time_scale must be positive, got {time_scale}")
        self.time_scale = time_scale

    @property
    def is_virtual(self) -> bool:
        return False

    def _sleep(self, duration: float) -> None:
        scaled = duration / self.time_scale
        if scaled > 0.0:
            time.sleep(scaled)

    async def _sleep_async(self, duration: float) -> None:
        scaled = duration / self.time_scale
        if scaled > 0.0:
            await asyncio.sleep(scaled)


def build_clock(
    mode: ClockMode,
    *,
    time_scale: float = 1.0,
    epoch: float = DEFAULT_EPOCH,
) -> Clock:
    """Construct the clock named by ``SimConfig.clock_mode`` (R1.3)."""
    if mode == "virtual":
        return VirtualClock(epoch=epoch)
    if mode == "real":
        return RealClock(epoch=epoch)
    if mode == "scaled":
        return ScaledClock(time_scale, epoch=epoch)
    raise ValueError(
        f"unknown clock_mode {mode!r}; expected one of 'virtual', 'real', 'scaled'"
    )
