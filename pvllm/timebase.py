"""The clock interface.

Upstream: (none -- pvllm addition)
Tier: B

R19.1 makes the engine core the clock's owner, but B1 forbids the engine core from
knowing a simulator exists. Both hold only if the *interface* lives above the boundary
and the *implementation* is supplied by the platform -- the same seam that supplies the
worker class and the attention backend (B2).

Upstream needs no such thing: it reads wall time. Here the timebase is a property of
the device, which makes the platform exactly the right place to get one from.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """The engine's notion of time.

    Implementations differ only in whether advancing spends real time. See
    `pvllm.sim.clock` for the three modes.
    """

    # Declared read-only so an implementation may narrow it -- `pvllm.sim.clock`
    # types it as a Literal of the three modes.
    @property
    def mode(self) -> str:
        """Which clock this is: `virtual`, `real`, or `scaled`."""
        ...

    def time(self) -> float:
        """Current time as a Unix timestamp."""
        ...

    @property
    def elapsed(self) -> float:
        """Time since the run started."""
        ...

    @property
    def is_virtual(self) -> bool:
        """Whether durations are modeled rather than measured.

        Surfaced in metric help strings so a dashboard can tell (R12.4).
        """
        ...

    def advance(self, duration: float) -> float:
        """Advance by `duration` seconds, sleeping if the mode calls for it."""
        ...

    async def advance_async(self, duration: float) -> float:
        """As `advance`, but yields to the event loop instead of blocking it."""
        ...
