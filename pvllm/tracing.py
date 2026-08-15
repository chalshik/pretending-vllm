"""The trace sink interface.

Upstream: (none -- pvllm addition)
Tier: B

R19.3's event trace is a pretending-vllm addition; upstream's scheduler emits no such
thing. But the *interface* belongs above the simulation boundary, because the control
plane is what emits events -- only the writing of them is Tier D.

Defining the protocol here rather than in `pvllm/sim/trace.py` is what lets
`v1/core` and `v1/engine` reference it without importing the simulator (B1). The
boundary lint in `tests/unit/test_purity.py` catches the alternative.

It is a `Protocol`, so `pvllm.sim.trace.TraceWriter` satisfies it structurally without
importing anything from here either.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TraceSink(Protocol):
    """Somewhere engine events can be recorded."""

    enabled: bool

    def emit(self, event_type: str, t: float | None = None, **fields: Any) -> None:
        """Append one event.

        `t` is modeled time from the engine core's clock, never wall-clock time
        (R19.1). Callers below the engine core leave it unset and let the core stamp
        the record, because they have no clock to read.
        """
        ...

    def close(self) -> None: ...
