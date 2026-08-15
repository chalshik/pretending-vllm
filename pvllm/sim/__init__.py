"""The simulated device, model, and clock -- the only unreal code in the repo.

Upstream: (none -- simulator)
Tier: D

B3: this package is the only place allowed to invent numbers. Memory ledger, clock,
cost model, hardware database, model database, token generator. Nothing outside it
imports randomness or reads wall-clock time, and `tests/unit/test_purity.py` fails
the build if that stops being true.

B4: below the simulation boundary is pure. The same seed plus the same
`SchedulerOutput` sequence gives byte-identical output and identical elapsed virtual
time.
"""

from pvllm.sim.clock import (
    DEFAULT_EPOCH,
    Clock,
    ClockMode,
    RealClock,
    ScaledClock,
    VirtualClock,
    build_clock,
)
from pvllm.sim.rng import RngFactory
from pvllm.sim.trace import (
    TRACE_SCHEMA_VERSION,
    NullTraceWriter,
    TraceSink,
    TraceWriter,
    read_header,
    read_trace,
)

__all__ = [
    "DEFAULT_EPOCH",
    "TRACE_SCHEMA_VERSION",
    "Clock",
    "ClockMode",
    "NullTraceWriter",
    "RealClock",
    "RngFactory",
    "ScaledClock",
    "TraceSink",
    "TraceWriter",
    "VirtualClock",
    "build_clock",
    "read_header",
    "read_trace",
]
