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
from pvllm.sim.cost_model import (
    ConstantCostModel,
    CostModel,
    RooflineCostModel,
    StepCost,
    StepProfile,
    build_cost_model,
)
from pvllm.sim.device import SimDevice
from pvllm.sim.hardware_db import DeviceCard, load_device_card
from pvllm.sim.memory import (
    MemoryLedger,
    MemoryProfile,
    SimOutOfMemoryError,
    compute_memory_profile,
)
from pvllm.sim.model import SimModel
from pvllm.sim.model_db import ModelCard, load_model_card
from pvllm.sim.rng import RngFactory
from pvllm.sim.trace import (
    TRACE_SCHEMA_VERSION,
    NullTraceWriter,
    TraceSink,
    TraceWriter,
    read_header,
    read_trace,
)
from pvllm.sim.weights import StartupTimeline, materialize_weights

__all__ = [
    "DEFAULT_EPOCH",
    "TRACE_SCHEMA_VERSION",
    "Clock",
    "ClockMode",
    "ConstantCostModel",
    "CostModel",
    "DeviceCard",
    "MemoryLedger",
    "MemoryProfile",
    "ModelCard",
    "NullTraceWriter",
    "RealClock",
    "RngFactory",
    "RooflineCostModel",
    "ScaledClock",
    "SimDevice",
    "SimModel",
    "SimOutOfMemoryError",
    "StartupTimeline",
    "StepCost",
    "StepProfile",
    "TraceSink",
    "TraceWriter",
    "VirtualClock",
    "build_clock",
    "build_cost_model",
    "compute_memory_profile",
    "load_device_card",
    "load_model_card",
    "materialize_weights",
    "read_header",
    "read_trace",
]
