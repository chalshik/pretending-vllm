"""The simulated device. R10.1.

Upstream: (none -- simulator)
Tier: D

Owns the memory ledger and the cost model for one device, and is the only object that
advances the clock. Everything below the simulation boundary that costs time or memory
goes through here, which makes the seam auditable: if a duration appeared in a trace,
`SimDevice.execute` put it there.

Streams and synchronization are deliberately absent rather than stubbed. Upstream's
worker overlaps H2D copies with compute on separate CUDA streams; modeling that would
mean modeling the overlap, and a stub that pretends to be a stream while doing nothing
would misreport the very thing it was added to represent. If overlap ever matters to
the cost model, it becomes an explicit term in `RooflineCostModel`, not a fake stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvllm.logger import init_logger
from pvllm.sim.clock import Clock
from pvllm.sim.cost_model import CostModel, StepCost, StepProfile
from pvllm.sim.hardware_db import DeviceCard
from pvllm.sim.memory import MemoryLedger, MemoryProfile, SimOutOfMemoryError

logger = init_logger(__name__)


@dataclass
class SimDevice:
    """One simulated accelerator.

    Args:
        card: The device's declared capabilities (section 8).
        clock: The engine core's clock. Passed in, never created: the core owns it
            (R19.1), and a device that made its own would let time advance in two
            places at once.
        cost_model: What turns a step's shape into a duration.
        rng: The seeded jitter stream, or `None` for no jitter.
        device_id: Index within the fleet.
    """

    card: DeviceCard
    clock: Clock
    cost_model: CostModel
    rng: np.random.Generator | None = None
    device_id: int = 0

    def __post_init__(self) -> None:
        self.ledger = MemoryLedger(self.card.memory_bytes)
        self._last_step_cost: StepCost | None = None
        self._num_steps = 0

    # --- memory --------------------------------------------------------------

    def allocate(self, pool: str, num_bytes: int) -> None:
        """Claim device memory, raising `SimOutOfMemoryError` if it does not fit."""
        self.ledger.allocate(pool, num_bytes)

    def apply_memory_profile(self, profile: MemoryProfile) -> None:
        """Record the resolved profile's pools on the ledger.

        Done as real allocations rather than bookkeeping so that a later allocation
        which would not fit actually fails -- the ledger is the thing that makes a
        capacity answer trustworthy.
        """
        self.allocate("weights", profile.weight_bytes)
        self.allocate("activation_peak", profile.activation_peak_bytes)
        self.allocate("non_torch_overhead", profile.non_torch_overhead_bytes)
        if profile.graph_bytes:
            self.allocate("graph", profile.graph_bytes)
        self.allocate("kv_cache", profile.num_gpu_blocks * profile.kv_bytes_per_block)

    @property
    def free_bytes(self) -> int:
        return self.ledger.free_bytes

    # --- execution -----------------------------------------------------------

    def execute(self, profile: StepProfile) -> StepCost:
        """Model one forward pass and advance the clock by its duration.

        The single place the clock moves during inference. Returning the breakdown
        rather than just the duration is what lets the debug surface explain *why* a
        step took what it took (D9).
        """
        cost = self.cost_model.step_cost(profile, self.rng)
        self.clock.advance(cost.duration)
        self._last_step_cost = cost
        self._num_steps += 1
        return cost

    async def execute_async(self, profile: StepProfile) -> StepCost:
        """As `execute`, but yields to the event loop instead of blocking it.

        Needed under a real or scaled clock in the async engine: blocking the loop
        for the modeled duration would stall the HTTP server that is meant to be
        streaming during it.
        """
        cost = self.cost_model.step_cost(profile, self.rng)
        await self.clock.advance_async(cost.duration)
        self._last_step_cost = cost
        self._num_steps += 1
        return cost

    def load_weights(self, weight_bytes: int) -> float:
        """Model the weight load and advance the clock. R10.4."""
        seconds = self.cost_model.weight_load_seconds(weight_bytes)
        self.clock.advance(seconds)
        return seconds

    def capture_graphs(self, num_shapes: int) -> float:
        """Model graph capture and advance the clock. R8.4."""
        seconds = self.cost_model.graph_capture_seconds(num_shapes)
        self.clock.advance(seconds)
        return seconds

    # --- introspection -------------------------------------------------------

    @property
    def last_step_cost(self) -> StepCost | None:
        """The most recent step's cost breakdown, for the debug endpoints (D9)."""
        return self._last_step_cost

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def __repr__(self) -> str:
        gib = 1 << 30
        return (
            f"SimDevice(card={self.card.name!r}, id={self.device_id}, "
            f"cost_model={self.cost_model.name!r}, "
            f"free={self.ledger.free_bytes / gib:.2f}GiB, steps={self._num_steps})"
        )


__all__ = ["SimDevice", "SimOutOfMemoryError", "StepCost", "StepProfile"]
