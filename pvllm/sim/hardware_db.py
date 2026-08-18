"""Device cards. Section 8.

Upstream: (none -- simulator)
Tier: D

Hardware becomes a JSON file. A card describes a device that does not exist well
enough to answer "does this model at this context length fit on N of these, and
roughly how fast would it be" -- which is the second of the three consequences that
justify this project's design.

Cards ship in `pvllm/sim/hardware/*.json`. Three are bundled: a datacenter class, a
workstation class, and a deliberately tiny one whose only job is to force preemption
and OOM in tests without needing a large synthetic workload.

**The performance numbers are approximate and uncalibrated.** They are order-of-
magnitude figures for a *class* of device, not measurements of a specific product,
and they are inputs to a roofline model with a published error band (R9.5). The
`provenance` field on every card says so. Run `tools/calibrate_cost_model.py` against
real hardware to replace them with fitted values.

The `memory_bytes` and bandwidth figures, by contrast, feed the *analytic* memory
model (R10.2), which is exact given the card -- `num_gpu_blocks` and `max_concurrency`
are arithmetic, not estimates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

HARDWARE_DIR = Path(__file__).parent / "hardware"

#: Used before a config resolves, and by `pvllm.platforms.sim.SimPlatform` when it is
#: asked for device facts outside a configured engine.
DEFAULT_DEVICE_CARD = "datacenter-80gb"


@dataclass(frozen=True)
class DeviceCard:
    """A simulated device. R9.1."""

    name: str
    #: Total device memory, in bytes. Feeds the analytic memory model exactly.
    memory_bytes: int
    #: HBM/VRAM bandwidth, bytes per second. The memory term of the roofline.
    memory_bandwidth: float
    #: Peak *dense* FLOPs per second, keyed by dtype name. Sparsity-doubled vendor
    #: figures are deliberately not used -- no kernel here exploits sparsity.
    peak_flops: dict[str, float]
    #: Per-direction interconnect bandwidth between devices, bytes per second.
    #: Feeds the tensor-parallel allreduce term.
    interconnect_bandwidth: float
    #: Per-kernel launch overhead, seconds.
    launch_overhead: float
    #: Bandwidth at which weights are "read from disk" at startup (R10.4).
    load_bandwidth: float
    #: How many of these the fleet has. Overridden by `--num-devices`.
    num_devices: int = 1
    #: Achievable fractions of peak. `mfu` for compute, `bw_eff` for memory,
    #: `link_eff` for interconnect. These are the knobs calibration fits (R9.4).
    mfu: float = 0.45
    bw_eff: float = 0.80
    link_eff: float = 0.75
    #: Where these numbers came from, and how much to trust them.
    provenance: str = "uncalibrated approximation"
    extra: dict[str, Any] = field(default_factory=dict)

    def peak_flops_for(self, dtype: str) -> float:
        """Peak dense FLOPs for a dtype, or a clear failure naming what is available."""
        try:
            return self.peak_flops[dtype]
        except KeyError:
            raise KeyError(
                f"device card {self.name!r} has no peak_flops entry for dtype "
                f"{dtype!r}; it declares {sorted(self.peak_flops)}"
            ) from None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceCard:
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        extra = {k: v for k, v in data.items() if k not in known}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs, extra=extra)


@cache
def available_device_cards() -> tuple[str, ...]:
    """Names of the bundled cards."""
    return tuple(sorted(p.stem for p in HARDWARE_DIR.glob("*.json")))


@cache
def load_device_card(name: str) -> DeviceCard:
    """Load a card by name, or by path to a JSON file.

    A user-supplied path lets you model hardware that does not ship with the package
    without vendoring a card into it.
    """
    candidate = Path(name)
    if candidate.suffix == ".json" and candidate.is_file():
        return DeviceCard.from_dict(json.loads(candidate.read_text(encoding="utf-8")))

    path = HARDWARE_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"unknown device card {name!r}; bundled cards are "
            f"{list(available_device_cards())}, or pass a path to a JSON file"
        )
    return DeviceCard.from_dict(json.loads(path.read_text(encoding="utf-8")))


_active_card: DeviceCard | None = None


def set_active_device_card(card: DeviceCard) -> None:
    """Record the card the running engine is configured with.

    `SimPlatform`'s device-introspection classmethods have no config in hand -- their
    upstream counterparts probe a real device instead. This module-level handle is the
    simulator's stand-in for that device.
    """
    global _active_card
    _active_card = card


def get_active_device_card() -> DeviceCard:
    """The configured card, or the default if no engine has configured one yet."""
    if _active_card is None:
        return load_device_card(DEFAULT_DEVICE_CARD)
    return _active_card


def reset_active_device_card() -> None:
    """Drop the configured card. For test isolation."""
    global _active_card
    _active_card = None
