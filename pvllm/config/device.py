"""Device configuration, and the simulator's own knobs.

Upstream: vllm/config/device.py
Tier: C

R1.3: every simulator-specific field lives in `SimConfig` and is reached through
`DeviceConfig`. Keeping them in one place means the rest of the config surface stays a
mirror of upstream, and a reader can see the entire "what is fake" surface at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pvllm.sim.clock import ClockMode
from pvllm.sim.hardware_db import DEFAULT_DEVICE_CARD, DeviceCard, load_device_card

OutputLengthPolicy = Literal[
    "fixed", "uniform", "lognormal", "from_request", "from_fixture"
]
ContentPolicy = Literal["pseudoword", "echo", "fixture"]
CostModelProfile = Literal["constant", "roofline"]


@dataclass
class SimConfig:
    """The simulator's knobs. R1.3.

    Everything here is a dial on something that does not exist. Nothing in this class
    has an upstream counterpart, because upstream has actual hardware.
    """

    #: Hardware card name or path to a JSON card (section 8).
    device_card: str = DEFAULT_DEVICE_CARD
    num_devices: int = 1

    #: R19.1. `virtual` never sleeps; `real` sleeps the modeled duration; `scaled`
    #: sleeps `duration / time_scale`.
    clock_mode: ClockMode = "virtual"
    time_scale: float = 1.0

    #: R9. `constant` is deterministic and fast, for tests. `roofline` is the
    #: calibrated-ish model whose numbers are labeled `modeled` wherever they surface.
    cost_model_profile: CostModelProfile = "constant"
    #: Multiplicative jitter on step latency, N(0, sigma). Seeded (R19.2).
    jitter_sigma: float = 0.0

    #: Explicit model card, overriding lookup by model name.
    model_card: str | None = None

    #: R11.2. What decides how many tokens a request generates. This knob is what
    #: makes workload experiments meaningful.
    output_length_policy: OutputLengthPolicy = "from_request"
    output_length_fixed: int = 128
    output_length_range: tuple[int, int] = (16, 256)
    output_length_lognormal: tuple[float, float] = (4.0, 0.75)
    #: R11.3. Content must detokenize to stable text so HTTP golden tests are possible.
    content_policy: ContentPolicy = "pseudoword"

    #: R14. How often a draft token is accepted, under speculative decoding. The
    #: one number a simulator cannot derive: it is the agreement between a draft
    #: model and a target model, and there is neither. Measure it on your real pair
    #: and set it here; everything downstream -- the scheduling, the token
    #: accounting, the spec_decode metrics -- is then faithful.
    spec_acceptance_rate: float = 0.7

    #: R19.2. One seed reproduces the whole run.
    seed: int = 0
    #: R19.3. Where the JSONL event trace goes. `None` disables tracing.
    trace_path: str | None = None

    resolved_device_card: DeviceCard = field(init=False)

    def __post_init__(self) -> None:
        if self.num_devices < 1:
            raise ValueError(f"num_devices must be at least 1, got {self.num_devices}")
        if self.time_scale <= 0.0:
            raise ValueError(f"time_scale must be positive, got {self.time_scale}")
        if not 0.0 <= self.spec_acceptance_rate <= 1.0:
            raise ValueError(
                f"spec_acceptance_rate must be in [0, 1], got "
                f"{self.spec_acceptance_rate}"
            )
        if self.jitter_sigma < 0.0:
            raise ValueError(
                f"jitter_sigma must be non-negative, got {self.jitter_sigma}"
            )
        if self.clock_mode not in ("virtual", "real", "scaled"):
            raise ValueError(
                f"unknown clock_mode {self.clock_mode!r}; expected 'virtual', 'real', "
                f"or 'scaled'"
            )
        if self.cost_model_profile not in ("constant", "roofline"):
            raise ValueError(
                f"unknown cost_model_profile {self.cost_model_profile!r}; expected "
                f"'constant' or 'roofline'"
            )
        low, high = self.output_length_range
        if low < 1 or high < low:
            raise ValueError(
                f"output_length_range must be a non-empty positive range, got "
                f"({low}, {high})"
            )

        card = load_device_card(self.device_card)
        # The card ships a device count; an explicit num_devices overrides it.
        self.resolved_device_card = (
            card
            if self.num_devices == card.num_devices
            else _with_devices(card, self.num_devices)
        )


def _with_devices(card: DeviceCard, num_devices: int) -> DeviceCard:
    from dataclasses import replace

    return replace(card, num_devices=num_devices)


@dataclass
class DeviceConfig:
    """Configuration for the device to execute on."""

    device: str = "auto"
    sim_config: SimConfig = field(default_factory=SimConfig)

    device_type: str = field(init=False)

    def __post_init__(self) -> None:
        if self.device == "auto":
            from pvllm.platforms import current_platform

            self.device_type = current_platform.device_type
        else:
            self.device_type = str(self.device)

        # Record the configured card so the platform's device-introspection
        # classmethods -- which have no config in hand, exactly as upstream's probe a
        # real device -- report the card this engine is actually running against.
        from pvllm.sim.hardware_db import set_active_device_card

        set_active_device_card(self.sim_config.resolved_device_card)
