"""How long a step takes. R9 -- the highest-risk requirement in the spec.

Upstream: (none -- simulator)
Tier: D

**These numbers are modeled, not measured (R9.5).** This is the one place the project
can be wrong in a way that misleads: everything else is either exactly right or
obviously fake, whereas a latency figure looks like a measurement whatever its
provenance. Anything that surfaces a duration from here is required to carry a
`modeled` label.

Two profiles:

* `constant` -- fixed per-token and per-step costs. Deterministic, fast, and
  deliberately unrealistic. The default for tests (R9.6), because a test asserting on
  step *counts* should not also depend on the roofline's coefficients.
* `roofline` -- the model in R9.2: a compute term, a memory term, their maximum, plus
  communication and launch overheads.

The roofline reproduces the qualitative regimes without special-casing them (R9.3),
and that is the bar it is held to rather than absolute accuracy. Prefill is
compute-bound and linear in tokens because `flops_step` scales with `T_step` while
`bytes_step` is dominated by the fixed weight read. Decode is memory-bound and nearly
flat because one token per request reads the same weights, until KV traffic grows with
context and starts to dominate. Neither regime is coded for; both fall out of the
`max(t_compute, t_memory)`.

Uncalibrated until `tools/calibrate_cost_model.py` is run against real hardware, which
fits `mfu`, `bw_eff`, and `launch_overhead` to observed latencies (R9.4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from pvllm.sim.hardware_db import DeviceCard
from pvllm.sim.model_db import DTYPE_BYTES, ModelCard

#: Kernel launches per layer, eager. Roughly: qkv, attention, output projection,
#: two norms, three MLP matmuls, plus residuals.
KERNELS_PER_LAYER_EAGER = 12
#: With a captured graph the whole step is a handful of launches (R8.4).
KERNELS_PER_STEP_CAPTURED = 8


@dataclass
class StepProfile:
    """What the runner scheduled this step -- the cost model's input.

    Built from real attention metadata (R8.2), which is why bugs in that metadata
    are observable as wrong latencies rather than being silently absorbed.
    """

    #: Total tokens across every request. `T_step`.
    num_tokens: int
    num_reqs: int
    #: Per request: new tokens this step (`T_new_r`) and total context (`ctx_r`).
    query_lens: list[int] = field(default_factory=list)
    seq_lens: list[int] = field(default_factory=list)
    #: R8.4: a captured graph shape and no chunked prefill means lower launch cost.
    is_graph_hit: bool = False


@dataclass(frozen=True)
class StepCost:
    """A step's modeled duration, broken down.

    The breakdown is kept rather than collapsed to a single float because the debug
    surface (D9) exposes it: seeing that a step was memory-bound at 82% weight
    traffic explains a latency curve in a way one number cannot.
    """

    duration: float
    compute_seconds: float
    memory_seconds: float
    comm_seconds: float
    overhead_seconds: float
    jitter_factor: float
    flops: float
    bytes_moved: float

    @property
    def is_compute_bound(self) -> bool:
        return self.compute_seconds >= self.memory_seconds

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "duration": self.duration,
            "compute_s": self.compute_seconds,
            "memory_s": self.memory_seconds,
            "comm_s": self.comm_seconds,
            "overhead_s": self.overhead_seconds,
            "jitter": self.jitter_factor,
            "flops": self.flops,
            "bytes": self.bytes_moved,
            "bound_by": "compute" if self.is_compute_bound else "memory",
            "provenance": "modeled",
        }


class CostModel(ABC):
    """Turns a step's shape into a duration."""

    name: str

    @abstractmethod
    def step_cost(
        self, profile: StepProfile, rng: np.random.Generator | None = None
    ) -> StepCost: ...

    @abstractmethod
    def weight_load_seconds(self, weight_bytes: int) -> float:
        """How long the simulated weight load takes (R10.4)."""

    @abstractmethod
    def graph_capture_seconds(self, num_shapes: int) -> float:
        """Startup cost of capturing graphs (R8.4)."""


class ConstantCostModel(CostModel):
    """Fixed costs. Deterministic and deliberately unrealistic. R9.6."""

    name = "constant"

    def __init__(
        self,
        per_token_seconds: float = 1e-5,
        per_step_seconds: float = 1e-3,
        per_request_seconds: float = 1e-5,
    ) -> None:
        self.per_token_seconds = per_token_seconds
        self.per_step_seconds = per_step_seconds
        self.per_request_seconds = per_request_seconds

    def step_cost(
        self, profile: StepProfile, rng: np.random.Generator | None = None
    ) -> StepCost:
        duration = (
            self.per_step_seconds
            + profile.num_tokens * self.per_token_seconds
            + profile.num_reqs * self.per_request_seconds
        )
        return StepCost(
            duration=duration,
            compute_seconds=duration,
            memory_seconds=0.0,
            comm_seconds=0.0,
            overhead_seconds=0.0,
            jitter_factor=1.0,
            flops=0.0,
            bytes_moved=0.0,
        )

    def weight_load_seconds(self, weight_bytes: int) -> float:
        return 0.0

    def graph_capture_seconds(self, num_shapes: int) -> float:
        return 0.0


class RooflineCostModel(CostModel):
    """The R9.2 model: max(compute, memory) plus communication and launch overhead."""

    name = "roofline"

    def __init__(
        self,
        model: ModelCard,
        device: DeviceCard,
        *,
        dtype: str,
        kv_cache_dtype: str | None = None,
        tp_size: int = 1,
        pp_size: int = 1,
        jitter_sigma: float = 0.0,
        enforce_eager: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.jitter_sigma = jitter_sigma
        self.enforce_eager = enforce_eager

        self.dtype_bytes = DTYPE_BYTES[dtype]
        self.peak_flops = device.peak_flops_for(dtype)
        self.heads_local = max(1, model.num_attention_heads // tp_size)
        self.kv_bytes_per_token = model.kv_bytes_per_token(kv_cache_dtype, tp_size)

        # **Pipeline parallelism shards memory, not step latency.** Each stage holds
        # `num_layers / pp` layers, so per-device memory divides -- but a batch still
        # traverses every stage before a token comes out, so the step costs the whole
        # model's work either way. Dividing the compute and memory terms by `pp_size`
        # (as an earlier version did) would report a step `pp_size` times faster than
        # it is.
        #
        # What that leaves unmodeled is the *throughput* gain: real pipeline
        # parallelism overlaps microbatches so the steady state approaches one
        # stage's time. There are no virtual engines here, so PP shows up as "same
        # latency, less memory per device" -- correct for a single request and
        # pessimistic for a saturated one.
        self.layers_local = model.num_hidden_layers
        self.layers_per_stage = max(1, model.num_hidden_layers // pp_size)

        from pvllm.sim.memory import compute_weight_bytes

        # Every stage's weights are read once per step, so the traffic for a full
        # traversal is the whole (TP-sharded) weight set regardless of `pp_size`.
        self.weight_bytes_local = compute_weight_bytes(model, dtype, tp_size)
        # MoE reads only the routed experts per token, so the *active* count is what
        # the compute term uses -- which is why an MoE is far cheaper to run than its
        # parameter count suggests.
        self.active_params_local = model.num_active_parameters // tp_size

    def step_cost(
        self, profile: StepProfile, rng: np.random.Generator | None = None
    ) -> StepCost:
        tokens = profile.num_tokens

        # --- compute -------------------------------------------------------
        # Two FLOPs per parameter per token (a multiply and an add), plus attention,
        # which is quadratic in context rather than linear in parameters.
        flops = 2.0 * self.active_params_local * tokens
        attention_positions = sum(
            q * s for q, s in zip(profile.query_lens, profile.seq_lens, strict=False)
        )
        flops += (
            4.0
            * self.layers_local
            * self.heads_local
            * self.model.head_dim
            * attention_positions
        )
        t_compute = flops / (self.device.mfu * self.peak_flops)

        # --- memory --------------------------------------------------------
        # Every step reads the whole weight set once; that fixed cost is what makes
        # decode memory-bound. KV traffic scales with total context, which is what
        # eventually overtakes it at long context.
        kv_bytes = sum(profile.seq_lens) * self.kv_bytes_per_token
        activation_bytes = tokens * self.model.hidden_size * self.dtype_bytes * 4
        bytes_moved = self.weight_bytes_local + kv_bytes + activation_bytes
        t_memory = bytes_moved / (self.device.bw_eff * self.device.memory_bandwidth)

        # --- communication --------------------------------------------------
        # Two all-reduces per layer under tensor parallelism, each moving a
        # hidden-sized activation per token. Zero at TP=1.
        t_comm = 0.0
        link = self.device.link_eff * self.device.interconnect_bandwidth
        if self.tp_size > 1:
            volume = 2.0 * tokens * self.model.hidden_size * self.dtype_bytes
            t_comm = volume / link * self.layers_local
        if self.pp_size > 1:
            # One activation hand-off per stage boundary, hidden-sized per token.
            # Far cheaper than tensor parallelism's per-layer all-reduces, which is
            # the whole reason pipeline parallelism is what crosses slow links.
            handoffs = self.pp_size - 1
            t_comm += (
                handoffs * tokens * self.model.hidden_size * self.dtype_bytes / link
            )

        # --- launch overhead -------------------------------------------------
        graph_hit = profile.is_graph_hit and not self.enforce_eager
        num_kernels = (
            KERNELS_PER_STEP_CAPTURED * self.pp_size
            if graph_hit
            else KERNELS_PER_LAYER_EAGER * self.layers_local
        )
        t_overhead = num_kernels * self.device.launch_overhead

        duration = max(t_compute, t_memory) + t_comm + t_overhead

        # Seeded, so a run with jitter is still reproducible (R19.2). Clamped at zero
        # because a large sigma could otherwise draw a negative multiplier and run
        # the clock backwards.
        jitter_factor = 1.0
        if self.jitter_sigma > 0.0 and rng is not None:
            jitter_factor = max(0.0, 1.0 + float(rng.normal(0.0, self.jitter_sigma)))
            duration *= jitter_factor

        return StepCost(
            duration=duration,
            compute_seconds=t_compute,
            memory_seconds=t_memory,
            comm_seconds=t_comm,
            overhead_seconds=t_overhead,
            jitter_factor=jitter_factor,
            flops=flops,
            bytes_moved=bytes_moved,
        )

    def weight_load_seconds(self, weight_bytes: int) -> float:
        return weight_bytes / self.device.load_bandwidth

    def graph_capture_seconds(self, num_shapes: int) -> float:
        """R8.4. Capture replays each shape a few times; cost scales with depth."""
        if self.enforce_eager:
            return 0.0
        return num_shapes * self.layers_local * self.device.launch_overhead * 50


def build_cost_model(
    profile_name: str,
    model: ModelCard,
    device: DeviceCard,
    *,
    dtype: str,
    kv_cache_dtype: str | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
    jitter_sigma: float = 0.0,
    enforce_eager: bool = False,
) -> CostModel:
    """Construct the model named by `SimConfig.cost_model_profile` (R1.3)."""
    if profile_name == "constant":
        return ConstantCostModel()
    if profile_name == "roofline":
        return RooflineCostModel(
            model,
            device,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            tp_size=tp_size,
            pp_size=pp_size,
            jitter_sigma=jitter_sigma,
            enforce_eager=enforce_eager,
        )
    raise ValueError(
        f"unknown cost_model_profile {profile_name!r}; expected 'constant' or "
        f"'roofline'"
    )
