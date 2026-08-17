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

#: R18.1. Parameters in the vision encoder, at ViT-L/14 scale -- roughly what
#: LLaVA-1.5 and Qwen-VL pair with a decoder, whatever the decoder's size. A
#: *count*, not a multiple of `hidden_size`: a vision tower does not grow with the
#: language model it is bolted to, and the previous form (12 x hidden_size, about
#: 49k parameters) modeled a 256-patch image at a tenth of a microsecond -- free,
#: against a 20 ms step. An image was documented as expensive and priced at nothing,
#: which is the one direction a cost model must not be wrong in when the question is
#: whether to cache encoder output at all.
#:
#: Uncalibrated like every other constant here (R9.5). Read it as "an image costs
#: roughly one short prefill", not as a measurement.
ENCODER_PARAMS = 300_000_000


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
    #: R18.1. Encoder embeddings computed this step. Zero for a text-only step, and
    #: for a multimodal one whose images were already cached -- which is the point
    #: of the encoder cache, and shows up here as a step that costs less.
    num_encoder_embeds: int = 0


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
    #: R18.1. The vision encoder's own pass. Its own field rather than folded into
    #: `compute_seconds`, because `bound_by` compares compute against memory and an
    #: encoder term added to the compute side flipped that verdict on any step
    #: carrying an image -- reporting a memory-bound decode step as compute-bound
    #: for a reason that has nothing to do with the decode.
    encoder_seconds: float = 0.0

    @property
    def is_compute_bound(self) -> bool:
        return self.compute_seconds >= self.memory_seconds

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "duration": self.duration,
            "compute_s": self.compute_seconds,
            "memory_s": self.memory_seconds,
            "comm_s": self.comm_seconds,
            "encoder_s": self.encoder_seconds,
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
        ep_size: int = 1,
        dp_size: int = 1,
        jitter_sigma: float = 0.0,
        enforce_eager: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.tp_size = tp_size
        self.pp_size = pp_size
        #: R13.4. `data_parallel_size * tensor_parallel_size` when expert parallelism
        #: is on, else 1. The experts divide by this instead of by `tp_size`.
        self.ep_size = ep_size
        #: Needed here only because the EP collective carries the *union* of the
        #: replicas' batches, which is the one place a replica's cost depends on how
        #: many other replicas there are.
        self.dp_size = dp_size
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
        #: Layers on the single simulated rank. A *step* traverses every stage, so
        #: `layers_local` above is right for latency -- but a per-device startup cost
        #: is not a traversal, and this is the count for those. `compute_memory_profile`
        #: and `SimWorker._weight_bytes` both divide the same way; graph capture did
        #: not, and was charged `pp_size` times too much until M9's review caught it.
        #: Ceiling, not floor: the busiest stage is the one that has to be buildable.
        self.layers_per_stage = max(1, -(-model.num_hidden_layers // pp_size))

        from pvllm.sim.memory import compute_weight_bytes

        # Every stage's weights are read once per step, so the traffic for a full
        # traversal is the whole (TP-sharded) weight set regardless of `pp_size`.
        expert_shard = ep_size if ep_size > 1 else None
        self.weight_bytes_local = compute_weight_bytes(
            model, dtype, tp_size, ep_size=expert_shard
        )
        # MoE reads only the routed experts per token, so the *active* count is what
        # the compute term uses -- which is why an MoE is far cheaper to run than its
        # parameter count suggests.
        #
        # R13.4. Expert parallelism does **not** reduce per-device MoE FLOPs, and the
        # arithmetic is worth writing down because dividing by `ep_size` here is the
        # obvious wrong move. Under TP each rank holds every expert sliced to `I/tp`
        # and runs `tokens * top_k` pairs through the slice: work = total/tp. Under EP
        # each rank holds `E/ep` whole experts and runs the `tokens_total * top_k /
        # ep` pairs that route to them at full width: work = total/ep -- but
        # `tokens_total` is the union across the `dp` replicas, so with `ep = dp * tp`
        # the two land on the same number. EP moves *where* the weights live, not how
        # much arithmetic each device does.
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
        if self.ep_size > 1 and self.model.is_moe and self.dp_size > 1:
            # R13.4. Under expert parallelism the MoE layer's all-reduce is *replaced*
            # by a dispatch/combine pair -- an all-gatherv and a reduce-scatterv over
            # the EP group on upstream's default `allgather_reducescatter` backend.
            # A ring all-reduce is a reduce-scatter followed by an all-gather, so the
            # byte volume is the same collective by another name: EP does not make
            # the MoE layer's communication bigger, it makes it *wider*.
            #
            # What changes is the token set. The EP group spans the data-parallel
            # replicas, so the collective carries the union of their batches rather
            # than one replica's -- `dp_size` times the tokens. That is the whole cost
            # of EP, and it is why `--data-parallel-size 8 --enable-expert-parallel`
            # is a different proposition from `--tensor-parallel-size 8`.
            #
            # At `dp_size == 1` there is no all-to-all at all: upstream's
            # `use_all2all_kernels` requires `dp_size > 1` and the layer issues the
            # same single all-reduce it would without EP. So this term is skipped, and
            # a TP-only EP run reports the same duration as the TP run -- which is
            # what upstream does, and is asserted in the tests.
            # How much of the MoE layer's collective is *not* already charged. The
            # existing tensor-parallel term bills two all-reduces per layer, one of
            # which is the MLP's -- so at tp > 1 only the extra `dp - 1` token-sets
            # are new. At tp == 1 that term is zero (a single device runs no
            # all-reduce), so the whole `dp` token-sets are new. Charging `dp - 1`
            # unconditionally billed half the collective at dp=2 and made every
            # sweep comparing EP against TP come out systematically cheap for EP --
            # which is precisely the comparison EP exists to inform.
            extra_tokens = tokens * (
                self.dp_size - 1 if self.tp_size > 1 else self.dp_size
            )
            t_comm += (
                extra_tokens * self.model.hidden_size * self.dtype_bytes / link
            ) * self.layers_local

        # --- encoder ---------------------------------------------------------
        # R18.1. A vision encoder is a separate forward pass over the image patches,
        # and it is compute-bound: a ViT over a few hundred patches does far more
        # FLOPs per token than a decoder step does. Charged as its own term rather
        # than folded into compute so `/debug/cost_model` can show a step that was
        # slow because of an image rather than because of the batch.
        t_encoder = 0.0
        if profile.num_encoder_embeds:
            encoder_flops = 2.0 * ENCODER_PARAMS * profile.num_encoder_embeds
            t_encoder = encoder_flops / (self.device.mfu * self.peak_flops)
            flops += encoder_flops

        # --- launch overhead -------------------------------------------------
        graph_hit = profile.is_graph_hit and not self.enforce_eager
        num_kernels = (
            KERNELS_PER_STEP_CAPTURED * self.pp_size
            if graph_hit
            else KERNELS_PER_LAYER_EAGER * self.layers_local
        )
        t_overhead = num_kernels * self.device.launch_overhead

        duration = max(t_compute, t_memory) + t_comm + t_overhead + t_encoder

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
            encoder_seconds=t_encoder,
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
        """R8.4. Capture replays each shape a few times; cost scales with depth.

        Depth *on this device*, so `layers_per_stage`. Capture is startup work done by
        one rank over the layers that rank holds -- not a traversal of the pipeline
        the way a step is. Using the whole-model count charged an 8-stage deployment
        eight times its real capture time, and startup time is what an autoscaler's
        cold-start budget is set from.
        """
        if self.enforce_eager:
            return 0.0
        return num_shapes * self.layers_per_stage * self.device.launch_overhead * 50


def build_cost_model(
    profile_name: str,
    model: ModelCard,
    device: DeviceCard,
    *,
    dtype: str,
    kv_cache_dtype: str | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
    ep_size: int = 1,
    dp_size: int = 1,
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
            ep_size=ep_size,
            dp_size=dp_size,
            jitter_sigma=jitter_sigma,
            enforce_eager=enforce_eager,
        )
    raise ValueError(
        f"unknown cost_model_profile {profile_name!r}; expected 'constant' or "
        f"'roofline'"
    )
