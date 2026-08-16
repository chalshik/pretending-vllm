"""The simulated worker.

Upstream: vllm/v1/worker/gpu_worker.py (counterpart, not a port)
Tier: B

R7.2: mirrors `GPUWorker`'s lifecycle exactly -- `init_device`, `load_model`,
`determine_available_memory`, `initialize_cache`, `compile_or_warm_up_model`,
`execute_model`. The class is named `Worker` because that is what
`SimPlatform.check_and_update_config` resolves `worker_cls` to, and what upstream's
executor expects to find.

This is the last object above the simulation boundary. It owns the `SimDevice`, the
`SimModel`, and the runner; everything it hands upward is shaped exactly as a real
worker's output.

`determine_available_memory` is where the memory model runs (R10.3), at the same point
upstream runs its profiling forward pass.
"""

from __future__ import annotations

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.sim.cost_model import build_cost_model
from pvllm.sim.device import SimDevice
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.memory import MemoryProfile, compute_memory_profile
from pvllm.sim.model import SimModel
from pvllm.sim.model_db import load_model_card
from pvllm.sim.rng import RngFactory
from pvllm.sim.weights import StartupTimeline
from pvllm.timebase import Clock
from pvllm.v1.core.kv_cache_utils import get_kv_cache_groups
from pvllm.v1.core.sched.output import SchedulerOutput
from pvllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from pvllm.v1.outputs import ModelRunnerOutput
from pvllm.v1.worker.gpu.attn_utils import get_kv_cache_spec
from pvllm.v1.worker.gpu.model_runner import SimModelRunner

logger = init_logger(__name__)


class Worker:
    """One simulated device's worker."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int = 0,
        rank: int = 0,
        clock: Clock | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.sim_config = vllm_config.sim_config
        self.local_rank = local_rank
        self.rank = rank

        # The clock is supplied by the engine core, which owns it (R19.1). A worker
        # that made its own would let time advance in two places.
        if clock is None:
            raise ValueError(
                "Worker requires the engine core's clock; it must not create one "
                "(requirement R19.1)"
            )
        self.clock = clock
        # Seeded here rather than passed down: B3 keeps randomness below the
        # boundary, and the worker is the first object that is below it.
        self.rng_factory = RngFactory(self.sim_config.seed)

        self.device: SimDevice | None = None
        self.model_runner: SimModelRunner | None = None
        self.memory_profile: MemoryProfile | None = None
        self.startup = StartupTimeline()

    # --- lifecycle (R7.2) ----------------------------------------------------

    def init_device(self) -> None:
        """Bring up the simulated device and its cost model."""
        model_card = load_model_card(
            self.sim_config.model_card or self.model_config.model
        )
        device_card = load_device_card(self.sim_config.device_card)

        cost_model = build_cost_model(
            self.sim_config.cost_model_profile,
            model_card,
            device_card,
            dtype=self.model_config.resolved_dtype,
            kv_cache_dtype=self.cache_config.resolved_cache_dtype,
            tp_size=self.parallel_config.tensor_parallel_size,
            pp_size=self.parallel_config.pipeline_parallel_size,
            ep_size=self.parallel_config.expert_parallel_size,
            dp_size=self.parallel_config.data_parallel_size,
            jitter_sigma=self.sim_config.jitter_sigma,
            enforce_eager=self.model_config.enforce_eager,
        )
        self.device = SimDevice(
            card=device_card,
            clock=self.clock,
            cost_model=cost_model,
            # Jitter draws from a named engine-level stream, not a per-request one:
            # it is a property of the step, not of any request (R19.2).
            rng=self.rng_factory.stream("jitter")
            if self.sim_config.jitter_sigma
            else None,
            device_id=self.local_rank,
        )
        self.model_card = model_card
        self.device_card = device_card

        # R13.1. vLLM refuses a tensor_parallel_size that does not divide the head
        # counts, and so must this: sharding 16 heads over 3 ranks drops a head, and
        # a capacity answer for a configuration that cannot start is worse than no
        # answer. Checked here rather than in ParallelConfig because only the model
        # card knows the head counts.
        tp_size = self.parallel_config.tensor_parallel_size
        if tp_size > 1:
            for label, count in (
                ("attention heads", model_card.num_attention_heads),
                ("KV heads", model_card.num_key_value_heads),
            ):
                if count % tp_size:
                    raise ValueError(
                        f"tensor_parallel_size={tp_size} does not divide the model's "
                        f"{count} {label}. vLLM refuses this configuration at "
                        f"startup, so reporting a capacity number for it would "
                        f"describe an engine that cannot run."
                    )
        # R13.4. Upstream's `_verify_with_expert_parallelism`: "Number of experts in
        # the model must be greater than 0 when expert parallelism is enabled".
        # Checked here rather than in ParallelConfig because only the model card knows
        # whether there are experts to spread.
        if self.parallel_config.enable_expert_parallel and not model_card.is_moe:
            raise ValueError(
                f"enable_expert_parallel needs a mixture-of-experts model, and "
                f"{model_card.name!r} declares num_experts=0. There is nothing to "
                f"spread across devices; use --tensor-parallel-size to shard a dense "
                f"model."
            )

        pp_size = self.parallel_config.pipeline_parallel_size
        if pp_size > model_card.num_hidden_layers:
            raise ValueError(
                f"pipeline_parallel_size={pp_size} exceeds the model's "
                f"{model_card.num_hidden_layers} layers; a stage with no layers "
                f"holds nothing and computes nothing."
            )

    def load_model(self) -> None:
        """ "Load" the weights, and spend the modeled time doing it. R10.4."""
        assert self.device is not None
        weight_bytes = self._weight_bytes()
        self.startup.load_weights_seconds = self.device.load_weights(weight_bytes)

        sim_model = SimModel(
            model=self.model_card,
            rng_factory=self.rng_factory,
            output_length_policy=self.sim_config.output_length_policy,
            content_policy=self.sim_config.content_policy,
            output_length_fixed=self.sim_config.output_length_fixed,
            output_length_range=self.sim_config.output_length_range,
            output_length_lognormal=self.sim_config.output_length_lognormal,
            spec_acceptance_rate=self.sim_config.spec_acceptance_rate,
        )
        self.model_runner = SimModelRunner(self.vllm_config, self.device, sim_model)
        logger.info(
            "Model weights loaded in %.2f seconds [modeled]",
            self.startup.load_weights_seconds,
        )

    def determine_available_memory(self) -> int:
        """Run the memory model and return the KV pool size in bytes. R10.3.

        At the same point upstream runs its profiling forward pass, and the profiling
        run's modeled cost is charged to the clock here too, so the startup timeline
        (R10.4) reflects it.
        """
        assert self.device is not None
        assert self.vllm_config.scheduler_config is not None
        scheduler_config = self.vllm_config.scheduler_config
        assert scheduler_config.max_num_batched_tokens is not None

        from pvllm.sim.cost_model import StepProfile

        # The profiling forward pass itself: one full-budget step, whose cost is
        # real time on the startup timeline.
        before = self.clock.elapsed
        self.device.execute(
            StepProfile(
                num_tokens=scheduler_config.max_num_batched_tokens,
                num_reqs=1,
                query_lens=[scheduler_config.max_num_batched_tokens],
                seq_lens=[scheduler_config.max_num_batched_tokens],
            )
        )
        self.startup.profile_run_seconds = self.clock.elapsed - before

        self.memory_profile = compute_memory_profile(
            self.model_card,
            self.device_card,
            dtype=self.model_config.resolved_dtype,
            kv_cache_dtype=self.cache_config.resolved_cache_dtype,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            max_model_len=scheduler_config.max_model_len,
            max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
            max_num_seqs=scheduler_config.max_num_seqs,
            tp_size=self.parallel_config.tensor_parallel_size,
            pp_size=self.parallel_config.pipeline_parallel_size,
            num_gpu_blocks_override=self.cache_config.num_gpu_blocks_override,
            lora_bytes=self._lora_bytes(),
            sliding_window=self.cache_config.sliding_window,
            ep_size=self.parallel_config.expert_parallel_size,
            # R6.7. The same groups the engine core sizes the pool from, so the
            # startup line and the scheduler's pool cannot disagree.
            kv_cache_groups=get_kv_cache_groups(self.get_kv_cache_spec()),
        )
        logger.info("%s", self.memory_profile.summary())
        # The *block count* is what the profile resolved -- including any
        # num_gpu_blocks_override -- so report the bytes those blocks occupy rather
        # than the raw pool size. Returning the pool size would let the engine core
        # re-derive a different count and silently ignore the override.
        return (
            self.memory_profile.num_gpu_blocks * self.memory_profile.kv_bytes_per_block
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return get_kv_cache_spec(self.vllm_config)

    def initialize_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Claim the KV pool on the ledger and build the block tables."""
        assert self.device is not None and self.model_runner is not None
        assert self.memory_profile is not None

        before = self.clock.elapsed
        self.device.apply_memory_profile(self.memory_profile)
        self.model_runner.initialize_kv_cache(kv_cache_config)
        self.startup.kv_cache_seconds = self.clock.elapsed - before

        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.cache_config.kv_cache_size_tokens = (
            kv_cache_config.num_blocks * self.cache_config.block_size
        )
        self.cache_config.kv_cache_max_concurrency = self.memory_profile.max_concurrency

    def compile_or_warm_up_model(self) -> None:
        """Simulate graph capture. R8.4."""
        assert self.model_runner is not None
        self.startup.graph_capture_seconds = self.model_runner.capture_model()

        gib = 1 << 30
        kv_gib = (
            self.memory_profile.num_gpu_blocks
            * self.memory_profile.kv_bytes_per_block
            / gib
            if self.memory_profile
            else 0.0
        )
        logger.info("%s", self.startup.summary(kv_gib))

    # --- the step ------------------------------------------------------------

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """The simulation boundary. Section 4."""
        assert self.model_runner is not None
        return self.model_runner.execute_model(scheduler_output)

    async def execute_model_async(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """The same boundary crossing, yielding while modeled time passes.

        Upstream has no counterpart: on real hardware the forward pass is launched
        asynchronously and awaited by CUDA, so there is no interval during which
        Python holds the loop. Here the interval is the whole simulation, so it has
        to be awaited explicitly or a real-clock server stops serving during it.
        """
        assert self.model_runner is not None
        return await self.model_runner.execute_model_async(scheduler_output)

    def execute_dummy_batch(self) -> float:
        """R13.4. A forward pass over nothing, to keep an EP collective whole.

        Returns the modeled duration, which the device has already charged to the
        clock -- reported so the deployment can count what its idle replicas spent.
        """
        assert self.model_runner is not None
        return self.model_runner.execute_dummy_batch().duration

    def check_health(self) -> None:
        return None

    # --- helpers -------------------------------------------------------------

    def _lora_bytes(self) -> int:
        """Device memory the resident adapters occupy. R16.1. Zero without LoRA."""
        lora_config = self.vllm_config.lora_config
        if lora_config is None:
            return 0
        from pvllm.sim.memory import compute_lora_bytes

        assert self.model_card is not None
        return compute_lora_bytes(
            self.model_card,
            self.model_config.resolved_dtype,
            max_loras=lora_config.max_loras,
            max_lora_rank=lora_config.max_lora_rank,
            num_target_modules=len(lora_config.target_modules),
            tp_size=self.parallel_config.tensor_parallel_size,
            pp_size=self.parallel_config.pipeline_parallel_size,
        )

    def _weight_bytes(self) -> int:
        from pvllm.sim.memory import compute_weight_bytes

        return (
            compute_weight_bytes(
                self.model_card,
                self.model_config.resolved_dtype,
                self.parallel_config.tensor_parallel_size,
                ep_size=(
                    self.parallel_config.expert_parallel_size
                    if self.parallel_config.enable_expert_parallel
                    else None
                ),
            )
            // self.parallel_config.pipeline_parallel_size
        )

    def __repr__(self) -> str:
        return f"Worker(rank={self.rank}, device={self.device})"
