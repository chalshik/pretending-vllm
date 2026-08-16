"""The composite config root.

Upstream: vllm/config/vllm.py
Tier: C

R1.1: `VllmConfig` composes every sub-config, and `__post_init__` is where the
platform gets to adjust the resolved result -- which is how `worker_cls` becomes a
simulated worker without anything above the boundary knowing (B2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pvllm import envs
from pvllm.config.cache import CacheConfig
from pvllm.config.device import DeviceConfig
from pvllm.config.kv_transfer import KVTransferConfig
from pvllm.config.load import LoadConfig
from pvllm.config.lora import LoRAConfig
from pvllm.config.model import ModelConfig
from pvllm.config.observability import ObservabilityConfig
from pvllm.config.parallel import ParallelConfig
from pvllm.config.scheduler import SchedulerConfig
from pvllm.config.speculative import SpeculativeConfig
from pvllm.config.structured_outputs import StructuredOutputsConfig
from pvllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class VllmConfig:
    """Every configuration the engine needs, resolved."""

    model_config: ModelConfig = field(default_factory=ModelConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig | None = None
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    load_config: LoadConfig = field(default_factory=LoadConfig)
    lora_config: LoRAConfig | None = None
    #: R16.1. `name=path` adapter specs from `--lora-modules`, resolved by the
    #: serving layer -- which is the only place that knows about model names.
    lora_modules: list[str] | None = None
    speculative_config: SpeculativeConfig | None = None
    structured_outputs_config: StructuredOutputsConfig = field(
        default_factory=StructuredOutputsConfig
    )
    observability_config: ObservabilityConfig = field(
        default_factory=ObservabilityConfig
    )
    kv_transfer_config: KVTransferConfig | None = None
    instance_id: str = ""

    def __post_init__(self) -> None:
        # The scheduler's max_model_len must agree with the model's, and the model is
        # what resolves it, so the scheduler config is built last when not supplied.
        if self.scheduler_config is None:
            self.scheduler_config = SchedulerConfig(
                max_model_len=self.model_config.max_model_len or 8192
            )
        elif self.scheduler_config.max_model_len != self.model_config.max_model_len:
            self.scheduler_config.max_model_len = (
                self.model_config.max_model_len or 8192
            )

        # B2: the platform gets the last word on the resolved config. This is where
        # worker_cls stops being "auto".
        from pvllm.platforms import current_platform

        current_platform.check_and_update_config(self)

        # R10.6: a max_model_len whose KV cannot fit even one request is a startup
        # error, not a request-time one. The full check needs the memory model and
        # happens in determine_available_memory; this catches the arithmetic case.
        #
        # *After* the platform hook, because the hook is what can move the block
        # size: a state-space model's alignment takes it from 16 to about 1040, and
        # reading `block_size` before that meant this warning could never fire for
        # the only models whose block size grows -- which are exactly the models
        # whose block size can overtake a reasonable `max_model_len`.
        block_size = self.cache_config.block_size
        if (
            self.model_config.max_model_len
            and self.model_config.max_model_len < block_size
        ):
            logger.warning(
                "max_model_len (%d) is smaller than block_size (%d); every request "
                "will occupy a full block.",
                self.model_config.max_model_len,
                block_size,
            )

    @property
    def sim_config(self):  # type: ignore[no-untyped-def]
        """Shorthand for the simulator knobs, which live under DeviceConfig (R1.3)."""
        return self.device_config.sim_config

    @property
    def use_v2_model_runner(self) -> bool:
        """Whether to use the V2 model runner shape.

        Upstream resolves this from the env var, then from architecture defaults
        (`_is_default_v2_model_runner_model`). At the pin it lands on True for any
        dense, non-MoE, non-hybrid generate model. pretending-vllm mirrors only the V2
        shape (D6), so this is True unless explicitly overridden -- and
        `SimPlatform.check_and_update_config` rejects an explicit False rather than
        silently running V2 while reporting V1.
        """
        override = envs.PVLLM_USE_V2_MODEL_RUNNER
        if override is not None:
            return bool(override)
        return True

    @property
    def max_concurrent_batches(self) -> int:
        """Batches in flight at once. One until pipeline parallelism lands (M4)."""
        return self.parallel_config.pipeline_parallel_size

    def __str__(self) -> str:
        model = self.model_config
        cache = self.cache_config
        sim = self.sim_config
        return (
            f"model={model.model!r}, card={model.hf_config.name!r}, "
            f"dtype={model.resolved_dtype}, max_model_len={model.max_model_len}, "
            f"block_size={cache.block_size}, "
            f"gpu_memory_utilization={cache.gpu_memory_utilization}, "
            f"enable_prefix_caching={cache.enable_prefix_caching}, "
            f"device={sim.device_card!r}, clock={sim.clock_mode}, "
            f"cost_model={sim.cost_model_profile}, seed={sim.seed}"
        )
