"""The simulated platform.

Upstream: vllm/platforms/cuda.py (counterpart, not a port)
Tier: D

This is the hinge. `check_and_update_config` fills in `worker_cls` from `"auto"`,
which is how a simulated worker gets in front of a real scheduler without any code
above the boundary knowing (B1). `CudaPlatform` does exactly the same thing with
`"vllm.v1.worker.gpu_worker.Worker"`.

Device facts (name, memory, count) come from the hardware card named by `SimConfig`,
not from probing hardware. That is the whole point: hardware becomes a JSON file, and
"does 70B at 128k context fit at gpu_memory_utilization 0.92 on eight devices" is
answerable without owning the hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pvllm.logger import init_logger
from pvllm.platforms.interface import Platform, PlatformEnum
from pvllm.timebase import Clock
from pvllm.tracing import TraceSink

if TYPE_CHECKING:
    from pvllm.config import VllmConfig

logger = init_logger(__name__)


class SimPlatform(Platform):
    """A device that does not exist, described by a JSON card."""

    _enum = PlatformEnum.OOT
    device_name: str = "sim"
    device_type: str = "sim"
    dist_backend: str = "sim"

    device_control_env_var: str = "PVLLM_VISIBLE_DEVICES"

    # No quantization is modeled yet; dtype affects memory and the cost model only.
    supported_dtypes: ClassVar[list[str]] = ["bfloat16", "float16", "float32"]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        from pvllm.sim.hardware_db import get_active_device_card

        return get_active_device_card().name

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        from pvllm.sim.hardware_db import get_active_device_card

        return get_active_device_card().memory_bytes

    @classmethod
    def get_device_count(cls) -> int:
        from pvllm.sim.hardware_db import get_active_device_card

        return get_active_device_card().num_devices

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return True

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        parallel_config = vllm_config.parallel_config

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "pvllm.v1.worker.sim_worker.Worker"

        # F1/D6: upstream defaults to the V2 model runner for dense generate models,
        # and pretending-vllm mirrors only the V2 shape. Setting the flag to 0 asks
        # for a runner that does not exist here, so say so rather than silently
        # running V2 and reporting V1.
        if vllm_config.use_v2_model_runner is False:
            raise NotImplementedError(
                "PVLLM_USE_V2_MODEL_RUNNER=0 requests the legacy V1 model runner "
                "(vllm/v1/worker/gpu_model_runner.py). pretending-vllm mirrors the "
                "V2 runner (vllm/v1/worker/gpu/model_runner.py), which is upstream's "
                "default at the pinned version. See D6 in UPSTREAM.md."
            )

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
        dtype: str,
        kv_cache_dtype: str | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
    ) -> str:
        if selected_backend not in (None, "SIM", "SIM_ATTN"):
            raise NotImplementedError(
                f"attention backend {selected_backend!r} has no simulated "
                f"counterpart; the simulator provides SIM_ATTN only"
            )
        return "pvllm.v1.attention.backends.sim_attn.SimAttentionBackend"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "pvllm.distributed.sim_communicator.SimCommunicator"

    # --- runtime services --------------------------------------------------

    @classmethod
    def build_clock(cls, mode: str, time_scale: float = 1.0) -> Clock:
        """R19.1. The engine core asks for this; only the platform knows it is
        simulated."""
        from pvllm.sim.clock import build_clock

        return build_clock(mode, time_scale=time_scale)  # type: ignore[arg-type]

    @classmethod
    def build_structured_output_backend(
        cls, vllm_config: Any, *, tokenizer: Any, vocab_size: int
    ) -> Any:
        """R15. Upstream resolves a compiled grammar engine here; the simulated one
        validates constraints and lets `SimModel` satisfy them."""
        from pvllm.sim.structured_output import SimStructuredOutputBackend

        return SimStructuredOutputBackend(
            vllm_config=vllm_config, tokenizer=tokenizer, vocab_size=vocab_size
        )

    @classmethod
    def build_kv_connector(cls, vllm_config: Any, role: Any) -> Any:
        """R17.1. Upstream resolves the configured connector class here too.

        The scheduler is Tier A and must not name a simulator module: upstream's
        `KVConnectorFactory` is the seam it goes through, and this is ours. Without
        it `v1/core/sched` imported `pvllm.sim` transitively -- a boundary crossing
        the purity check does not see, because the import is inside a function.
        """
        transfer = vllm_config.kv_transfer_config
        if transfer is None or transfer.kv_connector is None:
            return None
        from pvllm.distributed.kv_transfer.sim_connector import SimSharedStoreConnector

        connectors = {"SimSharedStoreConnector": SimSharedStoreConnector}
        connector_cls = connectors.get(transfer.kv_connector)
        if connector_cls is None:
            # `KVTransferConfig` refuses unknown names already; this is the second
            # gate, so adding a name there without wiring it here fails loudly.
            raise NotImplementedError(
                f"KV connector {transfer.kv_connector!r} is accepted by the config "
                f"but has no implementation here"
            )
        return connector_cls(vllm_config, role)

    @classmethod
    def build_trace_sink(
        cls,
        path: str | None,
        *,
        seed: int,
        clock_mode: str,
        upstream_version: str,
        config: dict[str, Any] | None = None,
    ) -> TraceSink:
        """R19.3."""
        from pvllm.sim.trace import NullTraceWriter, TraceWriter

        if path is None:
            return NullTraceWriter()
        return TraceWriter(
            path,
            seed=seed,
            clock_mode=clock_mode,
            upstream_version=upstream_version,
            config=config,
        )


def sim_platform_plugin() -> str | None:
    """Builtin platform plugin hook.

    Always activates: there is no hardware to detect, which is exactly why this
    package runs anywhere (NF3).
    """
    return "pvllm.platforms.sim.SimPlatform"
