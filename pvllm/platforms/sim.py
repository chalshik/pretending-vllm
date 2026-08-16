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
        cls._align_hybrid_block_size(vllm_config)

        if vllm_config.use_v2_model_runner is False:
            raise NotImplementedError(
                "PVLLM_USE_V2_MODEL_RUNNER=0 requests the legacy V1 model runner "
                "(vllm/v1/worker/gpu_model_runner.py). pretending-vllm mirrors the "
                "V2 runner (vllm/v1/worker/gpu/model_runner.py), which is upstream's "
                "default at the pinned version. See D6 in UPSTREAM.md."
            )

    @classmethod
    def _align_hybrid_block_size(cls, vllm_config: VllmConfig) -> None:
        """Make an attention page big enough to hold a Mamba state. R6.7.

        A state-space layer's page is a fixed recurrent state; an attention layer's is
        `block_size` tokens of KV. One pool cannot hold both sizes, and the state
        cannot shrink -- so upstream grows the *attention* block size until its page
        covers the state, then pads the state page up to match exactly.

        This runs here, at the same seam upstream uses, because it has to happen before
        any spec is built: `get_kv_cache_spec` reads `cache_config.block_size`, and the
        engine core resolves the pool from the specs it returns.

        The consequence is much larger than the padding it saves. A Nemotron-H-class
        model moves from a 16-token block to roughly 1040, which changes how many
        blocks a request holds (C2), how coarse the prefix cache is, and -- because a
        block hash is computed over `block_size` tokens -- every block hash value (C3).
        A run that kept `block_size` at 16 would be wrong on all three even with the
        state bytes right.
        """
        from pvllm.sim.model_db import DTYPE_BYTES, load_model_card

        # Tolerant of a partial config: `check_and_update_config` is the platform's
        # hook and is called with whatever the caller has built, which in tests is
        # sometimes only the parallel config. Nothing to align without a model.
        model_config = getattr(vllm_config, "model_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)
        if model_config is None or cache_config is None:
            return
        sim_config = getattr(vllm_config, "sim_config", None)
        try:
            card = load_model_card(
                (sim_config.model_card if sim_config else None) or model_config.model
            )
        except (KeyError, FileNotFoundError):
            return
        if not card.is_state_space:
            return

        tp_size = vllm_config.parallel_config.tensor_parallel_size
        state_bytes = card.mamba_state_bytes_per_layer(tp_size)

        kv_dtype = cache_config.resolved_cache_dtype or model_config.resolved_dtype
        attention_bytes_per_token = (
            2
            * max(1, card.num_key_value_heads // tp_size)
            * card.head_dim
            * DTYPE_BYTES[kv_dtype]
        )

        # Upstream's formula, rounded to the alignment the attention kernels want.
        alignment = max(16, cache_config.block_size)
        aligned = alignment * -(-state_bytes // (alignment * attention_bytes_per_token))
        if aligned > cache_config.block_size:
            logger.info(
                "State-space model: block_size %d -> %d so an attention page "
                "(%d B/token) covers one layer's %d B recurrent state (R6.7). This "
                "changes block counts and every block hash value.",
                cache_config.block_size,
                aligned,
                attention_bytes_per_token,
                state_bytes,
            )
            cache_config.block_size = aligned
        cache_config.mamba_page_size_padded = (
            cache_config.block_size * attention_bytes_per_token
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
