"""The platform abstraction -- the simulation boundary's selection mechanism.

Upstream: vllm/platforms/interface.py
Tier: B

B2: the simulator is selected exactly the way an out-of-tree hardware backend is
selected upstream. `current_platform` resolves to `SimPlatform`, which supplies the
worker class, the attention backend class, and the device communicator. Nothing above
the boundary branches on whether it is simulated -- it asks the platform, and the
platform happens to answer with simulated classes.

The upstream `PlatformEnum` members are reproduced verbatim, and `SimPlatform` uses
`OOT`. It is shipped in-tree with pvllm, but from vLLM's point of view a simulated
device *is* an out-of-tree backend, and using `OOT` keeps the seam identical to the
one a hardware vendor would use.

Thinned relative to upstream: everything torch-typed (`supported_dtypes` as
`torch.dtype`, `inference_mode`, `simple_compile_backend`, the inductor pass key) is
gone, since there is no torch. Dtypes are carried as strings.
"""

from __future__ import annotations

import enum
import platform
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar

from pvllm.timebase import Clock
from pvllm.tracing import TraceSink

if TYPE_CHECKING:
    from pvllm.config import VllmConfig


@cache
def in_wsl() -> bool:
    return "microsoft" in " ".join(platform.uname()).lower()


class PlatformEnum(enum.Enum):
    """Enumeration of supported hardware platforms."""

    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    POWERPC = enum.auto()
    RISCV = enum.auto()
    S390X = enum.auto()
    OTHER = enum.auto()
    UNKNOWN = enum.auto()


class Platform:
    """Base platform.

    A backend supplies its worker, attention backend, and device communicator by
    overriding the `get_*_cls` hooks, and adjusts the resolved config in
    `check_and_update_config`.
    """

    _enum: PlatformEnum
    device_name: str
    device_type: str

    # Empty string means the device does not support Ray. pretending-vllm never
    # will (NG5), but the attribute exists so the config surface matches.
    ray_device_key: str = ""

    #: The env var that controls device visibility, e.g. CUDA_VISIBLE_DEVICES.
    device_control_env_var: str = "PVLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"

    #: Dtype names, as strings. The first is the default fallback for "auto".
    supported_dtypes: ClassVar[list[str]] = ["bfloat16", "float16", "float32"]

    supported_quantization: ClassVar[list[str]] = []

    additional_env_vars: ClassVar[list[str]] = []

    dist_backend: str = ""

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    def is_unspecified(self) -> bool:
        return self._enum == PlatformEnum.UNSPECIFIED

    # --- device introspection ---------------------------------------------

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Total device memory in bytes."""
        raise NotImplementedError

    @classmethod
    def get_device_count(cls) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        raise NotImplementedError

    # --- class supply hooks -----------------------------------------------

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        """Adjust the resolved config for this backend.

        This is where `parallel_config.worker_cls` is filled in from `"auto"`, which
        is the hinge the whole simulation boundary turns on.
        """
        return None

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
        """Fully-qualified name of the attention backend class."""
        raise NotImplementedError

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Fully-qualified name of the device communicator class."""
        raise NotImplementedError

    # --- runtime services --------------------------------------------------
    #
    # No upstream counterpart: upstream reads wall time and needs no seeded RNG.
    # They are here because the engine core needs both (R19.1) and must not know a
    # simulator exists (B1) -- so the platform supplies them, exactly as it supplies
    # the worker class.

    @classmethod
    def build_clock(cls, mode: str, time_scale: float = 1.0) -> Clock:
        """The engine's timebase."""
        raise NotImplementedError

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
        """Where engine events are recorded (R19.3). `None` disables tracing."""
        raise NotImplementedError

    @classmethod
    def build_structured_output_backend(
        cls, vllm_config: Any, *, tokenizer: Any, vocab_size: int
    ) -> Any:
        """The grammar backend (R15). Simulator-supplied, like the clock."""
        raise NotImplementedError

    @classmethod
    def get_punica_wrapper(cls) -> str:
        """Fully-qualified name of the LoRA punica wrapper class (P3)."""
        raise NotImplementedError(
            "LoRA has no counterpart on this platform (requirement R16)"
        )

    # --- misc --------------------------------------------------------------

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64", "i386", "i686"):
            return CpuArchEnum.X86
        if machine.startswith("arm") or machine.startswith("aarch"):
            return CpuArchEnum.ARM
        if machine.startswith("ppc"):
            return CpuArchEnum.POWERPC
        if machine.startswith("riscv"):
            return CpuArchEnum.RISCV
        if machine.startswith("s390"):
            return CpuArchEnum.S390X
        return CpuArchEnum.OTHER

    @classmethod
    def __getattr__(cls, name: str) -> Any:
        raise AttributeError(
            f"{cls.__name__} has no attribute {name!r}. If this is a hook an "
            f"upstream backend provides, it has not been ported yet."
        )


class UnspecifiedPlatform(Platform):
    """Resolved when no platform plugin activates."""

    _enum = PlatformEnum.UNSPECIFIED
    device_name = "unspecified"
    device_type = ""
