"""The executor interface.

Upstream: vllm/v1/executor/abstract.py
Tier: B

R7.1. The executor is what the engine core talks to; it hides how many workers there
are and where they run. `UniProcExecutor` is the only implementation until the
multi-worker executors (tensor and pipeline parallelism) land in M4. The
multiprocess *engine core* (R4.2) is a different axis and already exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pvllm.config import VllmConfig
from pvllm.timebase import Clock
from pvllm.v1.core.sched.output import SchedulerOutput
from pvllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from pvllm.v1.outputs import ModelRunnerOutput


class Executor(ABC):
    """Runs the model across one or more workers."""

    def __init__(self, vllm_config: VllmConfig, clock: Clock) -> None:
        self.vllm_config = vllm_config
        self.clock = clock
        self._init_executor()

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type[Executor]:
        """Pick the executor for a config.

        Upstream dispatches on `distributed_executor_backend`. Only the in-process
        path exists, and asking for another names what is missing rather than
        silently running single-process -- which would report throughput for a
        parallelism that was never applied.
        """
        backend = vllm_config.parallel_config.distributed_executor_backend
        if backend in (None, "uni"):
            from pvllm.v1.executor.uniproc_executor import UniProcExecutor

            return UniProcExecutor
        raise NotImplementedError(
            f"distributed_executor_backend={backend!r} is not implemented. The "
            f"multi-worker executors (tensor and pipeline parallelism, R14) land "
            f"in M4; only the in-process executor exists. Note that the "
            f"multiprocess engine core (R4.2) is a separate axis and does exist -- "
            f"see PVLLM_ENABLE_V1_MULTIPROCESSING."
        )

    @abstractmethod
    def _init_executor(self) -> None: ...

    @abstractmethod
    def collective_rpc(
        self,
        method: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Call a method on every worker and collect the results."""

    @abstractmethod
    def determine_available_memory(self) -> list[int]:
        """KV pool bytes available on each worker. R10.3."""

    @abstractmethod
    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        """Each worker's KV cache spec, one entry per layer."""

    @abstractmethod
    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """Hand each worker its resolved KV layout."""

    @abstractmethod
    def compile_or_warm_up_model(self) -> None:
        """Simulate graph capture and warm-up. R8.4."""

    def execute_dummy_batch(self) -> float:
        """R13.4. A forward pass over nothing, to keep an EP collective whole.

        Concrete rather than abstract, and refusing: only the executors that model a
        device can charge for one, and an executor that cannot must say so rather than
        return a plausible zero -- a zero here would report an idle expert-parallel
        replica as free, which is the whole thing the dummy step exists to price.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot execute a dummy batch, which expert "
            f"parallelism needs to keep its collective whole (requirement R13.4)"
        )

    async def execute_dummy_batch_async(self) -> float:
        """As `execute_dummy_batch`, yielding while the modeled time passes."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot execute a dummy batch, which expert "
            f"parallelism needs to keep its collective whole (requirement R13.4)"
        )

    @abstractmethod
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Run one step. The simulation boundary."""

    @abstractmethod
    async def execute_model_async(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """Run one step, yielding to the event loop while modeled time passes.

        Abstract rather than a default that calls `execute_model`: a default would
        let an executor silently block the loop under a real clock, which is the
        exact failure this exists to prevent, and it would look correct in every
        virtual-clock test.
        """

    @abstractmethod
    def check_health(self) -> None: ...

    def shutdown(self) -> None:
        return None
