"""The in-process executor.

Upstream: vllm/v1/executor/uniproc_executor.py
Tier: B

One worker, in this process. The default for tests and the only implementation until
the multiprocess engine core lands in M3 (D2).

The worker class is *not* hardcoded: it comes from `parallel_config.worker_cls`, which
`SimPlatform.check_and_update_config` filled in from `"auto"` (B2). That indirection is
the simulation boundary's selection mechanism, and short-circuiting it here -- even
though there is only one possible answer today -- would remove the seam the whole
design rests on.
"""

from __future__ import annotations

from typing import Any

from pvllm.logger import init_logger
from pvllm.utils import resolve_obj_by_qualname
from pvllm.v1.core.sched.output import SchedulerOutput
from pvllm.v1.executor.abstract import Executor
from pvllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from pvllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


class UniProcExecutor(Executor):
    """A single worker, called directly."""

    def _init_executor(self) -> None:
        worker_cls_qualname = self.vllm_config.parallel_config.worker_cls
        if worker_cls_qualname == "auto":
            raise RuntimeError(
                "parallel_config.worker_cls is still 'auto'; the platform's "
                "check_and_update_config should have resolved it (B2)"
            )
        worker_cls = resolve_obj_by_qualname(worker_cls_qualname)
        self.driver_worker = worker_cls(
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=0,
            clock=self.clock,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def collective_rpc(
        self,
        method: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        return [getattr(self.driver_worker, method)(*args, **(kwargs or {}))]

    def determine_available_memory(self) -> list[int]:
        return [self.driver_worker.determine_available_memory()]

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        return [self.driver_worker.get_kv_cache_spec()]

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        self.driver_worker.initialize_cache(kv_cache_configs[0])

    def compile_or_warm_up_model(self) -> None:
        self.driver_worker.compile_or_warm_up_model()

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        # The worker is resolved by qualname at runtime (B2), so it is untyped here;
        # narrow explicitly rather than letting Any leak into the engine core.
        output: ModelRunnerOutput = self.driver_worker.execute_model(scheduler_output)
        return output

    def check_health(self) -> None:
        self.driver_worker.check_health()

    def __repr__(self) -> str:
        return f"UniProcExecutor(worker={self.driver_worker})"
