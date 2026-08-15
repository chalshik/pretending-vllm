"""How the frontend reaches the engine core.

Upstream: vllm/v1/engine/core_client.py
Tier: B

R4.2/D2: three implementations are intended -- in process, multiprocess synchronous,
and multiprocess asynchronous. Only the in-process one exists until M3.

The abstraction ships now anyway, because the *clock ownership* rule it implies
(R19.1) cannot be retrofitted. In process, a frontend that read `time.time()` would
appear to work; over a process boundary it would silently mix two timelines. Making
every timestamp come back across this interface from the start means the multiprocess
implementation is a transport change rather than a redesign.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pvllm.config import VllmConfig
from pvllm.v1.engine import EngineCoreOutputs, EngineCoreRequest
from pvllm.v1.engine.core import EngineCore
from pvllm.v1.executor.abstract import Executor


class EngineCoreClient(ABC):
    """The frontend's handle on the engine core."""

    @staticmethod
    def make_client(
        vllm_config: VllmConfig,
        *,
        multiprocess_mode: bool = False,
        asyncio_mode: bool = False,
        executor_class: type[Executor] | None = None,
        log_stats: bool = True,
    ) -> EngineCoreClient:
        if multiprocess_mode:
            raise NotImplementedError(
                "the multiprocess engine core (requirement R4.2) lands in M3. It uses "
                "real OS processes and ZeroMQ so that serialization cost and "
                "backpressure are real rather than modeled, which is why it is not "
                "stubbed here."
            )
        return InprocClient(
            vllm_config, executor_class=executor_class, log_stats=log_stats
        )

    @abstractmethod
    def add_request(self, request: EngineCoreRequest) -> None: ...

    @abstractmethod
    def abort_requests(self, request_ids: list[str]) -> None: ...

    @abstractmethod
    def get_output(self) -> dict[int, EngineCoreOutputs]: ...

    @abstractmethod
    def has_requests(self) -> bool:
        """Whether another step is worth taking."""

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Requests still waiting or running."""

    @abstractmethod
    def shutdown(self) -> None: ...


class InprocClient(EngineCoreClient):
    """The engine core in this process, called directly.

    The default for tests (D2): no IPC, no serialization, and a step is a plain
    function call, which keeps a failing test's traceback pointing at the bug.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor] | None = None,
        log_stats: bool = True,
    ) -> None:
        self.engine_core = EngineCore(
            vllm_config, executor_class=executor_class, log_stats=log_stats
        )

    def add_request(self, request: EngineCoreRequest) -> None:
        self.engine_core.add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids:
            self.engine_core.abort_requests(request_ids)

    def get_output(self) -> dict[int, EngineCoreOutputs]:
        outputs, _ = self.engine_core.step()
        return outputs

    def has_requests(self) -> bool:
        """Whether another step is worth taking.

        Distinct from `get_num_unfinished_requests`: this stays True while finished
        request ids are still queued for delivery to the worker, because that
        cleanup rides on the next step (R5.8).
        """
        return self.engine_core.has_requests()

    def get_num_unfinished_requests(self) -> int:
        """Requests still waiting or running -- cleanup bookkeeping excluded."""
        return self.engine_core.get_num_unfinished_requests()

    def reset_prefix_cache(self) -> bool:
        return self.engine_core.reset_prefix_cache()

    def make_stats(self) -> dict[str, Any]:
        return self.engine_core.make_stats()

    @property
    def clock_time(self) -> float:
        """The engine's time. The only way a frontend may learn it (R19.1)."""
        return self.engine_core.clock.time()

    def shutdown(self) -> None:
        self.engine_core.shutdown()
