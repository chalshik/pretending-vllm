"""How the frontend reaches the engine core.

Upstream: vllm/v1/engine/core_client.py
Tier: B

R4.2/D2: three implementations, as upstream has -- in process (`InprocClient` here),
multiprocess synchronous and multiprocess asynchronous (`core_client_mp.py`).

The interface was shaped around *clock ownership* (R19.1) from the first commit,
before either multiprocess client existed, because that part cannot be retrofitted.
In process, a frontend that read `time.time()` would appear to work; over a process
boundary it would silently mix two timelines. Making every timestamp come back across
this interface from the start is what made the multiprocess implementation a transport
change rather than a redesign -- see `core_client_mp.py`, where it paid off.
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
            if executor_class is not None:
                raise NotImplementedError(
                    "a custom executor_class cannot be sent to the multiprocess "
                    "engine core: the child receives its configuration by pickle, "
                    "and a class defined in a test module does not survive the spawn "
                    "start method. Use multiprocess_mode=False for custom executors."
                )
            from pvllm.v1.engine.core_client_mp import AsyncMPClient, SyncMPClient

            if vllm_config.parallel_config.data_parallel_size > 1:
                raise NotImplementedError(
                    "data parallelism over the multiprocess engine core is not "
                    "implemented: upstream runs one core process per replica behind "
                    "a ZeroMQ coordinator, and this build's replicas share a process "
                    "(requirement R13.3). Use the in-process core, which models the "
                    "routing and the partitioned prefix cache faithfully."
                )
            client_class = AsyncMPClient if asyncio_mode else SyncMPClient
            return client_class(vllm_config, log_stats=log_stats)
        if vllm_config.parallel_config.data_parallel_size > 1:
            # R13.3. Several whole engines behind a router, rather than one engine.
            from pvllm.v1.engine.dp_client import DPInprocClient

            return DPInprocClient(
                vllm_config, executor_class=executor_class, log_stats=log_stats
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
    async def get_output_async(self) -> dict[int, EngineCoreOutputs]:
        """As `get_output`, from an event loop.

        Separate rather than one method the caller may or may not await: in process
        this is where the step's modeled duration is *spent*, and a real clock must
        release the loop while spending it or the server stops streaming exactly
        when it should be streaming most.
        """

    @property
    @abstractmethod
    def clock_time(self) -> float:
        """The engine's time. The only way a frontend may learn it (R19.1).

        On the interface, not just on the in-process client: over a process boundary
        the frontend cannot reach the clock at all, so every consumer has to go
        through here or it will reach for `time.time()` the moment the core moves.
        """

    @abstractmethod
    def make_stats(self) -> dict[str, Any]:
        """A snapshot of engine statistics for the metrics layer."""

    async def make_stats_async(self) -> dict[str, Any]:
        """As `make_stats`, from an event loop.

        On the base class with a synchronous default, so an async caller has one
        method to call whatever the transport is. The multiprocess client overrides
        it with a round trip; in process the two are the same call. Leaving the async
        form off the interface is what made `/metrics` return 500 under the
        multiprocess core -- the server had only the synchronous form to reach for,
        and that transport refused it.
        """
        return self.make_stats()

    @abstractmethod
    def reset_prefix_cache(self) -> bool: ...

    async def reset_prefix_cache_async(self) -> bool:
        """As `reset_prefix_cache`, from an event loop."""
        return self.reset_prefix_cache()

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

    async def get_output_async(self) -> dict[int, EngineCoreOutputs]:
        outputs, _ = await self.engine_core.step_async()
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
