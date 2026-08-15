"""Talking to an engine core in another process. R4.2, D2.

Upstream: vllm/v1/engine/core_client.py (the `MPClient` half)
Tier: B

Two sockets and a child process. The frontend PUSHes tagged frames in and PULLs
tagged frames out; `pvllm/v1/engine/core_proc.py` is the other end.

**The clock is the interesting part.** In process, `clock_time` reads
`engine_core.clock.time()` directly. Across a process boundary the frontend cannot
reach the clock at all -- R19.1 puts it in the core and nowhere else -- so it learns
the time from the `timestamp` on every outputs frame, and by an explicit round trip
when it needs a fresh one. That is why `clock_time` has been on the `EngineCoreClient`
interface since the first commit: had the frontend ever reached for `time.time()`
instead, in-process runs would have looked fine and this transport would have silently
mixed two timelines.

Cached rather than round-tripped on every read, because reading the clock is on the
hot path of every `add_request` and a synchronous round trip there would serialize the
frontend against the engine's step loop -- turning a modeled latency measurement into a
measurement of IPC. The cache is exact where it matters anyway: per-request timing
comes from the QUEUED and SCHEDULED events the core stamps, not from the cached value.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
import uuid
from typing import Any

import msgspec
import zmq

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    UtilityCall,
    UtilityReply,
)
from pvllm.v1.engine.core_client import EngineCoreClient
from pvllm.v1.engine.core_proc import DEAD, OUTPUTS, READY, UTILITY, run_engine_core

logger = init_logger(__name__)

#: How long to wait for the child to finish startup before giving up. Generous: the
#: child models weight loading and a profiling run before it reports ready, and on a
#: loaded CI machine spawning a fresh interpreter is not fast.
STARTUP_TIMEOUT_SECONDS = 120.0

#: How long a utility round trip may take. Bounded because the core answers these
#: between steps, so a hung engine would otherwise hang the frontend forever.
UTILITY_TIMEOUT_SECONDS = 30.0

#: How long to wait for a graceful exit before signalling. Short: a healthy core
#: exits in tens of milliseconds, so anything approaching this is already wedged,
#: and a long wait here is paid on every engine teardown in a test suite.
SHUTDOWN_TIMEOUT_SECONDS = 2.0

_encoder = msgspec.msgpack.Encoder()
_outputs_decoder = msgspec.msgpack.Decoder(EngineCoreOutputs)
_utility_decoder = msgspec.msgpack.Decoder(UtilityReply)
_dict_decoder = msgspec.msgpack.Decoder(dict)


class EngineDeadError(RuntimeError):
    """The engine core process died. R4.5."""


class MPClient(EngineCoreClient):
    """Shared transport for the sync and async multiprocess clients."""

    def __init__(self, vllm_config: VllmConfig, log_stats: bool = True) -> None:
        self.vllm_config = vllm_config
        self._closed = False
        self._dead_error: BaseException | None = None

        # A unique inproc-style ipc path per client, so two engines in one process
        # (a test suite, a sweep) cannot bind each other's sockets.
        token = uuid.uuid4().hex[:12]
        # A wildcard port read back after binding, rather than a port picked in
        # advance: two clients racing for the same fixed port is the kind of failure
        # that only appears under a parallel test runner. TCP rather than ipc so the
        # same code path works on Windows, which the CI matrix includes.
        wildcard = "tcp://127.0.0.1:*"

        self.ctx = zmq.Context()
        self.input_socket = self.ctx.socket(zmq.PUSH)
        self.input_socket.bind(wildcard)
        self.output_socket = self.ctx.socket(zmq.PULL)
        self.output_socket.bind(wildcard)

        resolved_input = self.input_socket.getsockopt_string(zmq.LAST_ENDPOINT)
        resolved_output = self.output_socket.getsockopt_string(zmq.LAST_ENDPOINT)

        context = multiprocessing.get_context("spawn")
        self.proc = context.Process(
            target=run_engine_core,
            args=(vllm_config, resolved_input, resolved_output, log_stats),
            name=f"pvllm-engine-core-{token}",
            daemon=True,
        )
        self.proc.start()

        #: Last engine time the frontend has seen. Advanced by every outputs frame.
        self._clock_time = 0.0
        self._pending: dict[int, Any] = {}
        self._next_call_id = 0

        self._await_ready()

    # --- startup and teardown ------------------------------------------------

    def _await_ready(self) -> None:
        """Block until the child reports that startup finished.

        Constructor-blocking on purpose: `LLM(...)` returning before the engine can
        serve would make the first request's latency include weight loading, and
        R2.7's readiness contract would be a lie at the one moment anybody checks it.

        Polled in slices rather than waiting out the whole timeout in one call,
        because the common startup failure is the child dying immediately -- and
        blocking for two minutes on a socket nobody will ever write to is the worst
        possible way to report that. Checking liveness between slices turns it into
        an error that arrives as fast as the child fails.
        """
        waited = 0.0
        while True:
            if self.output_socket.poll(timeout=100):
                break
            waited += 0.1
            if not self.proc.is_alive():
                raise EngineDeadError(
                    f"the engine core process exited with code {self.proc.exitcode} "
                    f"before reporting ready.\n"
                    f"If the traceback above mentions 'bootstrapping phase', the "
                    f"engine was constructed at module scope: multiprocess mode uses "
                    f"the spawn start method, which re-imports the main module, so "
                    f'construction has to happen under `if __name__ == "__main__":` '
                    f"or inside a function."
                )
            if waited >= STARTUP_TIMEOUT_SECONDS:
                self._kill()
                raise EngineDeadError(
                    f"the engine core process did not report ready within "
                    f"{STARTUP_TIMEOUT_SECONDS:g}s, and is still running. It is "
                    f"probably stuck in startup -- check its log output above."
                )
        tag, payload = self.output_socket.recv_multipart()
        if tag == DEAD:
            self._kill()
            raise EngineDeadError(
                f"the engine core process died during startup: "
                f"{_dict_decoder.decode(payload).get('error')}"
            )
        if tag != READY:
            self._kill()
            raise EngineDeadError(
                f"expected a ready frame from the engine core, got tag {tag!r}"
            )
        self._clock_time = float(_dict_decoder.decode(payload)["clock_time"])

    def _kill(self) -> None:
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            if self.proc.is_alive():  # pragma: no cover - stubborn child
                self.proc.kill()
                self.proc.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)

    def shutdown(self) -> None:
        if self._closed:
            return
        try:
            # Asked politely *before* the client marks itself closed, because
            # `_check_alive` refuses to send on a closed client -- setting the flag
            # first would swallow this message and leave every shutdown to the
            # five-second kill path. The core closes its trace file on the way out,
            # and a terminated process leaves a truncated trace that reads as a
            # dropped-records bug in a conformance diff rather than as a hard stop.
            self._send(EngineCoreRequestType.UTILITY, UtilityCall(-1, "shutdown"))
            self.proc.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except Exception:  # pragma: no cover - already going down
            pass
        self._closed = True
        self._kill()
        self.input_socket.close(linger=0)
        self.output_socket.close(linger=0)
        self.ctx.term()

    # --- transport -----------------------------------------------------------

    def _send(self, request_type: EngineCoreRequestType, payload: Any) -> None:
        self._check_alive()
        self.input_socket.send_multipart([request_type.value, _encoder.encode(payload)])

    def _check_alive(self) -> None:
        if self._closed:
            # Checked first, and it raises: after shutdown the sockets are closed, so
            # falling through would surface a raw `ZMQError` from deep inside pyzmq
            # instead of saying that the engine is gone.
            raise EngineDeadError(
                "this engine core client was shut down; build a new one to run more "
                "requests"
            )
        if self._dead_error is not None:
            raise EngineDeadError(str(self._dead_error))
        if not self.proc.is_alive():
            raise EngineDeadError(
                f"the engine core process exited with code {self.proc.exitcode}"
            )

    def _consume(
        self, tag: bytes, payload: bytes
    ) -> dict[int, EngineCoreOutputs] | None:
        """Turn one frame into outputs, or route it and return `None`."""
        if tag == OUTPUTS:
            outputs = _outputs_decoder.decode(payload)
            # Every outputs frame carries the engine's time, so the frontend's clock
            # advances as a side effect of receiving work -- no extra round trip on
            # the hot path.
            self._clock_time = outputs.timestamp
            return {outputs.engine_index: outputs}
        if tag == UTILITY:
            reply = _utility_decoder.decode(payload)
            self._pending[reply.call_id] = reply
            return None
        if tag == DEAD:
            self._dead_error = EngineDeadError(
                f"the engine core process died: "
                f"{_dict_decoder.decode(payload).get('error')}"
            )
            raise self._dead_error
        raise EngineDeadError(f"unknown frame tag from the engine core: {tag!r}")

    def _take_call_id(self) -> int:
        self._next_call_id += 1
        return self._next_call_id

    # --- the request interface -----------------------------------------------

    def add_request(self, request: EngineCoreRequest) -> None:
        self._send(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids:
            self._send(EngineCoreRequestType.ABORT, request_ids)

    @property
    def clock_time(self) -> float:
        """The engine's time as of the last frame received.

        Not a round trip. See the module docstring: per-request timing comes from the
        core's own QUEUED and SCHEDULED stamps, so a slightly stale value here cannot
        skew a latency measurement -- it only affects the wall against which a
        frontend-side log line is dated.
        """
        return self._clock_time


class SyncMPClient(MPClient):
    """Blocking client. What `LLMEngine` uses in multiprocess mode."""

    def get_output(self) -> dict[int, EngineCoreOutputs]:
        # Buffered frames first. A utility call (a `/metrics` scrape, say) can land
        # mid-generation and read a step's outputs off the socket while waiting for
        # its own reply; those are a request's tokens, and they are delivered here.
        if self._buffered:
            return self._buffered.pop(0)

        while True:
            self._check_alive()
            if not self.output_socket.poll(timeout=100):
                # No frame yet. Returning empty rather than blocking forever lets the
                # caller's own loop condition (`has_unfinished_requests`) decide when
                # to stop, which is what it does in process too.
                if not self._blocking_call("has_requests"):
                    return {}
                # That call may itself have buffered a frame that arrived while it
                # was waiting, so re-check before polling again.
                if self._buffered:
                    return self._buffered.pop(0)
                continue
            outputs = self._consume(*self.output_socket.recv_multipart())
            if outputs is not None:
                return outputs

    async def get_output_async(self) -> dict[int, EngineCoreOutputs]:
        raise NotImplementedError(
            "SyncMPClient cannot be awaited. Build the client with asyncio_mode=True "
            "(AsyncLLM does) to get AsyncMPClient."
        )

    def _blocking_call(self, method: str, *args: Any) -> Any:
        call_id = self._take_call_id()
        self._send(
            EngineCoreRequestType.UTILITY, UtilityCall(call_id, method, list(args))
        )

        deadline_ms = int(UTILITY_TIMEOUT_SECONDS * 1000)
        waited = 0
        while call_id not in self._pending:
            self._check_alive()
            if not self.output_socket.poll(timeout=100):
                waited += 100
                if waited >= deadline_ms:
                    raise EngineDeadError(
                        f"the engine core did not answer {method!r} within "
                        f"{UTILITY_TIMEOUT_SECONDS:g}s"
                    )
                continue
            tag, payload = self.output_socket.recv_multipart()
            outputs = self._consume(tag, payload)
            if outputs is not None:
                # An outputs frame arrived while waiting for a utility reply. Held
                # rather than dropped: it is a step's worth of generated tokens, and
                # discarding it would lose a request's output entirely.
                self._buffered.append(outputs)

        reply = self._pending.pop(call_id)
        if reply.error is not None:
            raise EngineDeadError(f"engine core: {reply.error}")
        return reply.result

    def __init__(self, vllm_config: VllmConfig, log_stats: bool = True) -> None:
        self._buffered: list[dict[int, EngineCoreOutputs]] = []
        super().__init__(vllm_config, log_stats=log_stats)

    def has_requests(self) -> bool:
        return bool(self._buffered) or bool(self._blocking_call("has_requests"))

    def get_num_unfinished_requests(self) -> int:
        return int(self._blocking_call("get_num_unfinished_requests"))

    def reset_prefix_cache(self) -> bool:
        return bool(self._blocking_call("reset_prefix_cache"))

    def make_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = self._blocking_call("make_stats")
        return stats


class AsyncMPClient(MPClient):
    """Non-blocking client. What `AsyncLLM` uses in multiprocess mode.

    Sockets are read on a background thread rather than through `zmq.asyncio`.
    Upstream uses `zmq.asyncio`; a thread plus a queue is the smaller thing that
    works here, and the difference is invisible above the interface. What matters is
    that the event loop is never blocked on a socket, so the server keeps streaming
    while the engine steps.
    """

    def __init__(self, vllm_config: VllmConfig, log_stats: bool = True) -> None:
        self._frames: queue.Queue[dict[int, EngineCoreOutputs]] = queue.Queue()
        self._lock = threading.Lock()
        super().__init__(vllm_config, log_stats=log_stats)
        self._reader = threading.Thread(
            target=self._read_frames, name="pvllm-client-reader", daemon=True
        )
        self._reader.start()

    def _read_frames(self) -> None:
        while not self._closed:
            try:
                if not self.output_socket.poll(timeout=50):
                    continue
                tag, payload = self.output_socket.recv_multipart()
            except zmq.ZMQError:  # pragma: no cover - shutdown race
                return
            try:
                with self._lock:
                    outputs = self._consume(tag, payload)
            except EngineDeadError:
                return
            if outputs is not None:
                self._frames.put(outputs)

    def get_output(self) -> dict[int, EngineCoreOutputs]:
        raise NotImplementedError(
            "AsyncMPClient must be awaited. Use get_output_async, or build the "
            "client with asyncio_mode=False to get SyncMPClient."
        )

    async def get_output_async(self) -> dict[int, EngineCoreOutputs]:
        import asyncio

        while True:
            self._check_alive()
            try:
                return self._frames.get_nowait()
            except queue.Empty:
                # Yields to the loop rather than blocking on the socket, which is the
                # entire reason this client exists: the server has to keep answering
                # while the engine is mid-step.
                await asyncio.sleep(0.001)
                if not self._frames.qsize() and not await self._call_async(
                    "has_requests"
                ):
                    return {}

    async def _call_async(self, method: str, *args: Any) -> Any:
        import asyncio

        call_id = self._take_call_id()
        self._send(
            EngineCoreRequestType.UTILITY, UtilityCall(call_id, method, list(args))
        )

        waited = 0.0
        while call_id not in self._pending:
            self._check_alive()
            await asyncio.sleep(0.001)
            waited += 0.001
            if waited >= UTILITY_TIMEOUT_SECONDS:
                raise EngineDeadError(
                    f"the engine core did not answer {method!r} within "
                    f"{UTILITY_TIMEOUT_SECONDS:g}s"
                )
        reply = self._pending.pop(call_id)
        if reply.error is not None:
            raise EngineDeadError(f"engine core: {reply.error}")
        return reply.result

    def has_requests(self) -> bool:
        """Whether work remains, from what has already been received.

        Synchronous by necessity -- the interface is -- so it answers from the local
        frame queue rather than round-tripping. `get_output_async` does the
        authoritative check, where it can await.
        """
        return not self._frames.empty()

    def get_num_unfinished_requests(self) -> int:
        raise NotImplementedError(
            "AsyncMPClient.get_num_unfinished_requests would need to block the event "
            "loop on a round trip. AsyncLLM tracks its own in-flight requests; use "
            "that instead."
        )

    def reset_prefix_cache(self) -> bool:
        raise NotImplementedError(
            "use `await AsyncMPClient.reset_prefix_cache_async()`; the synchronous "
            "form would block the event loop on a round trip"
        )

    async def reset_prefix_cache_async(self) -> bool:
        return bool(await self._call_async("reset_prefix_cache"))

    def make_stats(self) -> dict[str, Any]:
        raise NotImplementedError(
            "use `await AsyncMPClient.make_stats_async()`; the synchronous form "
            "would block the event loop on a round trip"
        )

    async def make_stats_async(self) -> dict[str, Any]:
        stats: dict[str, Any] = await self._call_async("make_stats")
        return stats
