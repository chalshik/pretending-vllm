"""The engine core in its own process. R4.2, D2.

Upstream: vllm/v1/engine/core.py (the `EngineCoreProc` half)
Tier: B

Upstream runs the engine core in a background process so that ZMQ socket IO and
serialization overlap the GPU forward pass. There is no GPU here, so the *performance*
argument does not carry -- but two fidelity arguments do, and they are why this is a
real process rather than a thread pretending to be one:

**Serialization is real.** Every request is msgpack-encoded and decoded. A product
that sends something the wire format cannot carry finds out here, exactly as it would
against real vLLM, rather than getting away with it because the in-process client
passed the object by reference.

**Backpressure is real.** The sockets have real high-water marks and the core's busy
loop really does block. A frontend that outruns the engine hits the same wall.

**What this trades away is determinism.** In process, `LLM.generate` submits every
prompt before the first step, so the batch composition is fixed and B4 holds byte for
byte. Here the core steps whenever it has work, so whether request seven arrived
before step three depends on OS scheduling. The engine's *decisions* are still
deterministic given an arrival order -- but the arrival order is not. That is upstream's
behavior too, and it is why D2 makes the in-process client the default and why the
conformance suite (C1--C4) uses it exclusively.

The clock stays in this process (R19.1). The frontend learns the time only from the
`timestamp` on every `EngineCoreOutputs` and from an explicit utility call, which is
what the `EngineCoreClient` interface was shaped for from the first commit.
"""

from __future__ import annotations

import queue
import signal
import threading
from typing import Any

import msgspec
import zmq

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.v1.engine import (
    EngineCoreRequest,
    EngineCoreRequestType,
    UtilityCall,
    UtilityReply,
)
from pvllm.v1.engine.core import EngineCore
from pvllm.v1.executor.abstract import Executor

logger = init_logger(__name__)

#: Frame tags on the output socket. One byte, same reasoning as the input tags.
OUTPUTS = b"\x00"
UTILITY = b"\x01"
READY = b"\x02"
DEAD = b"\x03"

#: How long the busy loop waits for input when it has nothing to run. Bounded rather
#: than infinite so a shutdown that races the loop still lands within one tick
#: instead of needing a wakeup sentinel to be delivered perfectly.
IDLE_POLL_SECONDS = 0.02

_encoder = msgspec.msgpack.Encoder()
_request_decoder = msgspec.msgpack.Decoder(EngineCoreRequest)
_abort_decoder = msgspec.msgpack.Decoder(list[str])
_utility_decoder = msgspec.msgpack.Decoder(UtilityCall)


class EngineCoreProc(EngineCore):
    """`EngineCore` wrapped in sockets and a busy loop."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        input_address: str,
        output_address: str,
        executor_class: type[Executor] | None = None,
        log_stats: bool = True,
    ) -> None:
        super().__init__(
            vllm_config, executor_class=executor_class, log_stats=log_stats
        )

        self.input_queue: queue.Queue[tuple[EngineCoreRequestType, Any]] = queue.Queue()
        self.output_queue: queue.Queue[tuple[bytes, Any]] = queue.Queue()
        self._running = True

        self.ctx = zmq.Context()
        self.input_socket = self.ctx.socket(zmq.PULL)
        self.input_socket.connect(input_address)
        self.output_socket = self.ctx.socket(zmq.PUSH)
        self.output_socket.connect(output_address)

        # Socket IO on its own threads and the step loop on the main one, as
        # upstream does. Here it buys responsiveness rather than overlap: a request
        # arriving mid-step is queued immediately instead of waiting for the step to
        # end before anyone reads the socket.
        self._threads = [
            threading.Thread(
                target=self._read_input_socket, name="pvllm-core-input", daemon=True
            ),
            threading.Thread(
                target=self._write_output_socket, name="pvllm-core-output", daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()

        # Sent once startup (weight load, profiling, KV allocation) is complete, so
        # the client's constructor can return only when the engine is genuinely
        # ready -- R2.7's readiness, established at the transport level rather than
        # by polling.
        self.output_queue.put((READY, {"clock_time": self.clock.time()}))

    # --- socket threads ------------------------------------------------------

    def _read_input_socket(self) -> None:
        while self._running:
            try:
                if not self.input_socket.poll(timeout=50):
                    continue
                tag, payload = self.input_socket.recv_multipart()
            except zmq.ZMQError:  # pragma: no cover - shutdown race
                return

            # Decoding is the one place a *client* bug can reach this thread, and
            # letting it escape would kill the thread and leave the process running
            # with nobody draining the socket -- an engine that is not dead but
            # permanently deaf, whose symptom is a utility call timing out thirty
            # seconds later with a message naming the wrong cause. A frame that
            # cannot be decoded is dropped loudly instead.
            try:
                request_type = EngineCoreRequestType(tag)
                if request_type is EngineCoreRequestType.ADD:
                    decoded: Any = _request_decoder.decode(payload)
                elif request_type is EngineCoreRequestType.ABORT:
                    decoded = _abort_decoder.decode(payload)
                elif request_type is EngineCoreRequestType.UTILITY:
                    decoded = _utility_decoder.decode(payload)
                else:
                    continue
            except Exception:
                logger.exception(
                    "dropping an undecodable input frame (tag %r, %d bytes). The "
                    "engine is still serving; the request that produced this frame "
                    "is lost.",
                    tag,
                    len(payload),
                )
                continue
            self.input_queue.put((request_type, decoded))

    def _write_output_socket(self) -> None:
        while self._running:
            try:
                tag, payload = self.output_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.output_socket.send_multipart([tag, _encoder.encode(payload)])
            except zmq.ZMQError:  # pragma: no cover - shutdown race
                return

    # --- the loop ------------------------------------------------------------

    def run_busy_loop(self) -> None:
        """Step whenever there is work, wait on the socket when there is not.

        `while True` with an explicit break rather than `while self._running`: the
        flag is cleared inside `_process_input_queue`, and a loop condition that
        reads it up front suggests the check at the top is the one that matters when
        the one after the drain is.
        """
        while True:
            self._process_input_queue()
            if not self._running:
                break
            if not self.has_requests():
                continue
            outputs, _ = self.step()
            for client_outputs in outputs.values():
                self.output_queue.put((OUTPUTS, client_outputs))

    def _process_input_queue(self) -> None:
        """Drain pending input, blocking briefly when the engine is idle.

        Blocking when idle is what keeps a waiting engine off the CPU. Bounded, so
        the loop still notices `_running` going false; an unbounded wait would need
        the shutdown path to deliver a wakeup sentinel without ever racing, which is
        more machinery than the 20 ms costs.
        """
        block = not self.has_requests()
        while True:
            try:
                request_type, payload = self.input_queue.get(
                    block=block, timeout=IDLE_POLL_SECONDS if block else None
                )
            except queue.Empty:
                return
            block = False
            self._handle(request_type, payload)
            if self.input_queue.empty():
                return

    def _handle(self, request_type: EngineCoreRequestType, payload: Any) -> None:
        if request_type is EngineCoreRequestType.ADD:
            self.add_request(payload)
        elif request_type is EngineCoreRequestType.ABORT:
            self.abort_requests(payload)
        elif request_type is EngineCoreRequestType.UTILITY:
            self.output_queue.put((UTILITY, self._call_utility(payload)))
        elif request_type is EngineCoreRequestType.WAKEUP:
            self._running = False

    def _call_utility(self, call: UtilityCall) -> UtilityReply:
        """Run one frontend-initiated method and package the answer.

        Restricted to an explicit allow-list. Dispatching to `getattr(self, method)`
        would turn the input socket into arbitrary remote code execution against the
        engine, and the set of things a frontend legitimately asks for is six names
        long.
        """
        allowed = {
            "clock_time": lambda: self.clock.time(),
            "make_stats": self.make_stats,
            "reset_prefix_cache": self.reset_prefix_cache,
            "has_requests": self.has_requests,
            "get_num_unfinished_requests": self.get_num_unfinished_requests,
            "shutdown": self.request_stop,
        }
        handler = allowed.get(call.method)
        if handler is None:
            return UtilityReply(
                call_id=call.call_id,
                error=(
                    f"unknown utility method {call.method!r}; the engine core accepts "
                    f"{sorted(allowed)}"
                ),
            )
        try:
            return UtilityReply(call_id=call.call_id, result=handler(*call.args))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("utility call %s failed", call.method)
            return UtilityReply(
                call_id=call.call_id, error=f"{type(exc).__name__}: {exc}"
            )

    def request_stop(self) -> bool:
        """Ask the busy loop to finish the current step and exit.

        Safe from a signal handler: it sets a flag and returns, so the loop unwinds
        through `close()` and the trace file is flushed. Raising out of a handler
        would abort wherever the interpreter happened to be, which for a trace write
        means a half-written final record.
        """
        self._running = False
        return True

    def close(self) -> None:
        self._running = False
        for thread in self._threads:
            thread.join(timeout=1.0)
        self.input_socket.close(linger=0)
        self.output_socket.close(linger=0)
        self.ctx.term()
        super().shutdown()


def run_engine_core(
    vllm_config: VllmConfig,
    input_address: str,
    output_address: str,
    log_stats: bool = True,
) -> None:
    """The child process entry point.

    Module-level and taking only picklable arguments, because that is what
    `multiprocessing` with the spawn start method can deliver. Spawn rather than fork
    is the default on macOS and the safe choice everywhere: a forked child inherits
    the parent's threads and locks, and this process starts two threads immediately.
    """
    engine = None
    try:
        engine = EngineCoreProc(
            vllm_config,
            input_address=input_address,
            output_address=output_address,
            log_stats=log_stats,
        )
        # Installed only once the engine exists, and it *stops the loop* rather than
        # ignoring the signal. Ignoring it outright -- which is easy to reach for,
        # since the default handler would raise out of a socket call and look like a
        # crash -- makes `Process.terminate()` a no-op, so every teardown falls
        # through to SIGKILL and pays the full join timeout first. Before this point
        # the default handler is the right one: there is no trace file open yet.
        signal.signal(signal.SIGTERM, lambda *_: engine.request_stop())
        engine.run_busy_loop()
    except Exception as exc:
        logger.exception("engine core process died")
        # R4.5: a dead engine has to reach the frontend, or every in-flight request
        # hangs on a socket nobody will ever write to again.
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.connect(output_address)
            socket.send_multipart(
                [DEAD, _encoder.encode({"error": f"{type(exc).__name__}: {exc}"})]
            )
            socket.close(linger=1000)
            context.term()
        except Exception:  # pragma: no cover - the process is already failing
            pass
    finally:
        if engine is not None:
            engine.close()
