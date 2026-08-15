"""The async engine.

Upstream: vllm/v1/engine/async_llm.py
Tier: B

R4.3: submission, an output-draining loop, per-request asyncio queues, and abort
propagation. What the OpenAI server sits on.

The output handler runs as a single background task pulling from the engine core and
fanning results into per-request queues. One loop rather than one per request, because
a step produces output for the whole batch at once -- and because the engine core must
be stepped from exactly one place.

**Cancellation is the load-bearing behaviour here (R2.4).** When a client disconnects,
`asyncio` cancels the generator awaiting its queue; the `finally` aborts the request in
the engine core, which frees its blocks within one step. Getting this wrong is
invisible until capacity runs out under real traffic -- the disconnected requests keep
generating and holding blocks forever -- which is exactly the failure a product wants
to test for and cannot, against an engine that leaks them too.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from pvllm.config import VllmConfig
from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.logger import init_logger
from pvllm.outputs import RequestOutput
from pvllm.sampling_params import SamplingParams
from pvllm.tokenizers import get_tokenizer
from pvllm.tokenizers.protocol import TokenizerLike
from pvllm.v1.engine.core_client import EngineCoreClient, InprocClient
from pvllm.v1.engine.input_processor import InputProcessor
from pvllm.v1.engine.output_processor import OutputProcessor

logger = init_logger(__name__)


class EngineDeadError(RuntimeError):
    """The engine died. R4.5: propagated to every in-flight request."""


class AsyncLLM:
    """An asyncio frontend over the engine core."""

    def __init__(self, vllm_config: VllmConfig, log_stats: bool = True) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.log_stats = log_stats
        self.errored = False
        self._dead_error: BaseException | None = None

        self.tokenizer: TokenizerLike = get_tokenizer(
            self.model_config.tokenizer or self.model_config.model,
            tokenizer_mode=self.model_config.tokenizer_mode,
            vocab_size=self.model_config.get_vocab_size(),
        )
        self.input_processor = InputProcessor(vllm_config, self.tokenizer)
        self.output_processor = OutputProcessor(self.tokenizer, log_stats=log_stats)
        self.engine_core: EngineCoreClient = EngineCoreClient.make_client(
            vllm_config, log_stats=log_stats
        )

        self._queues: dict[str, asyncio.Queue[RequestOutput | BaseException]] = {}
        self._output_handler: asyncio.Task[None] | None = None

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs) -> AsyncLLM:
        return cls(engine_args.create_engine_config())

    # --- the output loop -----------------------------------------------------

    def _ensure_output_handler(self) -> None:
        if self._output_handler is None or self._output_handler.done():
            self._output_handler = asyncio.create_task(self._run_output_handler())

    async def _run_output_handler(self) -> None:
        """Step the engine and fan results into per-request queues."""
        try:
            while True:
                if not self.engine_core.has_requests():
                    # Nothing to do. Yield rather than spin; a new request will be
                    # picked up on the next pass.
                    await asyncio.sleep(0)
                    if not self._queues:
                        return
                    continue

                engine_core_outputs = self.engine_core.get_output()
                for client_outputs in engine_core_outputs.values():
                    request_outputs = self.output_processor.process_outputs(
                        client_outputs.outputs
                    )
                    stopped_by_string: list[str] = []
                    for output in request_outputs:
                        queue = self._queues.get(output.request_id)
                        if queue is not None:
                            queue.put_nowait(output)
                        if (
                            output.finished
                            and output.request_id
                            not in self.output_processor.request_states
                        ):
                            stopped_by_string.append(output.request_id)
                    # R11.5: stop strings are invisible to the scheduler.
                    if stopped_by_string:
                        self.engine_core.abort_requests(stopped_by_string)

                # Give the event loop a turn so streaming actually streams rather
                # than delivering everything at the end.
                await asyncio.sleep(0)
        except Exception as exc:  # pragma: no cover - defensive
            # R4.5: a dead engine must reach every in-flight request, not hang them.
            logger.exception("Engine output handler died")
            self.errored = True
            self._dead_error = exc
            for queue in self._queues.values():
                queue.put_nowait(EngineDeadError(str(exc)))

    # --- generation ----------------------------------------------------------

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Yield outputs for one request until it finishes.

        Cancelling the consumer aborts the request in the engine core (R2.4).
        """
        if self.errored:
            raise EngineDeadError(str(self._dead_error))

        queue: asyncio.Queue[RequestOutput | BaseException] = asyncio.Queue()
        self._queues[request_id] = queue

        try:
            engine_request = self.input_processor.process_inputs(
                request_id, prompt, sampling_params, priority=priority
            )
            self.engine_core.add_request(engine_request)
            assert isinstance(self.engine_core, InprocClient)
            self.output_processor.add_request(
                request_id=request_id,
                prompt=prompt if isinstance(prompt, str) else None,
                prompt_token_ids=engine_request.prompt_token_ids or [],
                sampling_params=sampling_params,
                arrival_time=self.engine_core.clock_time,
            )
            self._ensure_output_handler()

            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
                if item.finished:
                    return
        finally:
            # Runs on normal completion *and* on cancellation. The abort is what
            # returns a disconnected client's blocks within one step; without it they
            # are held until the request would have finished on its own.
            self._queues.pop(request_id, None)
            if request_id in self.output_processor.request_states:
                self.engine_core.abort_requests([request_id])
                self.output_processor.abort_requests([request_id])

    async def abort(self, request_id: str) -> None:
        self.engine_core.abort_requests([request_id])
        self.output_processor.abort_requests([request_id])
        queue = self._queues.pop(request_id, None)
        if queue is not None:
            queue.put_nowait(asyncio.CancelledError())

    # --- introspection -------------------------------------------------------

    async def get_num_unfinished_requests(self) -> int:
        return self.output_processor.num_requests

    async def is_ready(self) -> bool:
        """R2.7: ready only once load and profiling are done.

        True by construction here: the engine core runs both in its constructor, so
        an `AsyncLLM` that exists is an engine that finished starting up.
        """
        return not self.errored

    def make_stats(self) -> dict[str, Any]:
        assert isinstance(self.engine_core, InprocClient)
        return self.engine_core.make_stats()

    async def reset_prefix_cache(self) -> bool:
        assert isinstance(self.engine_core, InprocClient)
        return self.engine_core.reset_prefix_cache()

    async def check_health(self) -> None:
        if self.errored:
            raise EngineDeadError(str(self._dead_error))

    def shutdown(self) -> None:
        """R4.5."""
        if self._output_handler is not None:
            self._output_handler.cancel()
            self._output_handler = None
        self.engine_core.shutdown()
