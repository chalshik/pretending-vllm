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

from pvllm import envs
from pvllm.config import VllmConfig
from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.logger import init_logger
from pvllm.outputs import PoolingRequestOutput, RequestOutput
from pvllm.pooling_params import PoolingParams
from pvllm.sampling_params import SamplingParams
from pvllm.tokenizers import get_tokenizer
from pvllm.tokenizers.protocol import TokenizerLike
from pvllm.v1.engine.core_client import EngineCoreClient
from pvllm.v1.engine.input_processor import InputProcessor
from pvllm.v1.engine.output_processor import OutputProcessor
from pvllm.v1.engine.parallel_sampling import ParentRequest
from pvllm.v1.metrics.stats import IterationStats

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
        # R4.2. Off unless asked for, unlike upstream -- see
        # PVLLM_ENABLE_V1_MULTIPROCESSING in pvllm/envs.py for why the default is
        # inverted here.
        multiprocess = envs.PVLLM_ENABLE_V1_MULTIPROCESSING
        self.engine_core: EngineCoreClient = EngineCoreClient.make_client(
            vllm_config,
            multiprocess_mode=multiprocess,
            asyncio_mode=multiprocess,
            log_stats=log_stats,
        )

        self._queues: dict[
            str, asyncio.Queue[RequestOutput | PoolingRequestOutput | BaseException]
        ] = {}
        self._output_handler: asyncio.Task[None] | None = None
        #: Accumulated since the last drain. `/metrics` takes and clears it, so a
        #: histogram observation is recorded exactly once however often it scrapes.
        self._pending_stats = IterationStats()

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

                now = self.engine_core.clock_time
                # Awaited, not called: under a real or scaled clock this is where
                # the step's modeled duration is spent, and holding the loop through
                # it would stop the server streaming for exactly as long as the step
                # it is streaming through.
                engine_core_outputs = await self.engine_core.get_output_async()
                for client_outputs in engine_core_outputs.values():
                    request_outputs = self.output_processor.process_outputs(
                        client_outputs.outputs, now, self._pending_stats
                    )
                    for output in request_outputs:
                        queue = self._queues.get(output.request_id)
                        if queue is not None:
                            queue.put_nowait(output)
                    # R11.5: only stop-string terminations need aborting; the
                    # scheduler already knows about the ones it ended itself.
                    stopped = self.output_processor.take_stopped_by_string()
                    if stopped:
                        self.engine_core.abort_requests(stopped)

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
        lora_request: Any = None,
        mm_features: list[Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Yield outputs for one request until it finishes.

        Cancelling the consumer aborts the request in the engine core (R2.4).
        """
        if self.errored:
            raise EngineDeadError(str(self._dead_error))

        queue: asyncio.Queue[RequestOutput | PoolingRequestOutput | BaseException] = (
            asyncio.Queue()
        )
        self._queues[request_id] = queue
        #: The engine ids to abort if the consumer goes away. Equal to `[request_id]`
        #: unless `n > 1`, where the client's id names no engine request at all.
        children: list[str] = []

        try:
            # R11.7. The same fan-out as the offline engine: `n` children, one
            # response. The queue is registered under the *parent's* id, which is
            # what the output handler routes finished aggregates to.
            parent = (
                ParentRequest(request_id, sampling_params)
                if sampling_params.n > 1
                else None
            )
            for index in range(sampling_params.n):
                child_id, child_params = (
                    parent.child_info(index)
                    if parent is not None
                    else (request_id, sampling_params)
                )
                engine_request = self.input_processor.process_inputs(
                    child_id,
                    prompt,
                    child_params,
                    priority=priority,
                    lora_request=lora_request,
                    mm_features=mm_features,
                )
                self.engine_core.add_request(engine_request)
                self.output_processor.add_request(
                    request_id=child_id,
                    prompt=prompt if isinstance(prompt, str) else None,
                    prompt_token_ids=engine_request.prompt_token_ids or [],
                    sampling_params=child_params,
                    arrival_time=self.engine_core.clock_time,
                    parent_request=parent,
                    index=index,
                )
                children.append(child_id)
            self._ensure_output_handler()

            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                assert isinstance(item, RequestOutput)
                yield item
                if item.finished:
                    return
        finally:
            # Runs on normal completion *and* on cancellation. The abort is what
            # returns a disconnected client's blocks within one step; without it they
            # are held until the request would have finished on its own.
            self._queues.pop(request_id, None)
            live = [
                child
                for child in (children or [request_id])
                if child in self.output_processor.request_states
            ]
            if live:
                self.engine_core.abort_requests(live)
                self.output_processor.abort_requests(live)

    async def encode(
        self,
        prompt: str | list[int],
        pooling_params: PoolingParams,
        request_id: str,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Yield one pooling output and finish. R2.2.

        A generator rather than a coroutine because that is upstream's shape, and
        because it makes cancellation work the same way `generate` does: dropping the
        consumer aborts the request in the core.
        """
        if self.errored:
            raise EngineDeadError(str(self._dead_error))

        queue: asyncio.Queue[RequestOutput | PoolingRequestOutput | BaseException] = (
            asyncio.Queue()
        )
        self._queues[request_id] = queue

        try:
            engine_request = self.input_processor.process_inputs(
                request_id,
                prompt,
                priority=priority,
                pooling_params=pooling_params,
            )
            self.engine_core.add_request(engine_request)
            self.output_processor.add_request(
                request_id=request_id,
                prompt=prompt if isinstance(prompt, str) else None,
                prompt_token_ids=engine_request.prompt_token_ids or [],
                sampling_params=None,
                arrival_time=self.engine_core.clock_time,
                pooling_params=pooling_params,
            )
            self._ensure_output_handler()

            while True:
                item = await queue.get()
                if isinstance(item, BaseException):
                    raise item
                assert isinstance(item, PoolingRequestOutput)
                yield item
                if item.finished:
                    return
        finally:
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

    async def make_stats(self) -> dict[str, Any]:
        return await self.engine_core.make_stats_async()

    def take_iteration_stats(self) -> IterationStats:
        """Drain the accumulated per-request timings.

        Taken rather than read: each observation belongs in a histogram exactly
        once, and a scrape that left them in place would re-observe every request on
        every subsequent scrape, inflating the counts without bound.
        """
        stats, self._pending_stats = self._pending_stats, IterationStats()
        return stats

    async def reset_prefix_cache(self) -> bool:
        return await self.engine_core.reset_prefix_cache_async()

    async def check_health(self) -> None:
        if self.errored:
            raise EngineDeadError(str(self._dead_error))

    def shutdown(self) -> None:
        """R4.5."""
        if self._output_handler is not None:
            self._output_handler.cancel()
            self._output_handler = None
        self.engine_core.shutdown()
