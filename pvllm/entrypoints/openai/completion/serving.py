"""Serving /v1/completions.

Upstream: vllm/entrypoints/openai/completion/serving.py
Tier: B

R2.3 and R2.4 both land here.

**Streaming.** SSE chunks in OpenAI's shape, terminated by `data: [DONE]`, with a
final usage chunk when `stream_options.include_usage` is set. The usage chunk has an
empty `choices` list -- clients that iterate choices unconditionally break on it, and
matching that quirk is the point of C5.

**Disconnect.** When a client goes away mid-stream, the generator is closed and
`AsyncLLM.generate`'s `finally` aborts the request in the engine core, returning its
blocks within one step. Without it a disconnected client keeps consuming capacity
until it would have finished on its own -- the exact failure a product wants to test
for and cannot against an engine that leaks it too.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from pvllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    UsageInfo,
)
from pvllm.entrypoints.serve.utils.error_response import (
    create_error_response,
    model_not_found,
    to_error_response,
)
from pvllm.logger import init_logger
from pvllm.outputs import RequestOutput
from pvllm.sampling_params import SamplingParams
from pvllm.v1.engine.async_llm import AsyncLLM

logger = init_logger(__name__)


class OpenAIServingCompletion:
    """Turns HTTP requests into engine requests and back."""

    def __init__(self, engine: AsyncLLM, served_model_names: list[str]) -> None:
        self.engine = engine
        self.served_model_names = served_model_names
        self._counter = 0
        #: R16.1. Set by the server once the adapter registry exists. Absent,
        #: only the base model is servable -- which is what a bare instance is.
        self.models: Any = None

    def _next_request_id(self) -> str:
        self._counter += 1
        return f"cmpl-{self._counter:08d}"

    def _created(self) -> int:
        """The engine's clock, never wall time (R19.1)."""
        return int(self.engine.engine_core.clock_time)

    async def create_completion(
        self, request: CompletionRequest, raw_request: Request | None = None
    ) -> CompletionResponse | JSONResponse | AsyncGenerator[str, None]:
        served, lora_request = (
            self.models.resolve(request.model)
            if self.models is not None
            else (request.model in self.served_model_names, None)
        )
        if not served:
            return model_not_found(
                request.model,
                self.served_model_names
                + (list(self.models.lora_modules) if self.models is not None else []),
            )

        # The OpenAI schema allows four prompt shapes. A bare string or a flat list
        # of token ids is one prompt; a list of strings or a list of token-id lists
        # is a *batch*, which upstream fans out into separate requests. Batching is
        # not implemented, so it is refused rather than silently treated as one
        # prompt -- which would return a single completion for N inputs.
        prompt: str | list[int]
        if isinstance(request.prompt, str):
            prompt = request.prompt
        elif not request.prompt:
            return create_error_response("prompt must not be empty.", param="prompt")
        elif all(isinstance(item, int) for item in request.prompt):
            prompt = [int(item) for item in request.prompt]  # type: ignore[arg-type]
        else:
            return create_error_response(
                "Batched prompts are not supported; send one prompt per request.",
                err_type="NotImplementedError",
                param="prompt",
            )

        request_id = self._next_request_id()

        try:
            sampling_params = request.to_sampling_params(streaming=request.stream)
        except ValueError as exc:
            return create_error_response(str(exc))

        if request.stream:
            return self._stream(
                request, request_id, prompt, sampling_params, raw_request, lora_request
            )
        return await self._complete(
            request, request_id, prompt, sampling_params, raw_request, lora_request
        )

    async def _complete(
        self,
        request: CompletionRequest,
        request_id: str,
        prompt: Any,
        sampling_params: SamplingParams,
        raw_request: Request | None,
        lora_request: Any = None,
    ) -> CompletionResponse | JSONResponse:
        final: RequestOutput | None = None
        try:
            async for output in self.engine.generate(
                prompt,
                sampling_params,
                request_id,
                priority=request.priority,
                lora_request=lora_request,
            ):
                final = output
                if raw_request is not None and await raw_request.is_disconnected():
                    # R2.4: stop pulling; the generator's finally aborts in the core.
                    return create_error_response(
                        "Client disconnected.",
                        err_type="ClientDisconnected",
                        status_code=HTTPStatus.BAD_REQUEST,
                    )
        except Exception as exc:
            return to_error_response(exc)

        if final is None:
            return create_error_response("The engine produced no output.")

        # R11.7. One choice per completion, in index order. `n == 1` is the same
        # code path with one element, so the common case cannot drift from the
        # parallel one.
        completions = sorted(final.outputs, key=lambda item: item.index)
        generated = sum(len(item.token_ids) for item in completions)
        return CompletionResponse(
            id=request_id,
            created=self._created(),
            model=request.model,
            choices=[
                CompletionResponseChoice(
                    index=item.index,
                    text=item.text,
                    finish_reason=item.finish_reason,
                    stop_reason=item.stop_reason,
                )
                for item in completions
            ],
            usage=UsageInfo(
                prompt_tokens=len(final.prompt_token_ids or ()),
                # Every choice's tokens: the prompt is billed once and shared
                # through the prefix cache, but each completion was generated.
                completion_tokens=generated,
                total_tokens=len(final.prompt_token_ids or ()) + generated,
                prompt_tokens_details={"cached_tokens": final.num_cached_tokens},
            ),
        )

    async def _stream(
        self,
        request: CompletionRequest,
        request_id: str,
        prompt: Any,
        sampling_params: SamplingParams,
        raw_request: Request | None,
        lora_request: Any = None,
    ) -> AsyncGenerator[str, None]:
        created = self._created()
        model = request.model
        num_prompt_tokens = 0
        num_completion_tokens = 0

        try:
            async for output in self.engine.generate(
                prompt,
                sampling_params,
                request_id,
                priority=request.priority,
                lora_request=lora_request,
            ):
                num_prompt_tokens = len(output.prompt_token_ids or ())
                # R11.7. Under `n > 1` a step's output carries whichever child
                # advanced, and its `index` is the choice index the client sees. One
                # chunk per completion, as OpenAI streams them.
                for completion in output.outputs:
                    num_completion_tokens += len(completion.token_ids)
                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=completion.index,
                                text=completion.text,
                                finish_reason=completion.finish_reason,
                                stop_reason=completion.stop_reason,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_unset=False)}\n\n"

            if request.stream_options and request.stream_options.include_usage:
                # R2.3. Empty `choices` is OpenAI's shape here, and clients that
                # iterate choices unconditionally break on it -- which is exactly
                # why it must be reproduced.
                usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[],
                    usage=UsageInfo(
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=num_completion_tokens,
                        total_tokens=num_prompt_tokens + num_completion_tokens,
                    ),
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"
        except Exception as exc:
            logger.exception("Error while streaming completion %s", request_id)
            error = to_error_response(exc)
            yield f"data: {bytes(error.body).decode()}\n\n"

        yield "data: [DONE]\n\n"
