"""Serving /v1/chat/completions.

Upstream: vllm/entrypoints/openai/chat_completion/serving.py
Tier: B

Same streaming and disconnect handling as completions, with two chat-specific shapes
a client depends on: the first stream chunk carries `delta.role` and no content, and
subsequent chunks carry content and no role.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from pvllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
)
from pvllm.entrypoints.openai.completion.protocol import UsageInfo
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


class OpenAIServingChat:
    """Chat completions over the same engine."""

    def __init__(self, engine: AsyncLLM, served_model_names: list[str]) -> None:
        self.engine = engine
        self.served_model_names = served_model_names
        self._counter = 0

    def _next_request_id(self) -> str:
        self._counter += 1
        return f"chatcmpl-{self._counter:08d}"

    def _created(self) -> int:
        return int(self.engine.engine_core.clock_time)

    def _render(
        self, request: ChatCompletionRequest
    ) -> tuple[str | list[int], list[Any]]:
        """The prompt, and any multimodal placeholders in it. R18.

        Text-only requests still come back as a *string* and take exactly the path
        they did before multimodal existed. A request with an image comes back as
        token ids, because a placeholder is a token id and there is no text that
        tokenizes to one.
        """
        from pvllm.entrypoints.openai.multimodal import build_multimodal_prompt

        token_ids, features = build_multimodal_prompt(
            request.messages, self.engine.tokenizer
        )
        if token_ids is not None:
            return token_ids, features
        return str(
            self.engine.tokenizer.apply_chat_template(
                request.messages,
                add_generation_prompt=request.add_generation_prompt,
            )
        ), []

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request | None = None
    ) -> ChatCompletionResponse | JSONResponse | AsyncGenerator[str, None]:
        if request.model not in self.served_model_names:
            return model_not_found(request.model, self.served_model_names)
        if request.n != 1:
            return create_error_response(
                "n > 1 is handled by the parallel sampling layer (requirement R11.7), "
                "which lands in M2.",
                err_type="NotImplementedError",
                param="n",
            )
        if not request.messages:
            return create_error_response(
                "messages must not be empty.", param="messages"
            )

        request_id = self._next_request_id()
        try:
            prompt, mm_features = self._render(request)
            sampling_params = request.to_sampling_params(streaming=request.stream)
        except ValueError as exc:
            return create_error_response(str(exc))

        if request.stream:
            return self._stream(
                request, request_id, prompt, sampling_params, raw_request, mm_features
            )
        return await self._complete(
            request, request_id, prompt, sampling_params, raw_request, mm_features
        )

    async def _complete(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        raw_request: Request | None,
        mm_features: list[Any] | None = None,
    ) -> ChatCompletionResponse | JSONResponse:
        final: RequestOutput | None = None
        try:
            async for output in self.engine.generate(
                prompt,
                sampling_params,
                request_id,
                priority=request.priority,
                mm_features=mm_features,
            ):
                final = output
                if raw_request is not None and await raw_request.is_disconnected():
                    return create_error_response(
                        "Client disconnected.", err_type="ClientDisconnected"
                    )
        except Exception as exc:
            return to_error_response(exc)

        if final is None:
            return create_error_response("The engine produced no output.")

        completion = final.outputs[0]
        num_prompt_tokens = len(final.prompt_token_ids or ())
        return ChatCompletionResponse(
            id=request_id,
            created=self._created(),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=completion.text),
                    finish_reason=completion.finish_reason,
                    stop_reason=completion.stop_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=len(completion.token_ids),
                total_tokens=num_prompt_tokens + len(completion.token_ids),
                prompt_tokens_details={"cached_tokens": final.num_cached_tokens},
            ),
        )

    async def _stream(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        raw_request: Request | None,
        mm_features: list[Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        created = self._created()
        model = request.model
        num_prompt_tokens = 0
        num_completion_tokens = 0

        # The role chunk comes first, alone. A client that reads `delta.role` from
        # the first chunk and `delta.content` from the rest depends on the split.
        first = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0, delta=DeltaMessage(role="assistant")
                )
            ],
        )
        yield f"data: {first.model_dump_json()}\n\n"

        try:
            async for output in self.engine.generate(
                prompt,
                sampling_params,
                request_id,
                priority=request.priority,
                mm_features=mm_features,
            ):
                num_prompt_tokens = len(output.prompt_token_ids or ())
                completion = output.outputs[0]
                num_completion_tokens += len(completion.token_ids)

                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(content=completion.text),
                            finish_reason=completion.finish_reason,
                            stop_reason=completion.stop_reason,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            if request.stream_options and request.stream_options.include_usage:
                usage_chunk = ChatCompletionStreamResponse(
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
            logger.exception("Error while streaming chat completion %s", request_id)
            error = to_error_response(exc)
            yield f"data: {bytes(error.body).decode()}\n\n"

        yield "data: [DONE]\n\n"
