"""`POST /v1/embeddings`. R2.2, C5.

Upstream: vllm/entrypoints/pooling/embed/serving.py
Tier: B

The vectors are synthetic and carry no meaning; see `pvllm/pooling_params.py`. What
is real is everything a product's embedding path actually exercises: the request
schema, the batching, the token accounting, the context-length error, the queueing
behaviour of a page of documents arriving at once, and the prefix-cache sharing
between documents with a common preamble.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from pvllm.entrypoints.pooling.embed.protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    UsageInfo,
)
from pvllm.entrypoints.serve.utils.error_response import (
    model_not_found,
    not_implemented,
    to_error_response,
)
from pvllm.logger import init_logger
from pvllm.v1.engine.async_llm import AsyncLLM

logger = init_logger(__name__)


def split_inputs(
    value: str | list[str] | list[int] | list[list[int]],
) -> list[str | list[int]]:
    """The four OpenAI input shapes, resolved to a list of prompts.

    A flat list of ints is *one* prompt of token ids, not a batch of single-token
    prompts -- the ambiguity is in the OpenAI schema itself, and getting it backwards
    would silently turn one document into hundreds.
    """
    if isinstance(value, str):
        return [value]
    if not value:
        raise ValueError("input must not be empty.")
    if all(isinstance(item, int) for item in value):
        return [[int(item) for item in value]]  # type: ignore[arg-type]
    prompts: list[str | list[int]] = []
    for item in value:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, list) and all(isinstance(token, int) for token in item):
            prompts.append([int(token) for token in item])
        else:
            raise ValueError(
                "input must be a string, a list of strings, a list of token ids, or "
                "a list of token-id lists."
            )
    return prompts


class OpenAIServingEmbedding:
    """The embeddings endpoint's handler."""

    def __init__(self, engine: AsyncLLM, served_model_names: list[str]) -> None:
        self.engine = engine
        self.served_model_names = served_model_names
        self._counter = 0
        self.models: Any = None

    def _next_request_id(self) -> str:
        self._counter += 1
        return f"embd-{self._counter:08d}"

    def _created(self) -> int:
        return int(self.engine.engine_core.clock_time)

    async def create_embedding(
        self, request: EmbeddingRequest, raw_request: Request | None = None
    ) -> EmbeddingResponse | JSONResponse:
        # R16.1. The adapter, not just whether the name is served. Discarding it
        # returned 200 with `model: adapter-a` while the engine request carried no
        # adapter at all -- the corpus was hashed into the *base* model's
        # prefix-cache partition (C3) and cost no adapter memory, so the capacity
        # answer was the base model's reported as the adapter's.
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

        if request.encoding_format != "float":
            return not_implemented(
                f"encoding_format {request.encoding_format!r} is not supported; only "
                f"'float' is. Base64 encoding would compress a vector this engine "
                f"invents, which would be precision no number here has.",
                param="encoding_format",
            )

        try:
            prompts = split_inputs(request.input)
            pooling_params = request.to_pooling_params()
        except (ValueError, NotImplementedError) as exc:
            return to_error_response(exc)

        request_id = self._next_request_id()

        async def one(index: int, prompt: str | list[int]) -> Any:
            # Every document is its own engine request, which is what makes a page
            # of them queue, batch and share prefixes the way it would in production.
            stream = self.engine.encode(
                prompt,
                pooling_params,
                f"{request_id}-{index}",
                priority=request.priority,
                lora_request=lora_request,
            )
            try:
                async for output in stream:
                    if output.finished:
                        return output
            finally:
                # Closed explicitly rather than left to cancellation: an async
                # generator's `finally` -- which is what aborts the request in the
                # core -- runs on `aclose()` or on garbage collection, and waiting
                # for the collector means waiting an unbounded time while the engine
                # still holds the blocks.
                await stream.aclose()
            raise RuntimeError(f"embedding request {request_id}-{index} produced none")

        tasks = [
            asyncio.ensure_future(one(index, prompt))
            for index, prompt in enumerate(prompts)
        ]
        # FIRST_EXCEPTION, not `gather`. One bad document must not leave its siblings
        # running: `gather` waits for every task whatever happens, so a 400 returned
        # while the good documents were still prefilling left the engine holding
        # their blocks until they finished on their own -- and R2.4 says an abandoned
        # request's blocks come back within one step. Upstream gets the same effect
        # from `merge_async_iterators`, whose `finally` closes the rest.
        finished, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_EXCEPTION
        )
        failure = next(
            (
                exc
                for exc in (task.exception() for task in finished)
                if isinstance(exc, Exception)
            ),
            None,
        )
        if failure is not None:
            for task in pending:
                task.cancel()
            # Awaited so each cancellation actually runs the generator's `finally`,
            # which is what issues the abort. Returning before that would report the
            # error while the siblings were still holding KV.
            await asyncio.gather(*pending, return_exceptions=True)
            return to_error_response(failure)
        outputs = [task.result() for task in tasks]

        prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        return EmbeddingResponse(
            id=request_id,
            created=self._created(),
            model=request.model,
            data=[
                EmbeddingResponseData(index=index, embedding=output.outputs.data)
                for index, output in enumerate(outputs)
            ],
            # No completion tokens: an embedding request generates nothing, which is
            # exactly what OpenAI's usage block reports for it.
            usage=UsageInfo(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
        )


__all__ = ["OpenAIServingEmbedding", "split_inputs"]
