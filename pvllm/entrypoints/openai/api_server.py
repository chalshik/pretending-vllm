"""The OpenAI-compatible HTTP server.

Upstream: vllm/entrypoints/openai/api_server.py
Tier: B

R2.2's endpoint set, assembled. This is the surface a product actually points at
(G4), so the routes, their shapes, and their errors are the contract (C5, C7).

R2.7: `/health` reports ready only once load and profiling are complete. That is true
by construction -- the engine core runs both in its constructor -- but the endpoint
exists and reports it, because a product that polls readiness needs something to poll.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from pvllm import __version__
from pvllm.config import VllmConfig
from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from pvllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from pvllm.entrypoints.openai.completion.protocol import CompletionRequest
from pvllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from pvllm.entrypoints.openai.models.serving import (
    OpenAIServingModels,
    build_lora_modules,
)
from pvllm.entrypoints.openai.responses.protocol import ResponsesRequest
from pvllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from pvllm.entrypoints.pooling.embed.protocol import EmbeddingRequest
from pvllm.entrypoints.pooling.embed.serving import OpenAIServingEmbedding
from pvllm.entrypoints.serve.tokenize.serving import (
    DetokenizeRequest,
    OpenAIServingTokenization,
    TokenizeRequest,
)
from pvllm.entrypoints.serve.utils.error_response import to_error_response
from pvllm.logger import init_logger
from pvllm.v1.engine.async_llm import AsyncLLM
from pvllm.v1.metrics.loggers import PrometheusStatLogger
from pvllm.v1.metrics.stats import SchedulerStats

logger = init_logger(__name__)


class ServerState:
    """Everything the routes need, built once at startup."""

    def __init__(self, vllm_config: VllmConfig, registry: CollectorRegistry) -> None:
        self.vllm_config = vllm_config
        self.engine = AsyncLLM(vllm_config)
        model_config = vllm_config.model_config
        assert model_config.max_model_len is not None

        self.served_model_names = [model_config.served_model_name or model_config.model]
        # An alias so a client configured with the card name reaches the same model
        # as one configured with the Hugging Face id.
        if model_config.model not in self.served_model_names:
            self.served_model_names.append(model_config.model)

        created = int(self.engine.engine_core.clock_time)
        self.completion = OpenAIServingCompletion(self.engine, self.served_model_names)
        self.chat = OpenAIServingChat(self.engine, self.served_model_names)
        self.embedding = OpenAIServingEmbedding(self.engine, self.served_model_names)
        self.responses = OpenAIServingResponses(self.engine, self.served_model_names)
        # R16.1. `--lora-modules name=path`, resolved once. Each adapter gets a
        # stable integer id: the id partitions the prefix cache, so it must not
        # change between requests naming the same adapter.
        lora_modules = build_lora_modules(
            vllm_config.lora_config, getattr(vllm_config, "lora_modules", None)
        )
        self.models = OpenAIServingModels(
            self.served_model_names,
            model_config.max_model_len,
            created,
            lora_modules=lora_modules,
        )
        self.completion.models = self.models
        self.chat.models = self.models
        self.embedding.models = self.models
        self.responses.models = self.models
        self.tokenization = OpenAIServingTokenization(
            self.engine.tokenizer, model_config.max_model_len
        )
        self.registry = registry
        self.stat_logger = PrometheusStatLogger(vllm_config, registry=registry)

    async def refresh_metrics(self) -> None:
        """Pull the engine's current state into the Prometheus gauges.

        Scraped rather than pushed: the engine core owns the numbers, and having
        `/metrics` read them at scrape time means a scrape can never observe a
        half-updated step.
        """
        stats = await self.engine.make_stats()
        self.stat_logger.record(
            SchedulerStats(
                num_running_reqs=stats["num_running_reqs"],
                num_waiting_reqs=stats["num_waiting_reqs"],
                kv_cache_usage=stats["kv_cache_usage"],
                prefix_cache_queries=int(stats["prefix_cache_queries"]),
                prefix_cache_hits=int(stats["prefix_cache_hits"]),
                num_preemptions=stats["num_preemptions"],
                step_index=stats["step_index"],
                num_draft_tokens=int(stats.get("num_draft_tokens", 0)),
                num_accepted_tokens=int(stats.get("num_accepted_tokens", 0)),
                mm_cache_queries=int(stats.get("mm_cache_queries", 0)),
                mm_cache_hits=int(stats.get("mm_cache_hits", 0)),
                external_prefix_cache_queries=int(
                    stats.get("external_prefix_cache_queries", 0)
                ),
                external_prefix_cache_hits=int(
                    stats.get("external_prefix_cache_hits", 0)
                ),
            ),
            self.engine.take_iteration_stats(),
        )

    def shutdown(self) -> None:
        self.engine.shutdown()


async def convert_stream_to_sse_events(
    generator: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[str]:
    """Frame Responses events as named SSE. R2.3, C5.

    Two differences from every other streaming endpoint here, both of which a client
    can see. Each frame carries an `event:` line naming the type, because the payloads
    are heterogeneous and a client dispatches on it. And there is **no** trailing
    `data: [DONE]` -- the stream ends after `response.completed`. Chat completions
    sends the sentinel; this does not, and a client that waits for one waits forever.
    """
    async for event in generator:
        payload = json.dumps(event["payload"], separators=(",", ":"))
        yield f"event: {event['type']}\ndata: {payload}\n\n"


def build_app(
    vllm_config: VllmConfig,
    registry: CollectorRegistry | None = None,
    enable_debug_endpoints: bool = False,
) -> FastAPI:
    """Construct the ASGI app.

    Takes a resolved config rather than argv so tests can drive it in process, and
    an explicit registry so two apps in one process do not collide on metric names.

    `enable_debug_endpoints` attaches the read-only `/debug/*` router (D9). Off by
    default, mirroring upstream's `VLLM_SERVER_DEV_MODE` gate on its own `serve/dev/`
    routers.
    """
    registry = registry if registry is not None else CollectorRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        state.shutdown()

    app = FastAPI(title="pretending-vllm", version=__version__, lifespan=lifespan)
    state = ServerState(vllm_config, registry)
    app.state.server = state
    app.state.vllm_config = vllm_config

    if enable_debug_endpoints:
        from pvllm.entrypoints.serve.dev.api_router import attach_router
        from pvllm.entrypoints.serve.dev.introspect import EngineIntrospector

        app.state.introspector = EngineIntrospector(state.engine)
        attach_router(app)

    # --- OpenAI endpoints (R2.2) -----------------------------------------

    @app.post("/v1/completions")
    async def create_completion(
        request: CompletionRequest, raw_request: Request
    ) -> Any:
        try:
            result = await state.completion.create_completion(request, raw_request)
        except Exception as exc:
            return to_error_response(exc)
        if isinstance(result, JSONResponse):
            return result
        if isinstance(result, AsyncIterator):
            return StreamingResponse(result, media_type="text/event-stream")
        return result

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        request: ChatCompletionRequest, raw_request: Request
    ) -> Any:
        try:
            result = await state.chat.create_chat_completion(request, raw_request)
        except Exception as exc:
            return to_error_response(exc)
        if isinstance(result, JSONResponse):
            return result
        if isinstance(result, AsyncIterator):
            return StreamingResponse(result, media_type="text/event-stream")
        return result

    @app.post("/v1/responses")
    async def create_responses(request: ResponsesRequest, raw_request: Request) -> Any:
        try:
            result = await state.responses.create_responses(request, raw_request)
        except Exception as exc:
            return to_error_response(exc)
        if isinstance(result, JSONResponse):
            return result
        if isinstance(result, AsyncIterator):
            return StreamingResponse(
                convert_stream_to_sse_events(result), media_type="text/event-stream"
            )
        return result

    @app.get("/v1/responses/{response_id}")
    async def retrieve_responses(response_id: str) -> Any:
        try:
            return await state.responses.retrieve_responses(response_id)
        except Exception as exc:
            return to_error_response(exc)

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel_responses(response_id: str) -> Any:
        try:
            return await state.responses.cancel_responses(response_id)
        except Exception as exc:
            return to_error_response(exc)

    @app.post("/v1/embeddings")
    async def create_embedding(request: EmbeddingRequest, raw_request: Request) -> Any:
        """R2.2. The vectors are synthetic; see `pvllm/pooling_params.py`."""
        try:
            return await state.embedding.create_embedding(request, raw_request)
        except Exception as exc:
            return to_error_response(exc)

    @app.get("/v1/models")
    async def show_available_models() -> Any:
        return state.models.show_available_models()

    @app.post("/tokenize")
    async def tokenize(request: TokenizeRequest) -> Any:
        return state.tokenization.tokenize(request)

    @app.post("/detokenize")
    async def detokenize(request: DetokenizeRequest) -> Any:
        return state.tokenization.detokenize(request)

    # --- operational endpoints -------------------------------------------

    @app.get("/health")
    async def health() -> Response:
        """R2.7."""
        if not await state.engine.is_ready():
            return Response(status_code=HTTPStatus.SERVICE_UNAVAILABLE.value)
        return Response(status_code=HTTPStatus.OK.value)

    @app.get("/ping")
    async def ping() -> Response:
        return Response(status_code=HTTPStatus.OK.value)

    @app.get("/version")
    async def version() -> Any:
        from pvllm import UPSTREAM_VERSION

        return {
            "version": __version__,
            "upstream_version": UPSTREAM_VERSION,
            # Stated in the payload, not just the docs: anything reading this
            # endpoint to identify the backend should learn it is a simulator.
            "simulated": True,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        """R12.1, C6."""
        await state.refresh_metrics()
        return Response(
            content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST
        )

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache() -> Any:
        """R6.10."""
        return {"success": await state.engine.reset_prefix_cache()}

    return app


def build_app_from_args(
    args: AsyncEngineArgs, enable_debug_endpoints: bool = False
) -> FastAPI:
    return build_app(
        args.create_engine_config(), enable_debug_endpoints=enable_debug_endpoints
    )
