"""The `/debug/*` routes. D9.

Upstream: vllm/entrypoints/serve/dev/server_info/api_router.py
Tier: B

Upstream's `serve/dev/` tree holds routers that are attached conditionally --
`server_info`, `sleep`, `rlhf`, `rpc`, gated on `VLLM_SERVER_DEV_MODE`. This mirrors
that shape: an `APIRouter` plus `attach_router(app)`, attached only when the server was
started with `--enable-debug-endpoints`.

**Why a flag at all, for a simulator.** These responses contain prompt token ids and
per-request state. Nothing here is dangerous on a laptop driving a test double, but a
pvllm instance in a shared staging environment is still answering questions about
whoever's traffic is flowing through it. Matching upstream's gate costs one flag and
means the default is the safe one.

Every route is a GET and reports state. There is no debug route that changes engine
behavior -- `/reset_prefix_cache` already exists on the main app precisely because it
is a mutation and does not belong here.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from pvllm.entrypoints.serve.dev.introspect import EngineIntrospector
from pvllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


def _introspector(raw_request: Request) -> EngineIntrospector:
    return raw_request.app.state.introspector  # type: ignore[no-any-return]


@router.get("/scheduler")
async def scheduler_state(raw_request: Request) -> Any:
    """What the scheduler is doing, right now.

    The running and waiting lists are in scheduler order, so the answer to "why is my
    request not running yet" is the position of its id in `waiting`.
    """
    return JSONResponse(content=_introspector(raw_request).scheduler_state())


@router.get("/requests")
async def request_states(raw_request: Request) -> Any:
    """Every tracked request, counted by state. R5.1."""
    introspector = _introspector(raw_request)
    return JSONResponse(
        content={
            "counts": introspector.status_counts(),
            "request_ids": sorted(introspector.request_ids()),
        }
    )


@router.get("/requests/{request_id}")
async def request_state(raw_request: Request, request_id: str) -> Any:
    """One request's full state, including its block table.

    404 rather than an empty body for an unknown id: a request that finished and was
    freed is genuinely gone from the engine, and saying so is more useful than
    returning nulls that read like a live request with no tokens.
    """
    state = _introspector(raw_request).request_state(request_id)
    if state is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"no request {request_id!r} is tracked by the engine; it "
                f"either never arrived or has finished and been freed"
            },
        )
    return JSONResponse(content=state)


@router.get("/blocks")
async def block_pool_state(
    raw_request: Request,
    limit: Annotated[int, Query(ge=0, le=4096)] = 64,
) -> Any:
    """The block pool, and which requests hold which blocks. R6.2."""
    return JSONResponse(content=_introspector(raw_request).block_pool_state(limit))


@router.get("/prefix_cache")
async def prefix_cache_state(raw_request: Request) -> Any:
    """Prefix cache effectiveness, in tokens. R6.9."""
    return JSONResponse(content=_introspector(raw_request).prefix_cache_state())


@router.get("/cost_model")
async def step_costs(
    raw_request: Request,
    limit: Annotated[int, Query(ge=1, le=256)] = 16,
) -> Any:
    """The cost-model breakdown for recent steps. Modeled, not measured (R9.5)."""
    return JSONResponse(content=_introspector(raw_request).step_costs(limit))


@router.get("/config")
async def config_dump(raw_request: Request) -> Any:
    """The fully resolved config, including everything being simulated.

    What a product should read when its own numbers look wrong: `num_gpu_blocks` and
    `max_concurrency` here are *derived* from the device card and the memory model,
    not configured, and are usually the surprise.
    """
    return JSONResponse(content=_introspector(raw_request).config_dump())


def attach_router(app: FastAPI) -> None:
    """Attach the debug routes. Called only when the flag is set."""
    app.include_router(router)
    logger.warning(
        "debug endpoints enabled at /debug/* -- these expose prompt token ids and "
        "per-request state, and are read-only"
    )
