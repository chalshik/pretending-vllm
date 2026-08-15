"""`pvllm serve`.

Upstream: vllm/entrypoints/cli/serve.py
Tier: B

R2.6.
"""

from __future__ import annotations

import argparse

from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.logger import init_logger

logger = init_logger(__name__)


def add_serve_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--uvicorn-log-level", default="info")
    AsyncEngineArgs.add_cli_args(parser)
    return parser


def run_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from pvllm.entrypoints.openai.api_server import build_app

    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()

    logger.info("Starting pretending-vllm server: %s", vllm_config)
    logger.warning(
        "This is a SIMULATOR. Generated text is synthetic and latency figures are "
        "modeled, not measured. See the fidelity contract in the README."
    )

    app = build_app(vllm_config)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.uvicorn_log_level)
