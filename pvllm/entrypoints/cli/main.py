"""The `pvllm` command.

Upstream: vllm/entrypoints/cli/main.py
Tier: B

R2.6. `serve` works now; `bench` and `complete` land in M3 with the benchmark suite
(R20), and say so rather than existing as empty commands.
"""

from __future__ import annotations

import argparse
import sys

from pvllm import UPSTREAM_VERSION, __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pvllm",
        description=(
            "pretending-vllm: a structurally faithful reimplementation of the vLLM "
            "V1 engine with a simulated device and model. Generated text is "
            "synthetic; latency is modeled, not measured."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"pretending-vllm {__version__} (mirrors vLLM {UPSTREAM_VERSION})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from pvllm.entrypoints.cli.serve import add_serve_args

    add_serve_args(subparsers.add_parser("serve", help="run the OpenAI API server"))

    for name, requirement in (
        ("bench", "R20, benchmarks"),
        ("complete", "R2.6, interactive completion"),
    ):
        deferred = subparsers.add_parser(
            name, help=f"not yet implemented ({requirement})"
        )
        deferred.add_argument("rest", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "serve":
        from pvllm.entrypoints.cli.serve import run_serve

        run_serve(args)
        return 0

    print(
        f"`pvllm {args.command}` is not implemented yet; it lands in M3 with the "
        f"benchmark suite (requirement R20).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
