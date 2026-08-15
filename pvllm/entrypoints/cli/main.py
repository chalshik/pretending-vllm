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

    trace = subparsers.add_parser("trace", help="inspect a recorded trace")
    trace_sub = trace.add_subparsers(dest="trace_command", required=True)
    view = trace_sub.add_parser("view", help="render a step timeline")
    view.add_argument("path", help="path to a JSONL trace")
    view.add_argument("--format", default="text", choices=["text", "svg"])
    view.add_argument("--width", type=int, default=100, help="text columns")
    view.add_argument("-o", "--output", default=None, help="write to a file")

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

    if args.command == "trace":
        from pvllm.trace_viewer import render_svg, render_text, summarize

        summary = summarize(args.path)
        rendered = (
            render_svg(summary)
            if args.format == "svg"
            else render_text(summary, width=args.width)
        )
        if args.output:
            from pathlib import Path

            Path(args.output).write_text(rendered)
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            print(rendered)
        return 0

    print(
        f"`pvllm {args.command}` is not implemented yet; it lands in M3 with the "
        f"benchmark suite (requirement R20).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
