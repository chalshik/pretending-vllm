"""The `pvllm` command.

Upstream: vllm/entrypoints/cli/main.py
Tier: B

R2.6. `serve`, `trace`, and `bench` work; `complete` is not implemented and says so
rather than existing as an empty command.
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

    # Built only when it is being invoked: constructing the benchmark parsers
    # imports the engine and numpy, and `pvllm --help` should not pay for that.
    # Upstream defers the same way, for the same reason.
    argv = sys.argv[1:]
    if next((arg for arg in argv if not arg.startswith("-")), None) == "bench":
        from pvllm.entrypoints.cli.benchmark.main import add_bench_args

        add_bench_args(
            subparsers.add_parser(
                "bench",
                help="benchmark latency, throughput, serving, or a parameter sweep",
                usage="pvllm bench <bench_type> [options]",
            )
        )
    else:
        subparsers.add_parser(
            "bench", help="benchmark latency, throughput, serving, or a sweep"
        ).add_argument("rest", nargs=argparse.REMAINDER)

    deferred = subparsers.add_parser(
        "complete", help="not yet implemented (R2.6, interactive completion)"
    )
    deferred.add_argument("rest", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] == "bench":
        # `build_parser` decides whether to build the bench subparsers from
        # `sys.argv`, which is not what an in-process caller passed. Build them
        # directly here so `main(["bench", ...])` works from a test or a script.
        from pvllm.entrypoints.cli.benchmark.main import add_bench_args, run_bench

        parser = argparse.ArgumentParser(prog="pvllm bench")
        return run_bench(add_bench_args(parser).parse_args(argv[1:]))

    args = build_parser().parse_args(argv)

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
        f"`pvllm {args.command}` is not implemented yet (requirement R2.6).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
