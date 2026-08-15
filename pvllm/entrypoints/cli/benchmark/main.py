"""The `pvllm bench` subcommand.

Upstream: vllm/entrypoints/cli/benchmark/main.py
Tier: B

Same structure as upstream: subcommand modules imported lazily and registered by
subclassing `BenchmarkSubcommandBase`. Lazy because building the benchmark parsers
pulls in numpy and the engine, and `pvllm --help` should not pay for that.
"""

from __future__ import annotations

import argparse


def _import_bench_subcommand_modules() -> None:
    import pvllm.entrypoints.cli.benchmark.latency
    import pvllm.entrypoints.cli.benchmark.serve
    import pvllm.entrypoints.cli.benchmark.sweep
    import pvllm.entrypoints.cli.benchmark.throughput  # noqa: F401


def add_bench_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Build the `bench` subparsers."""
    from pvllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase

    _import_bench_subcommand_modules()
    subparsers = parser.add_subparsers(required=True, dest="bench_type")
    for command in sorted(
        BenchmarkSubcommandBase.__subclasses__(), key=lambda c: c.name
    ):
        subparser = subparsers.add_parser(
            command.name,
            help=command.help,
            description=command.help,
            usage=f"pvllm bench {command.name} [options]",
        )
        subparser.set_defaults(dispatch_function=command.cmd)
        command.add_cli_args(subparser)
    return parser


def run_bench(args: argparse.Namespace) -> int:
    from pvllm.logger import init_logger

    logger = init_logger(__name__)
    logger.warning(
        "Benchmark durations are MODELED by the simulated cost model, not measured "
        "(R9.5). They reproduce qualitative regimes -- where the knee is, which way "
        "a knob moves things -- and will not tell you your p99."
    )
    result = args.dispatch_function(args)
    return int(result) if result is not None else 0
