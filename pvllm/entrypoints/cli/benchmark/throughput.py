"""`pvllm bench throughput`.

Upstream: vllm/entrypoints/cli/benchmark/throughput.py
Tier: B
"""

from __future__ import annotations

import argparse

from pvllm.benchmarks.throughput import add_cli_args, main
from pvllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkthroughputSubcommand(BenchmarkSubcommandBase):
    name = "throughput"
    help = "Benchmark offline throughput with all requests submitted at once."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        return main(args)
