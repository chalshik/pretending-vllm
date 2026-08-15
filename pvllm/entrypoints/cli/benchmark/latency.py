"""`pvllm bench latency`.

Upstream: vllm/entrypoints/cli/benchmark/latency.py
Tier: B
"""

from __future__ import annotations

import argparse

from pvllm.benchmarks.latency import add_cli_args, main
from pvllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarklatencySubcommand(BenchmarkSubcommandBase):
    name = "latency"
    help = "Benchmark the latency of a single batch of requests."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        return main(args)
