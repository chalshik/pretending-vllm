"""`pvllm bench serve`.

Upstream: vllm/entrypoints/cli/benchmark/serve.py
Tier: B
"""

from __future__ import annotations

import argparse

from pvllm.benchmarks.serve import add_cli_args, main
from pvllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkserveSubcommand(BenchmarkSubcommandBase):
    name = "serve"
    help = "Benchmark serving with requests arriving at a target rate."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        return main(args)
