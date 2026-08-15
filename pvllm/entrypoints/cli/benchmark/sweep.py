"""`pvllm bench sweep`.

Upstream: vllm/entrypoints/cli/benchmark/sweep.py
Tier: B
"""

from __future__ import annotations

import argparse

from pvllm.benchmarks.sweep import add_cli_args, main
from pvllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarksweepSubcommand(BenchmarkSubcommandBase):
    name = "sweep"
    help = "Sweep engine parameters and emit tidy CSV."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        return main(args)
