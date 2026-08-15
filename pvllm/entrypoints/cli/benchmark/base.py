"""The base class for `pvllm bench` subcommands.

Upstream: vllm/entrypoints/cli/benchmark/base.py
Tier: B

Ported nearly verbatim. The registration-by-subclass pattern is upstream's, and
keeping it means a new benchmark is added the same way in both trees.
"""

from __future__ import annotations

import argparse


class BenchmarkSubcommandBase:
    """One `pvllm bench <name>` subcommand."""

    name: str
    help: str

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        raise NotImplementedError
