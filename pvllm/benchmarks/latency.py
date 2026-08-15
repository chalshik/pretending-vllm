"""`pvllm bench latency`. R20.

Upstream: vllm/benchmarks/latency.py
Tier: B

One fixed batch, run to completion, repeated. Answers "how long does a batch of this
shape take", which is the number you tune `max_num_batched_tokens` against.

Upstream repeats the batch 30 times and reports percentiles, because real hardware is
noisy. Here the run is deterministic unless `--jitter-sigma` is set, so repeating an
identical batch produces identical numbers -- the default iteration count is 3 rather
than 30, and the percentile spread is honest about being zero when there is no jitter.
Asking for 30 iterations of a noiseless model is 27 iterations of nothing.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from pvllm.benchmarks.lib.metrics import BenchmarkMetrics
from pvllm.benchmarks.lib.runner import BenchRequest, run_workload, synthetic_prompt
from pvllm.engine.arg_utils import EngineArgs
from pvllm.logger import init_logger

logger = init_logger(__name__)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-iters",
        type=int,
        default=3,
        help=(
            "iterations to run. The default is low because a run without "
            "--jitter-sigma is deterministic, so extra iterations repeat the same "
            "number."
        ),
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=0,
        help=(
            "warmup iterations. Zero by default: there are no caches to warm and no "
            "kernels to autotune, so a warmup here only warms the prefix cache -- "
            "which would make the measured iterations faster than the first, for a "
            "reason that has nothing to do with what is being measured."
        ),
    )
    parser.add_argument("--output-json", default=None, help="write results as JSON")
    EngineArgs.add_cli_args(parser)
    # Upstream disables prefix caching here for the same reason: identical batches
    # would hit the cache and report a decode-only latency for a prefill workload.
    parser.set_defaults(enable_prefix_caching=False)


def main(args: argparse.Namespace) -> int:
    engine_args = EngineArgs.from_cli_args(args)
    config = engine_args.create_engine_config()
    max_model_len = config.model_config.max_model_len
    assert max_model_len is not None

    if args.input_len + args.output_len > max_model_len:
        raise ValueError(
            f"input_len + output_len ({args.input_len} + {args.output_len}) exceeds "
            f"max_model_len ({max_model_len}); raise --max-model-len or shorten the "
            f"request"
        )

    from pvllm.entrypoints.llm import LLM

    vocab_size = config.model_config.get_vocab_size()
    engine = LLM.from_engine_args(engine_args)

    def one_iteration(offset: int) -> Any:
        requests = [
            BenchRequest(
                prompt_token_ids=synthetic_prompt(
                    offset * args.batch_size + i, args.input_len, vocab_size
                ),
                max_tokens=args.output_len,
            )
            for i in range(args.batch_size)
        ]
        return run_workload(engine, requests)

    for iteration in range(args.num_iters_warmup):
        one_iteration(-1 - iteration)

    durations: list[float] = []
    last = None
    for iteration in range(args.num_iters):
        last = one_iteration(iteration)
        durations.append(last.duration)

    assert last is not None
    metrics = BenchmarkMetrics.from_finished(last.finished, last.duration)
    engine.shutdown()

    print(
        metrics.render(
            f"Latency: batch={args.batch_size} in={args.input_len} "
            f"out={args.output_len}"
        )
    )
    print(
        f"{'Per-iteration modeled duration (s):':<42} "
        f"{[round(d, 4) for d in durations]}"
    )
    print(f"{'Engine steps:':<42} {last.num_steps:>19}")

    if args.output_json:
        payload = metrics.to_dict()
        payload.update(
            batch_size=args.batch_size,
            input_len=args.input_len,
            output_len=args.output_len,
            iteration_durations_s=durations,
            num_steps=last.num_steps,
        )
        with open(args.output_json, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("wrote %s", args.output_json)

    return 0
