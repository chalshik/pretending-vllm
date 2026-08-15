"""`pvllm bench throughput`. R20.

Upstream: vllm/benchmarks/throughput.py
Tier: B

Every request submitted at once, drained as fast as the engine can. Answers "what is
the ceiling", which is the number a capacity plan is built on -- and, unlike latency,
the one where the qualitative shape of the cost model is most likely to be right:
throughput saturates with batch size for structural reasons the roofline model
reproduces, not because any constant was calibrated well.
"""

from __future__ import annotations

import argparse
import json

from pvllm.benchmarks.lib.metrics import BenchmarkMetrics
from pvllm.benchmarks.lib.runner import BenchRequest, run_workload, synthetic_prompt
from pvllm.engine.arg_utils import EngineArgs
from pvllm.logger import init_logger

logger = init_logger(__name__)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument(
        "--shared-prefix-len",
        type=int,
        default=0,
        help=(
            "give every prompt this many identical leading tokens. The knob that "
            "turns a throughput benchmark into a prefix-cache benchmark, which is "
            "the comparison most worth running (G5)."
        ),
    )
    parser.add_argument("--output-json", default=None, help="write results as JSON")
    EngineArgs.add_cli_args(parser)


def main(args: argparse.Namespace) -> int:
    engine_args = EngineArgs.from_cli_args(args)
    config = engine_args.create_engine_config()
    max_model_len = config.model_config.max_model_len
    assert max_model_len is not None

    if args.input_len + args.output_len > max_model_len:
        raise ValueError(
            f"input_len + output_len ({args.input_len} + {args.output_len}) exceeds "
            f"max_model_len ({max_model_len})"
        )
    if args.shared_prefix_len > args.input_len:
        raise ValueError(
            f"shared_prefix_len ({args.shared_prefix_len}) cannot exceed input_len "
            f"({args.input_len})"
        )

    from pvllm.entrypoints.llm import LLM

    vocab_size = config.model_config.get_vocab_size()
    shared = (
        synthetic_prompt(0, args.shared_prefix_len, vocab_size)
        if (args.shared_prefix_len)
        else []
    )

    requests = [
        BenchRequest(
            prompt_token_ids=(
                shared
                + synthetic_prompt(
                    i + 1, args.input_len - args.shared_prefix_len, vocab_size
                )
            ),
            max_tokens=args.output_len,
        )
        for i in range(args.num_prompts)
    ]

    engine = LLM.from_engine_args(engine_args)
    result = run_workload(engine, requests)
    metrics = BenchmarkMetrics.from_finished(result.finished, result.duration)
    engine.shutdown()

    print(
        metrics.render(
            f"Throughput: {args.num_prompts} prompts, "
            f"in={args.input_len} out={args.output_len}"
        )
    )
    print(f"{'Engine steps:':<42} {result.num_steps:>19}")
    print(f"{'Preemptions:':<42} {result.num_preemptions:>19}")
    print(f"{'Prefix cache hit rate:':<42} {result.prefix_cache_hit_rate:>18.1%}")

    if args.output_json:
        payload = metrics.to_dict()
        payload.update(
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
            shared_prefix_len=args.shared_prefix_len,
            num_steps=result.num_steps,
            num_preemptions=result.num_preemptions,
            prefix_cache_hit_rate=result.prefix_cache_hit_rate,
        )
        with open(args.output_json, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("wrote %s", args.output_json)

    return 0
