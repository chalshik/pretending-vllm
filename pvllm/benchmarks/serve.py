"""`pvllm bench serve`. R20.

Upstream: vllm/benchmarks/serve.py
Tier: B

Requests arriving as a stochastic process at a target rate, which is the only one of
the three benchmarks that measures *queueing* -- and queueing is what a capacity
decision actually turns on. TTFT under load is a queueing number, not a compute one.

**In process, not over HTTP, and that is not a shortcut.** Upstream's `bench serve`
points at a running server and times the responses with a wall clock. Doing that here
would measure the wall-clock behavior of a server whose durations are modeled: under
the default virtual clock, every response returns as fast as Python can produce it, so
the measured TTFT would be a property of the host machine and nothing else. Driving
the engine directly means the reported latencies come from the modeled timeline, which
is the only timeline in this system that means anything.

`--base-url` therefore raises rather than quietly measuring the wrong thing. If you
genuinely want to time a live pvllm server end to end -- to test your own client's
timeout handling, say -- run `pvllm serve --clock-mode real` and point a real load
generator at it. That configuration spends the modeled durations, so a wall clock
observes them.
"""

from __future__ import annotations

import argparse
import json

from pvllm.benchmarks.lib.arrivals import arrival_times
from pvllm.benchmarks.lib.metrics import BenchmarkMetrics
from pvllm.benchmarks.lib.runner import BenchRequest, run_workload, synthetic_prompt
from pvllm.engine.arg_utils import EngineArgs
from pvllm.logger import init_logger

logger = init_logger(__name__)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help=(
            "requests per second. `inf` submits everything at once, which measures "
            "throughput rather than queueing -- set a finite rate to see TTFT move."
        ),
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help=(
            "gamma shape for the arrival gaps. 1.0 is Poisson; below 1 is burstier, "
            "which is what actually stresses a scheduler; inf is uniform spacing."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=argparse.SUPPRESS,  # see the module docstring; accepted only to refuse
    )
    parser.add_argument("--output-json", default=None, help="write results as JSON")
    EngineArgs.add_cli_args(parser)


def main(args: argparse.Namespace) -> int:
    if args.base_url:
        raise NotImplementedError(
            "`pvllm bench serve --base-url` is not supported. Timing a virtual-clock "
            "server with a wall clock measures the host machine, not the modeled "
            "engine -- see pvllm/benchmarks/serve.py. Drop --base-url to benchmark "
            "in process against modeled time, or run `pvllm serve --clock-mode real` "
            "and point a real load generator at it."
        )

    engine_args = EngineArgs.from_cli_args(args)
    config = engine_args.create_engine_config()
    max_model_len = config.model_config.max_model_len
    assert max_model_len is not None

    if args.input_len + args.output_len > max_model_len:
        raise ValueError(
            f"input_len + output_len ({args.input_len} + {args.output_len}) exceeds "
            f"max_model_len ({max_model_len})"
        )

    from pvllm.entrypoints.llm import LLM
    from pvllm.sim.rng import RngFactory

    # Seeded from the run seed, so a benchmark rerun is the same benchmark. Upstream
    # draws from global numpy state, which is fine when the noise floor is real
    # hardware and wrong when the rest of the system is exactly reproducible.
    rng = RngFactory(config.sim_config.seed).stream("benchmark-arrivals")
    arrivals = arrival_times(
        args.num_prompts, args.request_rate, rng, burstiness=args.burstiness
    )

    vocab_size = config.model_config.get_vocab_size()
    requests = [
        BenchRequest(
            prompt_token_ids=synthetic_prompt(i, args.input_len, vocab_size),
            max_tokens=args.output_len,
            arrival=arrival,
        )
        for i, arrival in enumerate(arrivals)
    ]

    engine = LLM.from_engine_args(engine_args)
    result = run_workload(engine, requests)
    metrics = BenchmarkMetrics.from_finished(result.finished, result.duration)
    engine.shutdown()

    rate = "inf" if args.request_rate == float("inf") else f"{args.request_rate:g}/s"
    print(metrics.render(f"Serving: {args.num_prompts} requests at {rate}"))
    print(f"{'Requested rate (req/s):':<42} {rate:>19}")
    print(f"{'Burstiness:':<42} {args.burstiness:>19g}")
    print(f"{'Engine steps:':<42} {result.num_steps:>19}")
    print(f"{'Preemptions:':<42} {result.num_preemptions:>19}")

    if args.output_json:
        payload = metrics.to_dict()
        payload.update(
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            num_steps=result.num_steps,
            num_preemptions=result.num_preemptions,
        )
        with open(args.output_json, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("wrote %s", args.output_json)

    return 0
