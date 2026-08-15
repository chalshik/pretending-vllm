"""`pvllm bench sweep`. R20, G5.

Upstream: vllm/benchmarks/sweep/param_sweep.py
Tier: B

The reason to have a simulator at all. Sweeping `max_num_seqs` across six values on
real hardware costs six engine startups and a GPU reservation; here it costs a second,
runs in CI, and is reproducible from a seed. That is G5 -- comparing configurations
cheaply -- and it is the capability the cost model's shape (rather than its accuracy)
is good enough to support.

Emits tidy CSV: one row per cell, one column per variable. Tidy rather than a matrix
because a sweep over two parameters is already a shape nobody wants to reshape by
hand, and every plotting library takes long form.

**Read the results as shape.** A sweep tells you where the knee is and which direction
a knob moves things. It does not tell you the throughput number you will get.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import sys
from dataclasses import dataclass
from typing import Any

from pvllm.benchmarks.lib.metrics import BenchmarkMetrics
from pvllm.benchmarks.lib.runner import BenchRequest, run_workload, synthetic_prompt
from pvllm.engine.arg_utils import EngineArgs
from pvllm.logger import init_logger

logger = init_logger(__name__)

#: Sweepable engine knobs, mapped to the `EngineArgs` field they set. Restricted on
#: purpose: a sweep over an arbitrary attribute name would accept typos silently and
#: report a flat line, which reads like a finding.
SWEEPABLE = {
    "max-num-seqs": "max_num_seqs",
    "max-num-batched-tokens": "max_num_batched_tokens",
    "block-size": "block_size",
    "gpu-memory-utilization": "gpu_memory_utilization",
    "device-card": "device_card",
    "enable-prefix-caching": "enable_prefix_caching",
    "enable-chunked-prefill": "enable_chunked_prefill",
    "request-rate": "__request_rate",
    "num-prompts": "__num_prompts",
    "input-len": "__input_len",
    "output-len": "__output_len",
}

#: Written for every cell. Fixed rather than derived from the metrics dict so the
#: column order is stable across runs and two CSVs can be diffed.
COLUMNS = [
    "completed",
    "duration_s",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "mean_e2el_ms",
    "mean_queue_ms",
    "num_steps",
    "num_preemptions",
    "prefix_cache_hit_rate",
    "provenance",
]


@dataclass
class Axis:
    """One swept parameter and its values."""

    name: str
    values: list[Any]


def parse_axis(spec: str) -> Axis:
    """`--sweep max-num-seqs=1,2,4,8` -> an `Axis`.

    Values are coerced by shape rather than declared: `4` becomes an int, `0.9` a
    float, `true` a bool, anything else a string. A device card name and a batch size
    have to travel through the same flag, and asking the user to declare types on the
    command line would be worse than a rule this predictable.
    """
    if "=" not in spec:
        raise ValueError(
            f"malformed sweep spec {spec!r}; expected NAME=v1,v2,v3 (for example "
            f"max-num-seqs=1,2,4)"
        )
    name, raw = spec.split("=", 1)
    name = name.strip()
    if name not in SWEEPABLE:
        raise ValueError(f"cannot sweep {name!r}; expected one of {sorted(SWEEPABLE)}")
    values = [_coerce(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError(f"sweep spec {spec!r} lists no values")
    return Axis(name=name, values=values)


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="NAME=v1,v2",
        help=(
            f"a parameter to sweep, repeatable. Sweepable: {sorted(SWEEPABLE)}. "
            f"Multiple --sweep flags take the cartesian product."
        ),
    )
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument(
        "-o",
        "--output-csv",
        default=None,
        help="where to write the tidy CSV (default: stdout)",
    )
    EngineArgs.add_cli_args(parser)


def run_cell(base: argparse.Namespace, overrides: dict[str, Any]) -> dict[str, Any]:
    """One configuration, start to finish."""
    from pvllm.benchmarks.lib.arrivals import arrival_times
    from pvllm.entrypoints.llm import LLM
    from pvllm.sim.rng import RngFactory

    settings = dict(vars(base))
    workload = {
        "input_len": base.input_len,
        "output_len": base.output_len,
        "num_prompts": base.num_prompts,
        "request_rate": base.request_rate,
    }
    for name, value in overrides.items():
        field = SWEEPABLE[name]
        if field.startswith("__"):
            workload[field[2:]] = value
        else:
            settings[field] = value

    namespace = argparse.Namespace(**settings)
    engine_args = EngineArgs.from_cli_args(namespace)
    config = engine_args.create_engine_config()
    vocab_size = config.model_config.get_vocab_size()

    rng = RngFactory(config.sim_config.seed).stream("benchmark-arrivals")
    arrivals = arrival_times(
        int(workload["num_prompts"]),
        float(workload["request_rate"]),
        rng,
        burstiness=base.burstiness,
    )
    requests = [
        BenchRequest(
            prompt_token_ids=synthetic_prompt(
                i, int(workload["input_len"]), vocab_size
            ),
            max_tokens=int(workload["output_len"]),
            arrival=arrival,
        )
        for i, arrival in enumerate(arrivals)
    ]

    engine = LLM.from_engine_args(engine_args)
    try:
        result = run_workload(engine, requests)
    finally:
        engine.shutdown()

    metrics = BenchmarkMetrics.from_finished(result.finished, result.duration)
    payload = metrics.to_dict()
    row: dict[str, Any] = {name: overrides[name] for name in overrides}
    row.update({column: payload.get(column) for column in COLUMNS if column in payload})
    row["p99_ttft_ms"] = next(
        (value for p, value in payload["percentiles_ttft_ms"] if p == 99.0), 0.0
    )
    row["num_steps"] = result.num_steps
    row["num_preemptions"] = result.num_preemptions
    row["prefix_cache_hit_rate"] = result.prefix_cache_hit_rate
    return row


def main(args: argparse.Namespace) -> int:
    if not args.sweep:
        raise ValueError(
            f"nothing to sweep. Pass at least one --sweep NAME=v1,v2 (sweepable: "
            f"{sorted(SWEEPABLE)})"
        )

    axes = [parse_axis(spec) for spec in args.sweep]
    combinations = list(itertools.product(*(axis.values for axis in axes)))
    logger.info(
        "sweeping %d cell(s) over %s",
        len(combinations),
        ", ".join(axis.name for axis in axes),
    )

    rows: list[dict[str, Any]] = []
    for index, combination in enumerate(combinations, start=1):
        overrides = {
            axis.name: value for axis, value in zip(axes, combination, strict=True)
        }
        logger.info("cell %d/%d: %s", index, len(combinations), overrides)
        rows.append(run_cell(args, overrides))

    fieldnames = [axis.name for axis in axes] + COLUMNS
    with contextlib.ExitStack() as stack:
        # `nullcontext` so stdout is not closed on the way out -- the same `with`
        # has to cover a file we own and a stream we do not.
        stream = (
            stack.enter_context(open(args.output_csv, "w", newline=""))
            if args.output_csv
            else stack.enter_context(contextlib.nullcontext(sys.stdout))
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    if args.output_csv:
        logger.info("wrote %s (%d rows)", args.output_csv, len(rows))
    return 0
