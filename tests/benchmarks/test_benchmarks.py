"""The benchmark CLI end to end. R20, G5.

Two things worth testing here, and they are different in kind.

The first is mechanical: the commands run, the JSON has the fields upstream's has,
`--output-csv` is well formed. Those catch breakage.

The second is the interesting one. A benchmark suite over a *simulator* is only worth
running if the numbers move the way a real system's would -- R9's qualitative regimes.
Those assertions are at the bottom, and they are the ones that would fail if the cost
model were rewired wrongly. They assert direction and ordering, never magnitude: the
model is uncalibrated, and a test pinning a millisecond count would be pinning a
number nobody should trust.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pvllm.entrypoints.cli.main import main

#: Shared by every invocation. `tiny-test` on `tiny-2gb` keeps the whole file inside
#: R21.5's budget for the suite.
BASE = [
    "--model",
    "tiny-test",
    "--device-card",
    "tiny-2gb",
    "--max-model-len",
    "256",
    "--disable-log-stats",
]


def run(subcommand: str, *extra: str) -> int:
    """Invoke `pvllm bench <subcommand>` with the shared engine flags."""
    return main(["bench", subcommand, *BASE, *extra])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# --- the commands run ------------------------------------------------------


def test_latency_reports_a_modeled_duration(tmp_path, capsys):
    out = tmp_path / "latency.json"
    assert (
        run(
            "latency",
            "--input-len",
            "32",
            "--output-len",
            "8",
            "--batch-size",
            "4",
            "--num-iters",
            "2",
            "--output-json",
            str(out),
        )
        == 0
    )
    assert "MODELED" in capsys.readouterr().out

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["completed"] == 4
    assert payload["total_output_tokens"] == 32
    assert payload["duration_s"] > 0
    assert len(payload["iteration_durations_s"]) == 2
    # R9.5: the label rides with the numbers, not just the terminal output. A JSON
    # file outlives the console it was printed next to.
    assert payload["provenance"] == "modeled"


def test_latency_iterations_repeat_exactly_without_jitter(tmp_path):
    """B4. It is also why the default iteration count is 3 and not upstream's 30:
    without jitter there is nothing for the extra runs to average out."""
    out = tmp_path / "latency.json"
    run(
        "latency",
        "--input-len",
        "32",
        "--output-len",
        "8",
        "--batch-size",
        "2",
        "--num-iters",
        "3",
        "--output-json",
        str(out),
    )
    durations = json.loads(out.read_text(encoding="utf-8"))["iteration_durations_s"]
    assert len(set(durations)) == 1, durations


def test_throughput_reports_upstream_field_names(tmp_path):
    """A comparison notebook written against vLLM's benchmark JSON should parse
    ours without a branch."""
    out = tmp_path / "throughput.json"
    assert (
        run(
            "throughput",
            "--input-len",
            "64",
            "--output-len",
            "8",
            "--num-prompts",
            "8",
            "--output-json",
            str(out),
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    for field in (
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "percentiles_ttft_ms",
        "mean_tpot_ms",
        "mean_e2el_ms",
    ):
        assert field in payload, field


def test_serve_honours_the_request_rate(tmp_path):
    """The arrival schedule has to reach the engine, not just be computed. If
    everything were submitted at once the run would finish far sooner than the
    arrivals alone imply."""
    out = tmp_path / "serve.json"
    assert (
        run(
            "serve",
            "--input-len",
            "32",
            "--output-len",
            "4",
            "--num-prompts",
            "12",
            "--request-rate",
            "20",
            "--output-json",
            str(out),
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    # Eleven gaps at ~20/s is ~0.55s of arrivals alone, and the run cannot be
    # shorter than the schedule that produced it.
    assert payload["duration_s"] > 0.4
    assert payload["completed"] == 12


def test_serve_refuses_to_time_a_live_server():
    """Unsupported-path discipline: measuring a virtual-clock server with a wall
    clock would produce a plausible number describing the host machine."""
    with pytest.raises(NotImplementedError, match="virtual-clock server"):
        run("serve", "--num-prompts", "2", "--base-url", "http://localhost:8000")


def test_a_request_longer_than_the_model_is_refused():
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        run("latency", "--input-len", "200", "--output-len", "100")


# --- the sweep -------------------------------------------------------------


def test_sweep_writes_one_tidy_row_per_cell(tmp_path):
    out = tmp_path / "sweep.csv"
    assert (
        run(
            "sweep",
            "--input-len",
            "32",
            "--output-len",
            "8",
            "--num-prompts",
            "8",
            "--sweep",
            "max-num-seqs=1,2,4",
            "-o",
            str(out),
        )
        == 0
    )

    rows = read_rows(out)
    assert len(rows) == 3
    assert [row["max-num-seqs"] for row in rows] == ["1", "2", "4"]
    assert all(row["provenance"] == "modeled" for row in rows)


def test_sweep_takes_the_product_of_its_axes(tmp_path):
    out = tmp_path / "sweep.csv"
    run(
        "sweep",
        "--input-len",
        "32",
        "--output-len",
        "4",
        "--num-prompts",
        "4",
        "--sweep",
        "max-num-seqs=1,2",
        "--sweep",
        "block-size=8,16",
        "-o",
        str(out),
    )
    rows = read_rows(out)
    assert len(rows) == 4
    assert {(row["max-num-seqs"], row["block-size"]) for row in rows} == {
        ("1", "8"),
        ("1", "16"),
        ("2", "8"),
        ("2", "16"),
    }


def test_an_unsweepable_parameter_is_refused():
    """Rather than accepting a typo and reporting a flat line, which reads like a
    finding: "max_num_seq made no difference"."""
    with pytest.raises(ValueError, match="cannot sweep"):
        run("sweep", "--sweep", "max-num-seq=1,2")


def test_a_sweep_with_no_axes_is_refused():
    with pytest.raises(ValueError, match="nothing to sweep"):
        run("sweep")


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("max-num-seqs=1,2", [1, 2]),
        ("gpu-memory-utilization=0.8,0.9", [0.8, 0.9]),
        ("enable-prefix-caching=true,false", [True, False]),
        ("device-card=tiny-2gb", ["tiny-2gb"]),
    ],
)
def test_sweep_values_are_coerced_by_shape(spec, expected):
    from pvllm.benchmarks.sweep import parse_axis

    assert parse_axis(spec).values == expected


# --- R9's regimes, which are the reason to run any of this -----------------


@pytest.mark.parametrize("metric", ["request_throughput", "output_throughput"])
def test_throughput_rises_with_concurrency(tmp_path, metric):
    """R9's saturation regime, and the single most common reason to sweep."""
    out = tmp_path / "sweep.csv"
    run(
        "sweep",
        "--input-len",
        "32",
        "--output-len",
        "8",
        "--num-prompts",
        "16",
        "--sweep",
        "max-num-seqs=1,2,4,8",
        "-o",
        str(out),
    )
    values = [float(row[metric]) for row in read_rows(out)]
    assert values == sorted(values), values
    assert values[-1] > values[0] * 2


def test_queueing_dominates_ttft_at_low_concurrency(tmp_path):
    """The number a capacity decision turns on, and the reason queue time is
    separated from prefill at all: at `max_num_seqs=1` almost all of TTFT is waiting,
    and the answer to that is more concurrency, not a smaller batch."""
    out = tmp_path / "sweep.csv"
    run(
        "sweep",
        "--input-len",
        "32",
        "--output-len",
        "8",
        "--num-prompts",
        "16",
        "--sweep",
        "max-num-seqs=1,8",
        "-o",
        str(out),
    )
    starved, roomy = read_rows(out)

    assert float(starved["mean_queue_ms"]) > float(roomy["mean_queue_ms"])
    assert float(starved["mean_queue_ms"]) > 0.5 * float(starved["mean_ttft_ms"])


def test_a_shared_prefix_raises_the_hit_rate_and_cuts_the_work(tmp_path):
    """G5's headline comparison: does prefix caching help this workload, and by how
    much. Answerable in a second here and only on a GPU otherwise."""
    shared = tmp_path / "shared.json"
    distinct = tmp_path / "distinct.json"
    common = (
        "--input-len",
        "96",
        "--output-len",
        "8",
        "--num-prompts",
        "12",
        # The budget has to bind for the saving to be visible. With an unconstrained
        # max_num_batched_tokens every prompt prefills in the same single step
        # whether or not it hit the cache -- the cache saves *tokens*, and tokens
        # only become steps once the budget is the thing rationing them.
        "--max-num-batched-tokens",
        "64",
    )
    run(
        "throughput", *common, "--shared-prefix-len", "64", "--output-json", str(shared)
    )
    run(
        "throughput",
        *common,
        "--shared-prefix-len",
        "0",
        "--output-json",
        str(distinct),
    )

    with_prefix = json.loads(shared.read_text(encoding="utf-8"))
    without = json.loads(distinct.read_text(encoding="utf-8"))

    assert with_prefix["prefix_cache_hit_rate"] > without["prefix_cache_hit_rate"]
    # Asserted on steps rather than on duration because steps are exact and duration
    # is modeled.
    assert with_prefix["num_steps"] < without["num_steps"]
