"""Trace emission and the timeline viewer. R19.3, R19.4."""

from __future__ import annotations

import pathlib

import pytest

from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.trace_viewer import (
    DECODE,
    IDLE,
    PREEMPTED,
    PREFILL_LARGE,
    render_svg,
    render_text,
    summarize,
)
from pvllm.tracing import read_trace

BASE = {
    "max_model_len": 48,
    "block_size": 8,
    "max_num_batched_tokens": 24,
    "device_card": "tiny-2gb",
    "num_gpu_blocks_override": 24,
    "disable_log_stats": True,
}


def run_traced(tmp_path, num_requests: int = 3, **overrides) -> str:
    path = str(tmp_path / "trace.jsonl")
    engine = LLM("tiny-test", trace_path=path, **{**BASE, **overrides})
    engine.generate(
        [f"shared prefix. question {i}" for i in range(num_requests)],
        SamplingParams(max_tokens=4),
    )
    engine.shutdown()
    return path


# --- emission (R19.3) ------------------------------------------------------


def test_a_run_produces_a_self_describing_trace(tmp_path):
    records = list(read_trace(run_traced(tmp_path)))
    header = records[0]
    assert header["type"] == "header"
    # The pin travels with the trace, so a golden trace can never be silently
    # compared across upstream versions.
    assert header["upstream_version"] == "0.27.1"
    assert header["config"]["model_card"] == "tiny-test"


def test_every_step_record_is_stamped(tmp_path):
    """A record without a timestamp is useless for what traces exist for. The
    scheduler builds these; only the engine core can date them (R19.1)."""
    steps = [r for r in read_trace(run_traced(tmp_path)) if r["type"] == "step"]
    assert steps
    assert all("t" in step for step in steps)
    assert all(step["t"] >= steps[0]["t"] for step in steps)


def test_lifecycle_transitions_are_recorded(tmp_path):
    """R19.3: one record per transition, so a trace answers 'when did this finish,
    and why' without replaying the step stream."""
    events = [r for r in read_trace(run_traced(tmp_path)) if r["type"] == "request"]
    by_event: dict[str, int] = {}
    for record in events:
        by_event[record["event"]] = by_event.get(record["event"], 0) + 1

    assert by_event["arrived"] == 3
    assert by_event["finished"] == 3
    # A request the scheduler finished must not also be reported as aborted.
    assert "aborted" not in by_event


def test_finished_records_carry_the_reason(tmp_path):
    finished = [
        r
        for r in read_trace(run_traced(tmp_path))
        if r["type"] == "request" and r["event"] == "finished"
    ]
    assert all(r["finish_reason"] == "length" for r in finished)


def test_step_records_carry_scheduling_and_cache_state(tmp_path):
    steps = [r for r in read_trace(run_traced(tmp_path)) if r["type"] == "step"]
    first = steps[0]
    for field in (
        "num_scheduled_tokens",
        "total_num_scheduled_tokens",
        "num_running",
        "num_waiting",
        "kv_usage",
        "prefix_cache_hits",
        "waiting_req_ids",
    ):
        assert field in first, f"missing {field}"


def test_two_runs_produce_identical_traces(tmp_path):
    """B4, at the artifact level -- which is what makes C1--C4 comparable at all."""
    first = (tmp_path / "a.jsonl").as_posix()
    second = (tmp_path / "b.jsonl").as_posix()

    def run(path: str) -> list[dict]:
        engine = LLM("tiny-test", trace_path=path, seed=5, **BASE)
        engine.generate(["one", "two"], SamplingParams(max_tokens=3))
        engine.shutdown()
        return list(read_trace(path))

    assert run(first) == run(second)


# --- the viewer (R19.4) ----------------------------------------------------


def test_the_timeline_has_a_row_per_request(tmp_path):
    summary = summarize(run_traced(tmp_path, num_requests=4))
    assert len(summary.request_ids) == 4
    assert summary.num_steps > 0


def test_prefill_and_decode_are_distinguishable(tmp_path):
    summary = summarize(run_traced(tmp_path))
    glyphs = {g for row in summary.rows.values() for g in row.values()}
    assert DECODE in glyphs
    assert PREFILL_LARGE in glyphs or ":" in glyphs


def test_queueing_is_visible(tmp_path):
    """A count of waiting requests cannot show *which* one waited, which is the
    thing a timeline is opened to find."""
    summary = summarize(run_traced(tmp_path, num_requests=4, max_num_seqs=1))
    assert any(IDLE in row.values() for row in summary.rows.values())


def test_preemption_survives_column_compression(tmp_path):
    """A preemption averaged away by compression hides the event the picture was
    opened to find."""
    from pvllm.trace_viewer import _most_significant

    assert _most_significant([DECODE, PREEMPTED, DECODE]) == PREEMPTED
    assert _most_significant([IDLE, DECODE]) == DECODE
    assert _most_significant([None, None]) == " "


def test_text_rendering_includes_the_legend_and_totals(tmp_path):
    text = render_text(summarize(run_traced(tmp_path)))
    assert "legend:" in text
    assert "steps=" in text
    assert "prefix cache:" in text
    assert "upstream 0.27.1" in text


def test_text_rendering_compresses_a_long_run():
    """A run with more steps than columns is bucketed, so a thousand-step trace
    still fits a terminal."""
    from pvllm.trace_viewer import TraceSummary

    summary = TraceSummary(header={})
    summary.num_steps = 500
    summary.rows = {"r0": dict.fromkeys(range(1, 501), DECODE)}
    summary.kv_usage = [0.5]

    rendered = render_text(summary, width=80)
    row = next(line for line in rendered.splitlines() if line.startswith("  r0"))
    assert len(row) <= 80
    assert "each column is" in rendered


def test_svg_rendering_is_well_formed(tmp_path):
    svg = render_svg(summarize(run_traced(tmp_path)))
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "<rect" in svg


def test_an_empty_trace_renders_without_crashing(tmp_path):
    path = tmp_path / "empty.jsonl"
    engine = LLM("tiny-test", trace_path=str(path), **BASE)
    engine.shutdown()
    assert "(no requests)" in render_text(summarize(path))


def test_the_cli_renders_a_trace(tmp_path, capsys):
    from pvllm.entrypoints.cli.main import main

    path = run_traced(tmp_path)
    assert main(["trace", "view", path]) == 0
    assert "legend:" in capsys.readouterr().out


def test_the_cli_writes_svg_to_a_file(tmp_path):
    from pvllm.entrypoints.cli.main import main

    path = run_traced(tmp_path)
    out = tmp_path / "timeline.svg"
    assert main(["trace", "view", path, "--format", "svg", "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<svg")


def test_a_truncated_trace_is_reported_as_broken(tmp_path):
    """Not as a behavioural difference: a conformance diff must distinguish
    'records were lost' from 'the engine did something else'."""
    original = (
        pathlib.Path(run_traced(tmp_path))
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )

    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text("".join(original[:2] + original[3:]), encoding="utf-8")

    with pytest.raises(ValueError, match="trace discontinuity"):
        summarize(truncated)
