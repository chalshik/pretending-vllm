"""JSONL event trace. R19.3."""

from __future__ import annotations

import io

import pytest

from pvllm.sim.trace import (
    TRACE_SCHEMA_VERSION,
    NullTraceWriter,
    TraceWriter,
    read_header,
    read_trace,
)


def _writer(path, **kwargs):
    kwargs.setdefault("seed", 42)
    kwargs.setdefault("clock_mode", "virtual")
    kwargs.setdefault("upstream_version", "0.27.1")
    return TraceWriter(path, **kwargs)


def test_header_makes_a_trace_self_describing(tmp_path):
    path = tmp_path / "t.jsonl"
    with _writer(path, config={"block_size": 16}):
        pass

    header = read_header(path)
    assert header["type"] == "header"
    assert header["schema_version"] == TRACE_SCHEMA_VERSION
    assert header["seed"] == 42
    assert header["clock_mode"] == "virtual"
    # The pin travels with the trace, so a golden trace can never be silently
    # compared across upstream versions.
    assert header["upstream_version"] == "0.27.1"
    assert header["config"] == {"block_size": 16}


def test_round_trip_preserves_fields(tmp_path):
    path = tmp_path / "t.jsonl"
    with _writer(path) as writer:
        writer.emit("request", t=1.0, request_id="r0", event="arrived")
        writer.emit("step", t=1.5, step=0, scheduled={"r0": 128}, kv_usage=0.25)

    records = list(read_trace(path))
    assert [r["type"] for r in records] == ["header", "request", "step"]
    assert records[1]["request_id"] == "r0"
    assert records[2]["scheduled"] == {"r0": 128}
    assert records[2]["kv_usage"] == 0.25


def test_seq_is_gap_free(tmp_path):
    path = tmp_path / "t.jsonl"
    with _writer(path) as writer:
        for i in range(5):
            writer.emit("step", t=float(i), step=i)
    assert [r["seq"] for r in read_trace(path)] == list(range(6))


def test_discontinuity_is_an_error_not_a_silent_diff(tmp_path):
    """A dropped record must not read as a behavioral change in a conformance diff."""
    path = tmp_path / "t.jsonl"
    with _writer(path) as writer:
        for i in range(4):
            writer.emit("step", t=float(i), step=i)

    lines = path.read_bytes().splitlines()
    path.write_bytes(b"\n".join(lines[:2] + lines[3:]) + b"\n")

    with pytest.raises(ValueError, match="trace discontinuity"):
        list(read_trace(path))


def test_two_runs_with_the_same_inputs_are_byte_identical():
    """B4. This is what makes C1--C4 comparable at all."""

    def run() -> bytes:
        stream = io.BytesIO()
        writer = TraceWriter(
            stream=stream, seed=7, clock_mode="virtual", upstream_version="0.27.1"
        )
        for step in range(20):
            writer.emit(
                "step",
                t=step * 0.01,
                step=step,
                scheduled={"a": 8, "b": 4},
                kv_usage=step / 100,
            )
        writer.close()
        return stream.getvalue()

    assert run() == run()


def test_emitting_after_close_is_an_error(tmp_path):
    writer = _writer(tmp_path / "t.jsonl")
    writer.close()
    with pytest.raises(RuntimeError, match="closed trace"):
        writer.emit("step", t=0.0)


def test_close_is_idempotent(tmp_path):
    writer = _writer(tmp_path / "t.jsonl")
    writer.close()
    writer.close()


def test_requires_exactly_one_destination():
    with pytest.raises(ValueError, match="exactly one"):
        TraceWriter(seed=1, clock_mode="virtual", upstream_version="0.27.1")
    with pytest.raises(ValueError, match="exactly one"):
        TraceWriter(
            "x.jsonl",
            stream=io.BytesIO(),
            seed=1,
            clock_mode="virtual",
            upstream_version="0.27.1",
        )


def test_null_writer_matches_the_interface():
    """Tracing off must not change control flow at the call site."""
    with NullTraceWriter() as writer:
        writer.emit("step", t=0.0, step=0)
    assert NullTraceWriter().enabled is False
