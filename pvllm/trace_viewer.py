"""Rendering a trace to a step timeline. R19.4.

Upstream: (none -- pvllm addition)
Tier: B

A JSONL trace answers any question you can write a query for. A timeline answers the
one you did not know to ask: *why did this run behave like that*. Preemption thrash,
a request starved behind a long prefill, a prefix cache that stopped hitting -- all of
those are obvious in a picture and tedious to find by grepping.

Each request is a row, each step a column. What a cell shows is the *scheduling
decision*, which is the thing the trace exists to explain:

    #  a large prefill chunk        =  a decode step (one token)
    :  a small prefill chunk        .  running, nothing scheduled this step
    ^  resumed after preemption     !  preempted this step

Deliberately not a Gantt chart over wall time. Under a virtual clock the steps are
what matter and their durations are modeled; laying rows out by duration would make
the modeled cost model look like the measured subject of the picture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pvllm.tracing import read_trace

#: Cell glyphs, in the order the legend lists them.
PREEMPTED = "!"
RESUMED = "^"
PREFILL_LARGE = "#"
PREFILL_SMALL = ":"
DECODE = "="
IDLE = "."
ABSENT = " "


@dataclass
class TraceSummary:
    """A trace, reduced to what a timeline needs."""

    header: dict[str, Any]
    #: request_id -> step index -> glyph.
    rows: dict[str, dict[int, str]] = field(default_factory=dict)
    num_steps: int = 0
    total_tokens: int = 0
    finish_reasons: dict[str, str] = field(default_factory=dict)
    prompt_tokens: dict[str, int] = field(default_factory=dict)
    #: Per step, for the footer sparklines.
    kv_usage: list[float] = field(default_factory=list)
    scheduled_tokens: list[int] = field(default_factory=list)
    num_preemptions: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_queries: int = 0

    @property
    def request_ids(self) -> list[str]:
        """In first-appearance order, so a row's position matches arrival."""
        return list(self.rows)


def summarize(path: str | os.PathLike[str]) -> TraceSummary:
    """Reduce a JSONL trace to a timeline."""
    summary = TraceSummary(header={})
    seen_order: dict[str, None] = {}

    for record in read_trace(path):
        kind = record.get("type")
        if kind == "header":
            summary.header = record
        elif kind == "request":
            request_id = record["request_id"]
            seen_order.setdefault(request_id, None)
            summary.rows.setdefault(request_id, {})
            if record.get("event") == "arrived":
                summary.prompt_tokens[request_id] = record.get("num_prompt_tokens", 0)
            elif record.get("event") in ("finished", "aborted"):
                summary.finish_reasons[request_id] = record.get(
                    "finish_reason", record.get("event", "")
                )
        elif kind == "step":
            step = int(record.get("step", summary.num_steps + 1))
            summary.num_steps = max(summary.num_steps, step)
            summary.kv_usage.append(float(record.get("kv_usage", 0.0)))
            summary.scheduled_tokens.append(
                int(record.get("total_num_scheduled_tokens", 0))
            )
            summary.total_tokens += int(record.get("total_num_scheduled_tokens", 0))
            summary.num_preemptions = max(
                summary.num_preemptions, int(record.get("num_preemptions_total", 0))
            )
            summary.prefix_cache_hits = max(
                summary.prefix_cache_hits, int(record.get("prefix_cache_hits", 0))
            )
            summary.prefix_cache_queries = max(
                summary.prefix_cache_queries,
                int(record.get("prefix_cache_queries", 0)),
            )
            _apply_step(summary, record, step, seen_order)

    return summary


def _apply_step(
    summary: TraceSummary,
    record: dict[str, Any],
    step: int,
    seen_order: dict[str, None],
) -> None:
    resumed = set(record.get("resumed_reqs") or ())
    preempted = set(record.get("preempted_req_ids") or ())
    scheduled: dict[str, int] = record.get("num_scheduled_tokens") or {}
    running = set(record.get("new_reqs") or ()) | set(record.get("cached_reqs") or ())

    for request_id, num_tokens in scheduled.items():
        seen_order.setdefault(request_id, None)
        row = summary.rows.setdefault(request_id, {})
        if request_id in resumed:
            row[step] = RESUMED
        elif num_tokens > 1:
            # A prefill chunk. Split by size so a long prompt being fed in slices is
            # visually distinct from one that fit in a single step.
            row[step] = PREFILL_LARGE if num_tokens >= 8 else PREFILL_SMALL
        else:
            row[step] = DECODE

    for request_id in preempted:
        summary.rows.setdefault(request_id, {})[step] = PREEMPTED

    # Running but unscheduled (the budget ran out before reaching it), or still in
    # the waiting queue. Both are shown, because a row of dots beside a busy step is
    # exactly what starvation looks like -- and a count alone cannot say which
    # request waited.
    waiting = set(record.get("waiting_req_ids") or ())
    for request_id in (running - set(scheduled)) | waiting:
        seen_order.setdefault(request_id, None)
        summary.rows.setdefault(request_id, {}).setdefault(step, IDLE)


def render_text(summary: TraceSummary, width: int = 100) -> str:
    """The timeline as text."""
    header = summary.header
    lines: list[str] = [
        f"pretending-vllm trace  (upstream {header.get('upstream_version', '?')}, "
        f"seed {header.get('seed', '?')}, clock {header.get('clock_mode', '?')})",
    ]
    config = header.get("config") or {}
    if config:
        lines.append(
            f"  model={config.get('model')!r} card={config.get('model_card')!r} "
            f"device={config.get('device_card')!r} block_size={config.get('block_size')} "
            f"cost_model={config.get('cost_model')!r}"
        )
    lines.append("")

    if not summary.rows:
        lines.append("  (no requests)")
        return "\n".join(lines)

    # Steps are columns; a long run is compressed so the picture still fits a
    # terminal. Compression takes the *most significant* glyph in each bucket --
    # a preemption must never be averaged away, since it is the thing you are
    # usually looking for.
    label_width = max(len(r) for r in summary.request_ids) + 2
    columns = max(1, width - label_width - 2)
    stride = max(1, (summary.num_steps + columns - 1) // columns)

    for request_id in summary.request_ids:
        row = summary.rows[request_id]
        cells = []
        for start in range(1, summary.num_steps + 1, stride):
            bucket = [row.get(s) for s in range(start, start + stride)]
            cells.append(_most_significant(bucket))
        finish = summary.finish_reasons.get(request_id, "")
        lines.append(f"  {request_id:<{label_width}}{''.join(cells)}  {finish}")

    lines.append("")
    lines.append(
        f"  steps={summary.num_steps}  tokens={summary.total_tokens}  "
        f"preemptions={summary.num_preemptions}  "
        f"peak_kv={max(summary.kv_usage or [0.0]):.1%}"
    )
    if summary.prefix_cache_queries:
        rate = summary.prefix_cache_hits / summary.prefix_cache_queries
        lines.append(
            f"  prefix cache: {summary.prefix_cache_hits}/"
            f"{summary.prefix_cache_queries} tokens ({rate:.1%})"
        )
    if stride > 1:
        lines.append(f"  (each column is {stride} steps)")
    lines.append(
        f"  legend: {PREFILL_LARGE} prefill  {PREFILL_SMALL} small prefill  "
        f"{DECODE} decode  {IDLE} waiting  {PREEMPTED} preempted  {RESUMED} resumed"
    )
    return "\n".join(lines)


def _most_significant(bucket: list[str | None]) -> str:
    """Collapse several steps into one column.

    Ordered by how much a reader needs to see it: a preemption in the bucket must
    survive compression, because averaging it away hides the event the picture was
    opened to find.
    """
    for glyph in (PREEMPTED, RESUMED, PREFILL_LARGE, PREFILL_SMALL, DECODE, IDLE):
        if glyph in bucket:
            return glyph
    return ABSENT


def render_svg(summary: TraceSummary, cell: int = 6, row_height: int = 14) -> str:
    """The timeline as SVG, for a trace too wide to read as text."""
    colors = {
        PREFILL_LARGE: "#2f6f9f",
        PREFILL_SMALL: "#6da9d2",
        DECODE: "#7fb069",
        IDLE: "#e0e0e0",
        PREEMPTED: "#d1495b",
        RESUMED: "#edae49",
    }
    label_width = 90
    width = label_width + summary.num_steps * cell + 20
    height = (len(summary.request_ids) + 4) * row_height + 20

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="monospace" font-size="10">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="6" y="14" font-size="11">pretending-vllm trace '
        f"(seed {summary.header.get('seed', '?')}, "
        f"{summary.num_steps} steps, {summary.num_preemptions} preemptions)</text>",
    ]

    for row_index, request_id in enumerate(summary.request_ids):
        y = (row_index + 2) * row_height
        parts.append(f'<text x="6" y="{y + 9}">{_escape(request_id)[:12]}</text>')
        for step, glyph in summary.rows[request_id].items():
            color = colors.get(glyph)
            if color is None:
                continue
            x = label_width + (step - 1) * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{row_height - 4}" '
                f'fill="{color}"/>'
            )

    legend_y = (len(summary.request_ids) + 3) * row_height
    for i, (glyph, label) in enumerate(
        [
            (PREFILL_LARGE, "prefill"),
            (DECODE, "decode"),
            (IDLE, "waiting"),
            (PREEMPTED, "preempted"),
            (RESUMED, "resumed"),
        ]
    ):
        x = 6 + i * 90
        parts.append(
            f'<rect x="{x}" y="{legend_y - 8}" width="8" height="8" '
            f'fill="{colors[glyph]}"/>'
        )
        parts.append(f'<text x="{x + 12}" y="{legend_y}">{label}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
