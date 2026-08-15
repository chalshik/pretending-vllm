"""Observability configuration.

Upstream: vllm/config/observability.py
Tier: C
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObservabilityConfig:
    """Configuration for metrics, logging, and tracing."""

    show_hidden_metrics_for_version: str | None = None
    otlp_traces_endpoint: str | None = None
    collect_detailed_traces: list[str] | None = None
    #: R12.3: emit the periodic stats log line.
    disable_log_stats: bool = False

    def __post_init__(self) -> None:
        if self.otlp_traces_endpoint is not None:
            raise NotImplementedError(
                "OpenTelemetry export is not implemented; the JSONL event trace "
                "(requirement R19.3) is the supported tracing surface"
            )
