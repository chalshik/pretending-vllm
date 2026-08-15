"""Structured output configuration.

Upstream: vllm/config/structured_outputs.py
Tier: C

Present so the config surface matches. R15 lands in M4; the scheduler-side call sites
(grammar bitmask, WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR) exist from M1 so the shape is
right even while the grammar work below them is absent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StructuredOutputsConfig:
    """Configuration for grammar-constrained decoding."""

    backend: str = "auto"
    disable_fallback: bool = False
    disable_any_whitespace: bool = False
    reasoning_parser: str = ""
