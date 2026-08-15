"""Speculative decoding configuration.

Upstream: vllm/config/speculative.py
Tier: C

Present so the config surface matches. Speculative decoding (R14) lands in M4 -- it is
where block accounting gets genuinely hard, which is why it is worth having eventually
and worth refusing to fake now.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    method: str | None = None
    model: str | None = None
    num_speculative_tokens: int | None = None
    #: R14.2: per-position Bernoulli acceptance with decay.
    acceptance_rate: float = 0.7
    acceptance_decay: float = 0.9

    def __post_init__(self) -> None:
        raise NotImplementedError(
            "speculative decoding (requirement R14) lands in M4; the scheduler and KV "
            "manager need speculative slots, rejection, and rollback first"
        )
