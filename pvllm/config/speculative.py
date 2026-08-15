"""Speculative decoding configuration. R14.

Upstream: vllm/config/speculative.py
Tier: C

A draft model proposes `num_speculative_tokens` continuations, the target model
verifies them in one forward pass, and every accepted draft is a token that cost no
extra step. The win is real and large when acceptance is high; when it is low, the
verification work is wasted and throughput *drops*.

That trade is the reason this is modeled at all: which side of it a deployment lands
on depends on the draft model's agreement with the target, and a product tuning
`num_speculative_tokens` needs to see the curve bend.

**What the simulator cannot supply is the acceptance itself.** There is no draft model
and no target distribution, so `SimConfig.spec_acceptance_rate` stands in for the
agreement between them -- the same shape of substitution the grammar backend makes.
Set it from a measurement of your real pair, and the scheduling, the token accounting,
and the metrics around it are then faithful.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Proposal methods upstream supports. Each needs a real draft model or a real
#: n-gram index over real text, so none of them is *executed* here -- the field
#: exists so a config round-trips, and the simulator proposes synthetic drafts
#: whatever it says.
SPECULATIVE_METHODS = (
    "ngram",
    "eagle",
    "eagle3",
    "medusa",
    "mlp_speculator",
    "draft_model",
)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    num_speculative_tokens: int = 0
    method: str = "ngram"
    model: str | None = None
    #: Upstream disables speculation when the running batch is larger than this,
    #: because verification stops paying at high concurrency. Modeled, because it is
    #: a knob that changes the answer.
    speculative_disable_by_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be at least 1 when speculative "
                f"decoding is configured, got {self.num_speculative_tokens}"
            )
        if self.num_speculative_tokens > 16:
            raise ValueError(
                f"num_speculative_tokens above 16 is beyond what any real draft "
                f"model sustains, got {self.num_speculative_tokens}"
            )
        if self.method not in SPECULATIVE_METHODS:
            raise ValueError(
                f"unknown speculative method {self.method!r}; expected one of "
                f"{list(SPECULATIVE_METHODS)}"
            )
