"""Sampling parameters.

Upstream: vllm/sampling_params.py
Tier: C

R11.4: the full parameter surface is accepted and validated exactly as upstream
validates it, because a client that sends an out-of-range `top_p` must get the same
error here as it would from real vLLM (C5, C7).

What differs is downstream: these values reach only as far as changing the PRNG draw
(B3). Outputs are schema-correct and deterministic, not semantically meaningful (NG1,
NG3). A parameter being *accepted* here is a statement about the API surface, not a
claim that it steers text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

_SAMPLING_EPS = 1e-5


class RequestOutputKind(IntEnum):
    CUMULATIVE = 0
    """Return the full output so far on every update."""
    DELTA = 1
    """Return only what is new since the last update. Streaming uses this."""
    FINAL_ONLY = 2
    """Return only the final output."""


@dataclass
class SamplingParams:
    """Parameters for text generation."""

    n: int = 1
    best_of: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    max_tokens: int | None = 16
    min_tokens: int = 0
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    logit_bias: dict[int, float] | None = None
    allowed_token_ids: list[int] | None = None
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    extra_args: dict[str, Any] | None = None

    #: Populated by the processor from `stop`; kept separate so the scheduler's
    #: stop-string checker does not re-normalize on every step.
    all_stop_token_ids: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        elif self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = []
        self.all_stop_token_ids = set(self.stop_token_ids)

        if self.best_of is not None and self.best_of != self.n:
            raise ValueError(
                f"best_of ({self.best_of}) must equal n ({self.n}); vLLM removed "
                f"support for best_of > n"
            )
        self._verify_args()

        # Greedy decoding: upstream normalizes near-zero temperature to exactly zero
        # so the sampler takes the argmax path rather than dividing by a tiny number.
        if self.temperature < _SAMPLING_EPS:
            self.temperature = 0.0
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0

    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                f"presence_penalty must be in [-2, 2], got {self.presence_penalty}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}."
            )
        if self.repetition_penalty <= 0.0:
            raise ValueError(
                f"repetition_penalty must be greater than zero, got "
                f"{self.repetition_penalty}."
            )
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < 0:
            raise ValueError(f"top_k must be 0 (disable) or greater, got {self.top_k}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}.")
        if self.min_tokens < 0:
            raise ValueError(f"min_tokens must be non-negative, got {self.min_tokens}.")
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) must not be greater than max_tokens "
                f"({self.max_tokens})."
            )
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(f"logprobs must be non-negative, got {self.logprobs}.")
        if self.prompt_logprobs is not None and self.prompt_logprobs < 0:
            raise ValueError(
                f"prompt_logprobs must be non-negative, got {self.prompt_logprobs}."
            )
        if self.stop and not self.detokenize:
            raise ValueError(
                "stop strings are only supported when detokenize is True. Set "
                "detokenize=True to use stop."
            )

    @property
    def all_stop_strings(self) -> list[str]:
        assert isinstance(self.stop, list)
        return self.stop

    def clone(self) -> SamplingParams:
        from copy import deepcopy

        return deepcopy(self)
