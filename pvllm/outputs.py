"""Public output types.

Upstream: vllm/outputs.py
Tier: C

What `LLM.generate` returns and what the OpenAI serving layer converts into HTTP
responses. Field names match upstream because product code destructures these directly
(G4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pvllm.sampling_params import RequestOutputKind


@dataclass
class CompletionOutput:
    """One of the `n` completions for a request."""

    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None = None
    logprobs: list[dict[int, Any]] | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (
            f"CompletionOutput(index={self.index}, text={self.text!r}, "
            f"token_ids={self.token_ids}, finish_reason={self.finish_reason!r})"
        )


@dataclass
class RequestOutput:
    """The output of one request."""

    request_id: str
    prompt: str | None
    prompt_token_ids: list[int] | None
    outputs: list[CompletionOutput]
    finished: bool
    prompt_logprobs: list[dict[int, Any] | None] | None = None
    metrics: Any = None
    #: Prompt tokens served from the prefix cache. Real, and the number a
    #: cache-effectiveness test should assert on (R6.9).
    num_cached_tokens: int = 0
    kv_transfer_params: dict[str, Any] | None = None

    def add(self, next_output: RequestOutput) -> None:
        """Merge a later chunk of the same request into this one."""
        self.finished |= next_output.finished
        for next_completion in next_output.outputs:
            for completion in self.outputs:
                if completion.index == next_completion.index:
                    completion.text += next_completion.text
                    completion.token_ids.extend(next_completion.token_ids)
                    completion.finish_reason = next_completion.finish_reason
                    completion.stop_reason = next_completion.stop_reason
                    break
            else:
                self.outputs.append(next_completion)

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id!r}, "
            f"prompt={self.prompt!r}, outputs={self.outputs}, "
            f"finished={self.finished}, num_cached_tokens={self.num_cached_tokens})"
        )


@dataclass
class PoolingOutput:
    """Embedding output. P3 (R2.2)."""

    data: list[float]


@dataclass
class PoolingRequestOutput:
    """The output of one pooling request. P3."""

    request_id: str
    outputs: PoolingOutput
    prompt_token_ids: list[int]
    finished: bool


__all__ = [
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "RequestOutput",
    "RequestOutputKind",
]
