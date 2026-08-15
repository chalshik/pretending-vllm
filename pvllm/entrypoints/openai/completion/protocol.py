"""OpenAI completions schema.

Upstream: vllm/entrypoints/openai/completion/protocol.py
Tier: C

C5 binds the request and response schema exactly, because that is the surface a
product actually integrates against (G4). Field names, defaults, and optionality
follow the OpenAI API rather than anything vLLM-specific.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from pvllm.entrypoints.openai.structured_outputs import build_structured_outputs
from pvllm.sampling_params import RequestOutputKind, SamplingParams


class StreamOptions(BaseModel):
    """R2.3."""

    include_usage: bool = False
    continuous_usage_stats: bool = False


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    best_of: int | None = None
    echo: bool = False
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: int | None = None
    max_tokens: int | None = 16
    n: int = 1
    presence_penalty: float = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    suffix: str | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    user: str | None = None

    # vLLM extensions, kept because products that target vLLM use them.
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    min_tokens: int = 0
    stop_token_ids: list[int] | None = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    priority: int = 0

    #: R15. OpenAI's own constraint field.
    response_format: dict[str, Any] | None = None
    #: R15. vLLM's guided-decoding extensions. Products targeting vLLM use these
    #: rather than response_format, so both surfaces are accepted.
    guided_json: str | dict[str, Any] | None = None
    guided_regex: str | None = None
    guided_choice: list[str] | None = None
    guided_grammar: str | None = None
    structural_tag: str | None = None
    guided_whitespace_pattern: str | None = None
    guided_decoding_backend: str | None = None

    def to_sampling_params(self, streaming: bool) -> SamplingParams:
        return SamplingParams(
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            seed=self.seed,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            logprobs=self.logprobs,
            skip_special_tokens=self.skip_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            logit_bias={int(k): v for k, v in (self.logit_bias or {}).items()} or None,
            output_kind=(
                RequestOutputKind.DELTA if streaming else RequestOutputKind.FINAL_ONLY
            ),
            structured_outputs=build_structured_outputs(
                response_format=self.response_format,
                guided_json=self.guided_json,
                guided_regex=self.guided_regex,
                guided_choice=self.guided_choice,
                guided_grammar=self.guided_grammar,
                structural_tag=self.structural_tag,
                guided_whitespace_pattern=self.guided_whitespace_pattern,
                guided_decoding_backend=self.guided_decoding_backend,
            ),
        )


class CompletionLogProbs(BaseModel):
    text_offset: list[int] = Field(default_factory=list)
    token_logprobs: list[float | None] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    top_logprobs: list[dict[str, float] | None] = Field(default_factory=list)


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: CompletionLogProbs | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    #: Unix timestamp from the *engine's* clock, not wall time (R19.1). Under a
    #: virtual clock this is deterministic, which is what lets a response be
    #: golden-tested.
    created: int
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: CompletionLogProbs | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionResponseStreamChoice]
    usage: UsageInfo | None = None
