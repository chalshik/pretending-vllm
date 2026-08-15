"""OpenAI chat completions schema.

Upstream: vllm/entrypoints/openai/chat_completion/protocol.py
Tier: C

C5. Shares `UsageInfo` and `StreamOptions` with the completions schema, as upstream
does, so a client sees one usage shape across both endpoints.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from pvllm.entrypoints.openai.completion.protocol import StreamOptions, UsageInfo
from pvllm.entrypoints.openai.structured_outputs import build_structured_outputs
from pvllm.sampling_params import RequestOutputKind, SamplingParams


class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int = 1
    presence_penalty: float = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    user: str | None = None

    # vLLM extensions.
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    min_tokens: int = 0
    stop_token_ids: list[int] | None = None
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    add_generation_prompt: bool = True
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
        # `max_completion_tokens` supersedes `max_tokens`, which OpenAI deprecated.
        # Preferring the new field matters: a client sending both means the new one.
        max_tokens = self.max_completion_tokens or self.max_tokens
        return SamplingParams(
            n=self.n,
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
            max_tokens=max_tokens,
            min_tokens=self.min_tokens,
            logprobs=self.top_logprobs if self.logprobs else None,
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


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: dict[str, Any] | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    #: From the engine's clock, not wall time (R19.1).
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: dict[str, Any] | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionResponseStreamChoice] = Field(default_factory=list)
    usage: UsageInfo | None = None
