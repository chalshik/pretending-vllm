"""OpenAI Responses API schema. R2.2, C5.

Upstream: vllm/entrypoints/openai/responses/protocol.py
Tier: C

Upstream declares only `ResponsesRequest`, `ResponsesResponse` and the usage triple
itself; every other type on the wire -- the input and output items, `Tool`,
`ResponseStatus`, `IncompleteDetails`, `Reasoning` -- it imports from the `openai`
PyPI package, which pvllm does not depend on. So they are re-declared here, field for
field, against `openai>=2.0.0` as vLLM v0.27.1 pins it.

**Declaration order is load-bearing.** Upstream serialises with
`model_dump(mode="json", by_alias=True)` and *no* `exclude_none`, so every declared
field appears in the body -- nulls included -- in the order it was declared. A client
diffing two bodies sees key order, so the order below is upstream's, including the
places it is not alphabetical (`top_p, background, ...`, and the penalties trailing
after `user`).

`extra="allow"` matches upstream's `OpenAIBaseModel`: an unknown request field is
accepted and ignored, never a 422. That is what lets a client written against a newer
OpenAI SDK keep working.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pvllm.entrypoints.openai.structured_outputs import build_structured_outputs
from pvllm.sampling_params import RequestOutputKind, SamplingParams

#: Upstream's fallbacks, applied in `to_sampling_params` when the request leaves the
#: field unset. They are *not* the pydantic defaults: the request fields default to
#: `None` so that "unset" stays distinguishable, and these fill in afterwards.
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_TOP_P = 1.0
_DEFAULT_TOP_K = 0

ResponseStatus = Literal[
    "queued", "in_progress", "completed", "incomplete", "failed", "cancelled"
]
#: An output *item* has a narrower life than the response containing it: a message is
#: never "queued", and never "failed" or "cancelled" -- those belong to the response,
#: which may carry no message at all.
ItemStatus = Literal["in_progress", "completed", "incomplete"]


class InputTokensDetails(BaseModel):
    cached_tokens: int = 0
    #: Populated only by the multi-turn tool loop, which pvllm refuses. Always empty,
    #: and present because the field is on the wire.
    input_tokens_per_turn: list[int] = Field(default_factory=list)
    cached_tokens_per_turn: list[int] = Field(default_factory=list)


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    tool_output_tokens: int = 0
    output_tokens_per_turn: list[int] = Field(default_factory=list)
    tool_output_tokens_per_turn: list[int] = Field(default_factory=list)


class ResponseUsage(BaseModel):
    """C5. Note the field names.

    This endpoint reports `input_tokens`/`output_tokens`, where completions and chat
    completions report `prompt_tokens`/`completion_tokens`. Reusing pvllm's shared
    `UsageInfo` here would put the wrong names on the wire, so this is a separate
    model on purpose rather than by oversight.
    """

    input_tokens: int = 0
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
    output_tokens: int = 0
    output_tokens_details: OutputTokensDetails = Field(
        default_factory=OutputTokensDetails
    )
    total_tokens: int = 0


class ResponseOutputText(BaseModel):
    #: Required-then-optional, alphabetical within each run: that is how the
    #: Stainless-generated OpenAI models order their fields, and the order shows.
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    text: str = ""
    type: Literal["output_text"] = "output_text"
    logprobs: list[dict[str, Any]] | None = None


class ResponseOutputMessage(BaseModel):
    id: str
    content: list[ResponseOutputText] = Field(default_factory=list)
    role: Literal["assistant"] = "assistant"
    status: ItemStatus = "completed"
    type: Literal["message"] = "message"


class IncompleteDetails(BaseModel):
    reason: str


class ResponsesRequest(BaseModel):
    """C5. Exactly one required field: `input`.

    `model` is optional here, unlike `ChatCompletionRequest` and `CompletionRequest`
    where it is required. A Responses request that names no model is legal upstream
    and serves the base model, so refusing it would be a divergence a client hits on
    its first call.
    """

    model_config = ConfigDict(extra="allow")

    background: bool | None = False
    include: list[str] | None = None
    #: The only required field. A bare string is a single user turn; a list is the
    #: item form, which carries prior turns and their roles.
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: dict[str, str] | None = None
    model: str | None = None
    logit_bias: dict[str, float] | None = None
    parallel_tool_calls: bool | None = True
    previous_response_id: str | None = None
    prompt: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    #: A vLLM extension, not OpenAI's.
    include_reasoning: bool = True
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] = "auto"
    store: bool | None = True
    stream: bool | None = False
    temperature: float | None = None
    text: dict[str, Any] | None = None
    tool_choice: str | dict[str, Any] = "auto"
    tools: list[dict[str, Any]] = Field(default_factory=list)
    top_logprobs: int | None = 0
    top_p: float | None = None
    top_k: int | None = None
    truncation: Literal["auto", "disabled"] | None = "disabled"
    user: str | None = None
    skip_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    #: Accepted and ignored, as upstream: its own docstring says vLLM ignores it.
    prompt_cache_key: str | None = None

    # --- vLLM extensions ----------------------------------------------------

    #: Becomes `ResponsesResponse.id` verbatim, so a client can choose its own id --
    #: and the store is keyed by it, which means two POSTs sharing one `request_id`
    #: collide and the second overwrites the first. That is upstream's behaviour.
    #:
    #: Upstream defaults this to `resp_{random_uuid()}`. pvllm mints a counter in the
    #: same shape instead (`OpenAIServingResponses._next_request_id`), because B4
    #: requires the same seed and config to produce byte-identical output and the
    #: purity lint keeps randomness inside `pvllm/sim/`. The shape a client parses is
    #: preserved; the entropy is not.
    request_id: str | None = None
    media_io_kwargs: dict[str, dict[str, Any]] | None = None
    mm_processor_kwargs: dict[str, Any] | None = None
    priority: int = 0
    cache_salt: str | None = None
    enable_response_messages: bool = False
    previous_input_messages: list[dict[str, Any]] | None = None
    structured_outputs: dict[str, Any] | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    #: Upstream's default really is a mutable `[]` rather than `None`.
    stop: str | list[str] | None = Field(default_factory=list)
    ignore_eos: bool = False
    vllm_xargs: dict[str, Any] | None = None
    kv_transfer_params: dict[str, Any] | None = None
    ec_transfer_params: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def check_tool_usage(cls, data: Any) -> Any:
        """Upstream rewrites `tool_choice` before anything else sees it.

        With no `tools`, "auto" becomes "none" -- and it is the *rewritten* value the
        response echoes, so a plain request comes back saying `"tool_choice": "none"`
        despite having sent nothing. The rewrite has to happen here, before
        `from_request` reads the field.

        Asking for a tool without supplying one is an error upstream, but it is *not*
        raised here: a `ValueError` inside a validator becomes a 422, and upstream
        answers 400. `OpenAIServingResponses._refuse` makes that call instead, where
        the status is ours to choose.
        """
        if not isinstance(data, dict):
            return data
        if data.get("tools"):
            return data
        if data.get("tool_choice", "auto") in ("auto", "none"):
            data["tool_choice"] = "none"
        return data

    def is_include_output_logprobs(self) -> bool:
        return bool(self.include) and "message.output_text.logprobs" in (
            self.include or []
        )

    def to_sampling_params(self, default_max_tokens: int) -> SamplingParams:
        """Resolve the request against upstream's fallbacks.

        `default_max_tokens` is what is left of the context window once the prompt is
        counted. Upstream passes the same thing, and the resolved value is echoed
        back as `max_output_tokens` -- so a request that set nothing gets a number,
        not the `null` it sent.
        """
        max_tokens = self.max_output_tokens or default_max_tokens
        stop = self.stop if self.stop else []
        if isinstance(stop, str):
            stop = [stop]

        extra_args: dict[str, Any] = dict(self.vllm_xargs or {})
        if self.kv_transfer_params is not None:
            extra_args["kv_transfer_params"] = self.kv_transfer_params
        if self.ec_transfer_params is not None:
            extra_args["ec_transfer_params"] = self.ec_transfer_params

        return SamplingParams(
            temperature=(
                self.temperature
                if self.temperature is not None
                else _DEFAULT_TEMPERATURE
            ),
            top_p=self.top_p if self.top_p is not None else _DEFAULT_TOP_P,
            top_k=self.top_k if self.top_k is not None else _DEFAULT_TOP_K,
            repetition_penalty=(
                self.repetition_penalty if self.repetition_penalty is not None else 1.0
            ),
            presence_penalty=(
                self.presence_penalty if self.presence_penalty is not None else 0.0
            ),
            frequency_penalty=(
                self.frequency_penalty if self.frequency_penalty is not None else 0.0
            ),
            seed=self.seed,
            stop=stop,
            ignore_eos=self.ignore_eos,
            max_tokens=max_tokens,
            # `top_logprobs` defaults to 0, not None, and is forwarded *only* when
            # `include` asks for logprobs. Treating it as an independent switch is
            # the plausible wrong answer here.
            logprobs=self.top_logprobs if self.is_include_output_logprobs() else None,
            skip_special_tokens=self.skip_special_tokens,
            include_stop_str_in_output=self.include_stop_str_in_output,
            logit_bias={int(k): v for k, v in (self.logit_bias or {}).items()} or None,
            output_kind=(
                RequestOutputKind.DELTA if self.stream else RequestOutputKind.FINAL_ONLY
            ),
            structured_outputs=_structured_outputs_from_text(self.text),
            extra_args=extra_args or None,
        )


def _structured_outputs_from_text(text: dict[str, Any] | None) -> Any:
    """R15. `text.format` is where this endpoint puts what chat calls
    `response_format`.

    Same machinery underneath, reached through a different field name -- so a JSON
    schema sent to `/v1/responses` constrains decoding exactly as it would on
    `/v1/chat/completions`.
    """
    if not text:
        return None
    text_format = text.get("format")
    if not isinstance(text_format, dict):
        return None
    return build_structured_outputs(response_format=text_format)


class ResponsesResponse(BaseModel):
    """C5. Every field appears in the body, nulls included, in this order."""

    id: str
    #: From the engine's clock, not wall time (R19.1).
    created_at: int
    incomplete_details: IncompleteDetails | None = None
    instructions: str | None = None
    metadata: dict[str, str] | None = None
    model: str
    object: Literal["response"] = "response"
    output: list[ResponseOutputMessage] = Field(default_factory=list)
    parallel_tool_calls: bool = True
    temperature: float
    tool_choice: str | dict[str, Any] = "auto"
    tools: list[dict[str, Any]] = Field(default_factory=list)
    top_p: float
    background: bool = False
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    previous_response_id: str | None = None
    prompt: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    service_tier: str = "auto"
    status: ResponseStatus = "completed"
    text: dict[str, Any] | None = None
    top_logprobs: int | None = None
    truncation: str | None = "disabled"
    usage: ResponseUsage | None = None
    user: str | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    kv_transfer_params: dict[str, Any] | None = None
    ec_transfer_params: dict[str, Any] | None = None
    #: Always null: they are the `enable_response_messages` debug surface, which is
    #: refused by name. Declared so the key is still on the wire.
    input_messages: list[dict[str, Any]] | None = None
    output_messages: list[dict[str, Any]] | None = None

    @classmethod
    def from_request(
        cls,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        *,
        response_id: str,
        model_name: str,
        created_at: int,
        output: list[ResponseOutputMessage],
        status: ResponseStatus,
        usage: ResponseUsage | None,
        incomplete_details: IncompleteDetails | None = None,
    ) -> ResponsesResponse:
        """Echo the *resolved* request, not the request as sent.

        `temperature`, `top_p`, the penalties and `max_output_tokens` all come off
        the sampling params rather than the request fields, so a request that set
        none of them still gets concrete numbers back -- which is what upstream does
        and what a client comparing the two bodies would notice.
        """
        return cls(
            id=response_id,
            created_at=created_at,
            incomplete_details=incomplete_details,
            instructions=request.instructions,
            metadata=request.metadata,
            model=model_name,
            output=output,
            parallel_tool_calls=(
                request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True
            ),
            temperature=sampling_params.temperature,
            tool_choice=request.tool_choice,
            tools=request.tools,
            top_p=sampling_params.top_p,
            background=bool(request.background),
            max_output_tokens=sampling_params.max_tokens,
            max_tool_calls=request.max_tool_calls,
            previous_response_id=request.previous_response_id,
            prompt=request.prompt,
            reasoning=request.reasoning,
            service_tier=request.service_tier,
            status=status,
            text=request.text,
            top_logprobs=sampling_params.logprobs,
            truncation=request.truncation,
            usage=usage,
            user=request.user,
            presence_penalty=sampling_params.presence_penalty,
            frequency_penalty=sampling_params.frequency_penalty,
            kv_transfer_params=request.kv_transfer_params,
            ec_transfer_params=request.ec_transfer_params,
        )
