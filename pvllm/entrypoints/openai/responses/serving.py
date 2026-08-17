"""The Responses API, over the same engine. R2.2, C5, C7.

Upstream: vllm/entrypoints/openai/responses/serving.py
Tier: B

Upstream's file is 1,561 lines and most of it is tool calling, the harmony/gpt-oss
reasoning parser, and the MCP tool server -- three subsystems pvllm has no basis for,
because they operate on token streams a simulated model does not produce. Those are
refused by name rather than approximated: answering a tool-calling request with a
plain assistant message is the kind of plausible wrong answer that costs more than an
error would.

`reasoning` is *not* one of them, despite the name. It selects no parser: upstream
turns `reasoning.effort` into a chat-template kwarg and echoes the field back, so a
stock server answers 200 and this one does too.

What is left is the part a client can actually observe: the wire schema, the nine-event
stream, and the response store. The store is worth having *because* it is pure control
plane -- four dicts and a lock, no device anywhere near it -- and so there is nothing
to simulate and no excuse for faking it.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from pvllm.entrypoints.openai.responses.protocol import (
    IncompleteDetails,
    InputTokensDetails,
    ItemStatus,
    OutputTokensDetails,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
)
from pvllm.entrypoints.serve.utils.error_response import (
    create_error_response,
    model_not_found,
    to_error_response,
)
from pvllm.outputs import RequestOutput
from pvllm.sampling_params import SamplingParams
from pvllm.v1.engine.async_llm import AsyncLLM

#: Upstream reads this through `vllm.envs`, which parses `int(os.getenv(name, "0"))`.
#: Same name and same default here, so one runbook flag flips both.
_STORE_ENV = "VLLM_ENABLE_RESPONSES_API_STORE"

#: Terminal statuses. `cancel` is a no-op against any of them.
_TERMINAL: frozenset[str] = frozenset(
    {"completed", "incomplete", "failed", "cancelled"}
)


def store_enabled() -> bool:
    """Whether the response store is on. Off by default, exactly as upstream.

    This matters more than it looks. With the store off, a stock vLLM answers every
    `GET /v1/responses/{id}` with a 404 and rejects `background=True` -- so a pvllm
    that stored by default would *succeed* where the real thing fails, and the
    divergence would only surface when the user swapped the real engine back in.
    """
    try:
        return bool(int(os.getenv(_STORE_ENV, "0")))
    except ValueError:
        return False


def construct_input_messages(
    *,
    request_instructions: str | None = None,
    request_input: str | list[dict[str, Any]],
    prev_msg: list[dict[str, Any]] | None = None,
    prev_response_output: list[ResponseOutputMessage] | None = None,
) -> list[dict[str, Any]]:
    """Flatten a Responses turn into the chat messages the template renders.

    The one rule worth stating: instructions do **not** carry over between turns.
    A stored conversation's system messages are dropped and only the current
    request's `instructions` becomes a system message, which is what the OpenAI spec
    says and the opposite of what "replay the previous messages" would do.
    """
    messages: list[dict[str, Any]] = []
    if request_instructions:
        messages.append({"role": "system", "content": request_instructions})

    if prev_msg is not None:
        messages.extend(m for m in prev_msg if m.get("role") != "system")
    if prev_response_output is not None:
        for item in prev_response_output:
            for content in item.content:
                messages.append({"role": "assistant", "content": content.text})

    # A bare string is a single user turn: the Responses API takes plain text without
    # the chat envelope, which is most of why it exists.
    if isinstance(request_input, str):
        messages.append({"role": "user", "content": request_input})
    else:
        messages.extend(_normalise_input_items(request_input))
    return messages


#: Item types that carry no text and belong to a subsystem this build refuses. Named
#: rather than silently dropped, because dropping one would change the prompt.
_UNSUPPORTED_ITEM_TYPES = frozenset(
    {
        "item_reference",
        "function_call",
        "function_call_output",
        "computer_call",
        "computer_call_output",
        "image_generation_call",
        "local_shell_call",
        "local_shell_call_output",
        "code_interpreter_call",
        "file_search_call",
        "web_search_call",
        "custom_tool_call",
        "custom_tool_call_output",
        "reasoning",
    }
)


def _normalise_input_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten each item's `content` down to the text the chat template renders.

    Extending the message list with the raw dicts put the *Python repr of a list of
    part dicts* into the prompt whenever `content` was anything but a bare string --
    which is what the OpenAI SDK sends for everything except the simplest turn. A 200
    with a corrupted prompt, a wrong `usage.input_tokens` and different generated
    text: the plausible wrong answer this project exists to avoid.
    """
    normalised: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"input items must be objects, got {type(item).__name__}")

        item_type = item.get("type")
        if item_type in _UNSUPPORTED_ITEM_TYPES:
            raise NotImplementedError(
                f"Responses API input items of type {item_type!r} are not modelled "
                "by pretending-vllm."
            )

        role = item.get("role")
        if role is None:
            raise ValueError(f"input item is missing a role: {item!r}")

        content = item.get("content")
        if isinstance(content, str) or content is None:
            normalised.append({"role": role, "content": content or ""})
            continue
        if not isinstance(content, list):
            raise ValueError(f"input item content must be a string or a list: {item!r}")

        # `input_text` / `output_text` resolve to their text. An `input_image` part
        # would have to reach the multimodal path, and this endpoint does not wire it
        # -- stringifying it into the prompt would silently invent a caption.
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                raise ValueError(f"unrecognised content part: {part!r}")
            part_type = part.get("type")
            if part_type in ("input_text", "output_text", "text"):
                parts.append(str(part.get("text", "")))
            elif part_type in ("input_image", "input_file", "input_audio"):
                raise NotImplementedError(
                    f"Responses API {part_type!r} content parts are not modelled by "
                    "pretending-vllm: /v1/responses does not reach the multimodal "
                    "path. Use /v1/chat/completions for images."
                )
            else:
                raise ValueError(f"unrecognised content part type: {part_type!r}")
        normalised.append({"role": role, "content": "".join(parts)})
    return normalised


class OpenAIServingResponses:
    """`/v1/responses`, and the two routes that read what it stored."""

    def __init__(self, engine: AsyncLLM, served_model_names: list[str]) -> None:
        self.engine = engine
        self.served_model_names = served_model_names
        self._counter = 0
        #: R16.1. Grafted on by the server once the adapter registry exists, as the
        #: other handlers do.
        self.models: Any = None

        self.enable_store = store_enabled()
        #: Upstream's four containers, same names. Unbounded and never evicted --
        #: upstream says so in three FIXMEs and warns at startup. Bounding them here
        #: would be a *divergence*: a `previous_response_id` from 10,000 requests ago
        #: resolves upstream and would 404 against an LRU.
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()
        self.msg_store: dict[str, list[dict[str, Any]]] = {}
        self.background_tasks: dict[str, asyncio.Task[Any]] = {}

    # --- ids and time --------------------------------------------------------

    def _next_request_id(self) -> str:
        """`resp_` and sixteen hex digits, which is the shape upstream's
        `resp_{random_uuid()}` produces.

        A counter rather than a uuid: B4 requires the same seed and config to produce
        byte-identical output, and the purity lint keeps randomness inside
        `pvllm/sim/`. A client parsing the id sees what it expects; a client relying
        on the entropy does not, and that is documented rather than hidden.
        """
        self._counter += 1
        return f"resp_{self._counter:016x}"

    def _next_item_id(self, prefix: str = "msg") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:016x}"

    def _next_bare_item_id(self) -> str:
        """The id the *streaming* events carry, which has no prefix.

        Upstream mints a bare uuid in `streaming_events.py` and a separate
        `msg_`-prefixed id for the final body, so the two do not match and a client
        correlating them by id finds they never did. Reproduced rather than tidied:
        it is on the wire.
        """
        self._counter += 1
        return f"{self._counter:016x}"

    def _created(self) -> int:
        return int(self.engine.engine_core.clock_time)

    # --- refusals ------------------------------------------------------------

    def _refuse(self, request: ResponsesRequest) -> JSONResponse | None:
        """Everything this build cannot model, named, before the 200 is committed.

        Each of these has a real implementation upstream that reads or writes a token
        stream a simulated model does not produce. Refusing is the discipline the
        whole project rests on: the alternative is a response that looks right and
        silently is not.
        """
        if request.tools:
            return create_error_response(
                "The vLLM Responses API tool-calling path (--tool-call-parser) is not "
                "modelled by pretending-vllm.",
                err_type="NotImplementedError",
                param="tools",
            )
        # Upstream's own validation, raised here rather than in the pydantic
        # validator so the status is 400 rather than FastAPI's 422.
        if request.tool_choice == "required":
            return create_error_response(
                "Tool choice 'required' must be specified with 'tools' parameter.",
                param="tool_choice",
            )
        if (
            isinstance(request.tool_choice, dict)
            and request.tool_choice.get("type") == "function"
        ):
            return create_error_response(
                "Tool choice 'function' not found in 'tools' parameter.",
                param="tool_choice",
            )
        # `reasoning` is NOT refused. It does not select the harmony path: upstream
        # turns `reasoning.effort` into a `reasoning_effort` chat-template kwarg and
        # flips `enable_thinking`, then echoes the field back. On a stock server it is
        # served, so refusing it was a live C5/C7 divergence.
        #
        # `include` is likewise accept-and-ignore. Its only consumer upstream is the
        # logprobs check below; the other five members are validated and then inert.
        if request.truncation not in (None, "disabled"):
            # Upstream turns this into `truncate_prompt_tokens=-1`, which makes the
            # renderer trim the prompt to fit. pvllm has no prompt-truncating
            # renderer, and quietly not truncating would turn a request upstream
            # serves into one that overflows the window.
            return create_error_response(
                'Responses API prompt truncation (`truncation: "auto"`) is not '
                "modelled by pretending-vllm.",
                err_type="NotImplementedError",
                param="truncation",
            )
        if request.cache_salt is not None:
            # pvllm *does* model cache salting -- it is an extra key in the block
            # hash (C3) -- but no entrypoint plumbs it to the engine yet. Accepting
            # it silently would promise a cache partition that does not happen, and
            # the difference is visible in the prefix-cache metrics.
            return create_error_response(
                "`cache_salt` is not plumbed to the engine by pretending-vllm's "
                "Responses endpoint.",
                err_type="NotImplementedError",
                param="cache_salt",
            )
        if request.prompt is not None:
            return create_error_response(
                "Responses API prompt templates are not supported.",
                err_type="NotImplementedError",
                param="prompt",
            )
        if request.is_include_output_logprobs():
            return create_error_response(
                "Responses API output logprobs are not modelled by pretending-vllm: "
                "the simulated model has no logprobs to report.",
                err_type="NotImplementedError",
                param="include",
            )
        if request.previous_input_messages is not None:
            return create_error_response(
                "The vLLM Responses API `previous_input_messages` path requires the "
                "harmony message format, which is not modelled by pretending-vllm.",
                err_type="NotImplementedError",
                param="previous_input_messages",
            )
        if request.enable_response_messages:
            return create_error_response(
                "`enable_response_messages` is not modelled by pretending-vllm.",
                err_type="NotImplementedError",
                param="enable_response_messages",
            )
        if request.background and not self.enable_store:
            # Upstream's own error when the store is off, which is its default.
            return create_error_response(
                "background mode requires the response store to be enabled. Set "
                f"{_STORE_ENV}=1.",
                param="background",
            )
        if request.background:
            return create_error_response(
                "Responses API background mode is not modelled by pretending-vllm: "
                "it would need a task whose progress no simulated clock advances.",
                err_type="NotImplementedError",
                param="background",
            )
        return None

    # --- the request path ----------------------------------------------------

    async def create_responses(
        self, request: ResponsesRequest, raw_request: Request | None = None
    ) -> ResponsesResponse | JSONResponse | AsyncGenerator[Any, None]:
        model_name = request.model or self.served_model_names[0]
        if request.model is not None:
            served, lora_request = (
                self.models.resolve(request.model)
                if self.models is not None
                else (request.model in self.served_model_names, None)
            )
            if not served:
                return model_not_found(
                    request.model,
                    self.served_model_names
                    + (
                        list(self.models.lora_modules)
                        if self.models is not None
                        else []
                    ),
                )
        else:
            # `model` is optional on this endpoint, unlike chat and completions. A
            # request that names none serves the base model rather than 422-ing.
            lora_request = None

        refusal = self._refuse(request)
        if refusal is not None:
            return refusal

        # Upstream accepts `store=True` and quietly drops it when the store is off,
        # rather than erroring -- because the OpenAI SDK sends `store=True` by default
        # and rejecting it would break every unmodified client. This is the one place
        # a silent no-op is the correct port: it is what is on the wire.
        if request.store and not self.enable_store:
            request.store = False

        prev_response: ResponsesResponse | None = None
        if request.previous_response_id is not None:
            async with self.response_store_lock:
                prev_response = self.response_store.get(request.previous_response_id)
            if prev_response is None:
                return create_error_response(
                    f"Response with id '{request.previous_response_id}' not found.",
                    err_type="NotFoundError",
                    status_code=HTTPStatus.NOT_FOUND,
                    param="previous_response_id",
                )

        response_id = request.request_id or self._next_request_id()
        # The client chose this id, so it can collide with one already running -- and
        # the same string is the engine's request id. Caught here so the answer is a
        # 409 rather than a torn-down engine.
        if response_id in self.engine.in_flight_request_ids:
            return create_error_response(
                f"Response with id '{response_id}' is already in progress.",
                err_type="ConflictError",
                status_code=HTTPStatus.CONFLICT,
                param="request_id",
            )
        messages = construct_input_messages(
            request_instructions=request.instructions,
            request_input=request.input,
            prev_msg=self.msg_store.get(prev_response.id) if prev_response else None,
            prev_response_output=prev_response.output if prev_response else None,
        )

        try:
            prompt = render_chat_prompt(
                self.engine.tokenizer,
                messages,
                chat_template_kwargs=_chat_template_kwargs(request),
            )
            sampling_params = request.to_sampling_params(
                self._default_max_tokens(prompt)
            )
        except NotImplementedError as exc:
            return to_error_response(exc)
        except ValueError as exc:
            return create_error_response(str(exc))

        if request.store:
            self.msg_store[response_id] = messages

        if request.stream:
            from pvllm.entrypoints.openai.responses.streaming import stream_responses

            return stream_responses(
                self,
                request,
                response_id=response_id,
                model_name=model_name,
                prompt=prompt,
                sampling_params=sampling_params,
                raw_request=raw_request,
                lora_request=lora_request,
            )
        return await self._complete(
            request,
            response_id=response_id,
            model_name=model_name,
            prompt=prompt,
            sampling_params=sampling_params,
            raw_request=raw_request,
            lora_request=lora_request,
        )

    def _default_max_tokens(self, prompt: str) -> int:
        """What is left of the context window once the prompt is counted.

        Upstream resolves this before generating and echoes it back as
        `max_output_tokens`, so a request that asked for nothing still gets a number,
        and a request that asked for too much is clamped down to this rather than
        rejected.

        A prompt that does not fit at all is upstream's own 400 naming `input`, not
        the engine's context-length error surfacing from three layers down.
        """
        max_model_len = self.engine.model_config.max_model_len
        assert max_model_len is not None
        prompt_len = len(self.engine.tokenizer.encode(prompt))
        if prompt_len >= max_model_len:
            raise ValueError(
                f"The engine prompt length {prompt_len} exceeds the max_model_len "
                f"{max_model_len}. Please reduce prompt."
            )
        return max_model_len - prompt_len

    async def _complete(
        self,
        request: ResponsesRequest,
        *,
        response_id: str,
        model_name: str,
        prompt: str,
        sampling_params: SamplingParams,
        raw_request: Request | None,
        lora_request: Any = None,
    ) -> ResponsesResponse | JSONResponse:
        # Read before generating. Upstream stamps `created_time` on arrival, so this
        # is when the request came in -- not, as it was, when it finished.
        created_at = self._created()
        final: RequestOutput | None = None
        try:
            async for output in self.engine.generate(
                prompt,
                sampling_params,
                response_id,
                priority=request.priority,
                lora_request=lora_request,
            ):
                final = output
                if raw_request is not None and await raw_request.is_disconnected():
                    return create_error_response(
                        "Client disconnected.", err_type="ClientDisconnected"
                    )
        except Exception as exc:
            return to_error_response(exc)

        if final is None:
            return create_error_response("The engine produced no output.")

        response = self._build_response(
            request,
            sampling_params,
            response_id=response_id,
            model_name=model_name,
            created_at=created_at,
            final=final,
        )
        if request.store:
            async with self.response_store_lock:
                self.response_store[response.id] = response
        return response

    def _build_response(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        *,
        response_id: str,
        model_name: str,
        created_at: int,
        final: RequestOutput,
    ) -> ResponsesResponse:
        completion = final.outputs[0]
        num_prompt_tokens = len(final.prompt_token_ids or ())
        num_output_tokens = len(completion.token_ids)

        # `length` is the only finish reason that makes a response *incomplete*
        # rather than complete: the model was still going when the budget ran out.
        incomplete = completion.finish_reason == "length"
        status: ItemStatus = "incomplete" if incomplete else "completed"

        return ResponsesResponse.from_request(
            request,
            sampling_params,
            response_id=response_id,
            model_name=model_name,
            created_at=created_at,
            output=[
                ResponseOutputMessage(
                    id=self._next_item_id(),
                    content=[ResponseOutputText(text=completion.text)],
                    status=status,
                )
            ],
            status=status,
            usage=ResponseUsage(
                input_tokens=num_prompt_tokens,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=final.num_cached_tokens
                ),
                output_tokens=num_output_tokens,
                output_tokens_details=OutputTokensDetails(),
                total_tokens=num_prompt_tokens + num_output_tokens,
            ),
            incomplete_details=(
                IncompleteDetails(reason="max_output_tokens") if incomplete else None
            ),
        )

    # --- the two routes that read the store ----------------------------------

    async def retrieve_responses(
        self, response_id: str
    ) -> ResponsesResponse | JSONResponse:
        async with self.response_store_lock:
            response = self.response_store.get(response_id)
        if response is None:
            return self._not_found(response_id)
        return response

    async def cancel_responses(
        self, response_id: str
    ) -> ResponsesResponse | JSONResponse:
        async with self.response_store_lock:
            response = self.response_store.get(response_id)
            if response is None:
                return self._not_found(response_id)
            if response.status in _TERMINAL:
                return create_error_response(
                    f"Cannot cancel a response with status {response.status}.",
                    param="response_id",
                )
            response.status = "cancelled"
            self.response_store[response_id] = response

        task = self.background_tasks.get(response_id)
        if task is not None and not task.done():
            task.cancel()
        await self.engine.abort(response_id)
        return response

    def _not_found(self, response_id: str) -> JSONResponse:
        """The same 404 whether the id was never seen or the store is simply off.

        Upstream cannot distinguish these either, and the indistinguishability is the
        point: a client against a default-configured vLLM gets exactly this.
        """
        return create_error_response(
            f"Response with id '{response_id}' not found.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="response_id",
        )


def _chat_template_kwargs(request: ResponsesRequest) -> dict[str, Any]:
    """Fold `reasoning.effort` into the chat-template kwargs, as upstream does.

    This is the whole of what `reasoning` means on a non-gpt-oss model: an effort
    string the template may read, and an `enable_thinking` flag for templates that
    require explicit opt-in. `enable_thinking` is only injected when the caller did
    not set it themselves.
    """
    kwargs: dict[str, Any] = dict(request.chat_template_kwargs or {})
    effort = (request.reasoning or {}).get("effort")
    if effort is not None:
        kwargs["reasoning_effort"] = effort
        if "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = effort != "none"
    return kwargs


def render_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """One chat-template path for every endpoint that has messages.

    Shared with `/v1/chat/completions` deliberately: two renderers would let the same
    conversation tokenize to two different lengths, and the token count is what the
    scheduler budgets and what `usage` reports.
    """
    return str(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            **(chat_template_kwargs or {}),
        )
    )
