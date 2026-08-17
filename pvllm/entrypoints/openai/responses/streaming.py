"""The Responses API event stream. R2.3, C5.

Upstream: vllm/entrypoints/openai/responses/streaming_events.py
Tier: B

This does *not* stream chat-completion-style deltas. It streams typed, named events
carrying a single global monotonic `sequence_number`, framed as

    event: <type>\\n
    data: <compact json>\\n
    \\n

and -- unlike `/v1/chat/completions` -- it sends **no** `data: [DONE]` sentinel. The
stream simply ends after `response.completed`. A client that waits for `[DONE]` here
hangs, which is exactly why it is worth a test rather than a comment.

Upstream's file covers twenty-two event types; the rest are tool, reasoning, MCP or
code-interpreter events reached only through a parser pvllm does not have. The nine
below are the whole sequence for a text generation, which is the only sequence this
build can produce -- and a generation that emits no text produces only three of them,
because the output item and its content part are opened lazily on the first delta.

There is no `response.incomplete`. An incomplete response still ends with
`response.completed`, carrying `status: "incomplete"` in the body; upstream has no
such event type and its own union could not parse one.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastapi import Request

from pvllm.entrypoints.openai.responses.protocol import (
    IncompleteDetails,
    InputTokensDetails,
    ItemStatus,
    OutputTokensDetails,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponseStatus,
    ResponseUsage,
)
from pvllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from pvllm.entrypoints.openai.responses.serving import OpenAIServingResponses


def _event(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": event_type, "payload": payload}


async def stream_responses(
    serving: OpenAIServingResponses,
    request: ResponsesRequest,
    *,
    response_id: str,
    model_name: str,
    prompt: str,
    sampling_params: SamplingParams,
    raw_request: Request | None,
    lora_request: Any = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield `{"type", "payload"}` pairs; the router does the SSE framing.

    `sequence_number` is stamped here, at the yield site, from one counter shared by
    every event -- it is global and monotonic across the whole response, not per item.
    """
    sequence = 0

    def stamp(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Stamp the sequence number and re-sort into the openai models' key order.

        Those models are Stainless-generated: required fields alphabetically, then
        optional ones. So `sequence_number` and `type` are not simply appended --
        `output_text.done` carries `text` *after* `sequence_number`, and appending
        would put it before. Sorting the required keys reproduces that order without
        having to hand-place every field.
        """
        nonlocal sequence
        payload["sequence_number"] = sequence
        payload["type"] = event_type
        sequence += 1
        return _event(event_type, dict(sorted(payload.items())))

    # Deliberately *not* the id the final response's output item carries; see
    # `_next_bare_item_id`.
    item_id = serving._next_bare_item_id()

    # Captured ONCE, before generation. Reading the engine clock per event gave the
    # same response two different `created_at` values, because the sim clock advances
    # while the model runs. Upstream threads a single `created_time` through every
    # event of a response.
    created_at = serving._created()

    def envelope(status: ResponseStatus, **overrides: Any) -> dict[str, Any]:
        response = ResponsesResponse.from_request(
            request,
            sampling_params,
            response_id=response_id,
            model_name=model_name,
            created_at=created_at,
            output=overrides.pop("output", []),
            status=status,
            usage=overrides.pop("usage", None),
            incomplete_details=overrides.pop("incomplete_details", None),
        )
        return response.model_dump(mode="json", by_alias=True)

    # The envelope is computed once and reused by both opening events, as upstream
    # does -- and, with `created_at` captured before generation starts, it is the same
    # `created_at` the terminal event carries.
    opening = envelope("in_progress")
    yield stamp("response.created", {"response": opening})
    yield stamp("response.in_progress", {"response": opening})

    text = ""
    num_prompt_tokens = 0
    num_output_tokens = 0
    num_cached_tokens = 0
    finish_reason: str | None = None
    opened = False

    async for output in serving.engine.generate(
        prompt,
        sampling_params,
        response_id,
        priority=request.priority,
        lora_request=lora_request,
    ):
        if raw_request is not None and await raw_request.is_disconnected():
            await serving.engine.abort(response_id)
            return
        num_prompt_tokens = len(output.prompt_token_ids or ())
        num_cached_tokens = output.num_cached_tokens
        for completion in output.outputs:
            if completion.text:
                # Opened lazily, on the first non-empty delta. A generation that
                # produces no text therefore emits three events, not nine: upstream
                # never opens the item or the content part, so there is nothing to
                # close either.
                if not opened:
                    opened = True
                    yield stamp(
                        "response.output_item.added",
                        {
                            "item": {
                                "id": item_id,
                                "content": [],
                                "role": "assistant",
                                "status": "in_progress",
                                "type": "message",
                                "phase": None,
                            },
                            "output_index": 0,
                        },
                    )
                    yield stamp(
                        "response.content_part.added",
                        {
                            "content_index": 0,
                            "item_id": item_id,
                            "output_index": 0,
                            "part": {
                                "annotations": [],
                                "text": "",
                                "type": "output_text",
                                # `[]` here and `null` on the closing part. Upstream
                                # builds the two through different constructors.
                                "logprobs": [],
                            },
                        },
                    )
                text += completion.text
                num_output_tokens += len(completion.token_ids)
                yield stamp(
                    "response.output_text.delta",
                    {
                        "content_index": 0,
                        "delta": completion.text,
                        "item_id": item_id,
                        "logprobs": [],
                        "output_index": 0,
                    },
                )
            if completion.finish_reason is not None:
                finish_reason = completion.finish_reason

    incomplete = finish_reason == "length"
    status: ItemStatus = "incomplete" if incomplete else "completed"

    if opened:
        yield stamp(
            "response.output_text.done",
            {
                "content_index": 0,
                "item_id": item_id,
                "logprobs": [],
                "output_index": 0,
                "text": text,
            },
        )
        closing_part: dict[str, Any] = {
            "annotations": [],
            "text": text,
            "type": "output_text",
            # `null` on the closing part where the opening one carried `[]`:
            # upstream builds the two through different constructors.
            "logprobs": None,
        }
        yield stamp(
            "response.content_part.done",
            {
                "content_index": 0,
                "item_id": item_id,
                "output_index": 0,
                "part": closing_part,
            },
        )
        yield stamp(
            "response.output_item.done",
            {
                "item": {
                    "id": item_id,
                    "content": [closing_part],
                    "role": "assistant",
                    # Hardcoded, even when the response as a whole is incomplete: the
                    # item finished, and it is the *response* that ran out of budget.
                    "status": "completed",
                    "type": "message",
                    "phase": None,
                    "summary": [],
                },
                "output_index": 0,
            },
        )

    final_output = [
        ResponseOutputMessage(
            id=serving._next_item_id(),
            content=[ResponseOutputText(text=text)] if text else [],
            status=status,
        )
    ]
    usage = ResponseUsage(
        input_tokens=num_prompt_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=num_cached_tokens),
        output_tokens=num_output_tokens,
        output_tokens_details=OutputTokensDetails(),
        total_tokens=num_prompt_tokens + num_output_tokens,
    )
    final = ResponsesResponse.from_request(
        request,
        sampling_params,
        response_id=response_id,
        model_name=model_name,
        created_at=created_at,
        output=final_output,
        status=status,
        usage=usage,
        incomplete_details=(
            IncompleteDetails(reason="max_output_tokens") if incomplete else None
        ),
    )
    if request.store:
        async with serving.response_store_lock:
            serving.response_store[final.id] = final

    # Always `response.completed`, even when the response is incomplete. vLLM v0.27.1
    # has no `response.incomplete` event: `ResponseIncompleteEvent` is not a member of
    # its `StreamingResponsesResponse` union, so its own TypeAdapter could not parse
    # such a frame, and a client dispatching on the type would fall off the end. The
    # incompleteness travels *inside* this event, as `response.status`.
    yield stamp(
        "response.completed",
        {"response": final.model_dump(mode="json", by_alias=True)},
    )
