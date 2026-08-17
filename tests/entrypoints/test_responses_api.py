"""The Responses API. R2.2, R2.3, C5, C7.

Three things distinguish this endpoint from the two that came before it, and each is
a place a client breaks if the port is wrong: the usage field names, the named-event
stream with no `[DONE]`, and a response store that is off by default.

The store default is the subtle one. Stock vLLM ships `VLLM_ENABLE_RESPONSES_API_STORE`
unset, so `GET /v1/responses/{id}` 404s and `previous_response_id` 404s. A pvllm that
stored by default would *succeed* where the real engine fails -- a divergence that
surfaces only once the user swaps real vLLM back in, which is the worst time to find
it. So both positions of the flag are tested.
"""

from __future__ import annotations

import json

import httpx
import pytest
from prometheus_client import CollectorRegistry

from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.entrypoints.openai.api_server import build_app

MODEL = "test-model"


def _config():
    return AsyncEngineArgs(
        model="dense-0.6b",
        served_model_name=MODEL,
        max_model_len=512,
        block_size=16,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        device_card="workstation-24gb",
        disable_log_stats=True,
    ).create_engine_config()


@pytest.fixture
async def client():
    app = build_app(_config(), registry=CollectorRegistry())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client


@pytest.fixture
async def stored_client(monkeypatch):
    """The same server with the response store switched on."""
    monkeypatch.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
    app = build_app(_config(), registry=CollectorRegistry())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client


def sse_frames(text: str) -> list[tuple[str, dict]]:
    """Parse named SSE into `(event_type, payload)`.

    The existing `sse_events` helper reads only `data:` lines, which is enough for
    chat completions and loses the whole dispatch key here.
    """
    frames: list[tuple[str, dict]] = []
    event_type: str | None = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event_type = line[len("event: ") :]
        elif line.startswith("data: "):
            assert event_type is not None, "a data line arrived before its event line"
            frames.append((event_type, json.loads(line[len("data: ") :])))
            event_type = None
    return frames


# --- the wire schema (C5) ---------------------------------------------------


async def test_a_plain_text_turn_returns_the_documented_body(client):
    """The Responses API takes bare text with no chat envelope, which is most of why
    it exists."""
    response = await client.post(
        "/v1/responses", json={"model": MODEL, "input": "hello", "max_output_tokens": 4}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "response"
    assert body["id"].startswith("resp_")
    assert body["model"] == MODEL
    assert body["status"] in {"completed", "incomplete"}

    # One assistant message, one output_text part.
    assert len(body["output"]) == 1
    item = body["output"][0]
    assert item["type"] == "message"
    assert item["role"] == "assistant"
    assert item["id"].startswith("msg_")
    assert item["content"][0]["type"] == "output_text"
    assert isinstance(item["content"][0]["text"], str)


async def test_usage_uses_input_output_not_prompt_completion(client):
    """C5. The names differ from every other endpoint here, and a client reading
    `prompt_tokens` off a Responses body gets a KeyError against real vLLM too."""
    response = await client.post(
        "/v1/responses", json={"model": MODEL, "input": "hello", "max_output_tokens": 4}
    )
    usage = response.json()["usage"]
    assert set(usage) == {
        "input_tokens",
        "input_tokens_details",
        "output_tokens",
        "output_tokens_details",
        "total_tokens",
    }
    assert "prompt_tokens" not in usage
    assert "completion_tokens" not in usage
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["output_tokens"] == 4


async def test_every_declared_field_is_present_including_nulls(client):
    """Upstream dumps without `exclude_none`, so a client sees the full key set.
    Dropping the nulls would be invisible in a happy-path test and wrong on the wire.
    """
    body = (
        await client.post("/v1/responses", json={"model": MODEL, "input": "hi"})
    ).json()
    for field in (
        "incomplete_details",
        "instructions",
        "metadata",
        "max_tool_calls",
        "previous_response_id",
        "prompt",
        "reasoning",
        "text",
        "top_logprobs",
        "user",
        "kv_transfer_params",
        "input_messages",
        "output_messages",
    ):
        assert field in body, f"{field} missing from the response body"


async def test_the_response_echoes_resolved_sampling_not_what_was_sent(client):
    """A request that sets nothing still gets numbers back: upstream echoes the
    *resolved* sampling params, so `max_output_tokens` is what is left of the context
    window rather than the `null` the client sent."""
    body = (
        await client.post("/v1/responses", json={"model": MODEL, "input": "hi"})
    ).json()
    assert body["temperature"] == 1.0
    assert body["top_p"] == 1.0
    assert body["presence_penalty"] == 0.0
    assert body["frequency_penalty"] == 0.0
    assert isinstance(body["max_output_tokens"], int)
    assert 0 < body["max_output_tokens"] < 512


async def test_tool_choice_is_rewritten_to_none_when_no_tools_are_given(client):
    """Upstream rewrites it in a pre-validator, and it is the rewritten value the
    response echoes -- so a plain request comes back saying "none" despite the
    request's default being "auto"."""
    body = (
        await client.post("/v1/responses", json={"model": MODEL, "input": "hi"})
    ).json()
    assert body["tool_choice"] == "none"


async def test_asking_for_a_required_tool_without_tools_is_an_error(client):
    response = await client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "hi", "tool_choice": "required"},
    )
    assert response.status_code == 400


async def test_model_may_be_omitted(client):
    """`model` is optional on this endpoint, unlike chat and completions. A client
    that omits it must not get a 422."""
    response = await client.post("/v1/responses", json={"input": "hi"})
    assert response.status_code == 200
    assert response.json()["model"] == MODEL


async def test_an_unknown_model_is_still_a_404(client):
    response = await client.post(
        "/v1/responses", json={"model": "not-served", "input": "hi"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "NotFoundError"


async def test_an_unknown_field_is_accepted_rather_than_rejected(client):
    """`extra="allow"`, as upstream. This is what lets a client written against a
    newer OpenAI SDK keep working against an older server."""
    response = await client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "hi", "some_field_from_the_future": 7},
    )
    assert response.status_code == 200


async def test_a_client_may_choose_its_own_response_id(client):
    """`request_id` becomes `id` verbatim. Dropping the field would silently hand the
    client a different id than it asked for."""
    body = (
        await client.post(
            "/v1/responses",
            json={"model": MODEL, "input": "hi", "request_id": "resp_chosen"},
        )
    ).json()
    assert body["id"] == "resp_chosen"


async def test_instructions_become_a_system_turn(client):
    """Observable through the token count: a longer prompt is a bigger
    `input_tokens`, which is the only handle on prompt construction from outside."""
    plain = (
        await client.post(
            "/v1/responses",
            json={"model": MODEL, "input": "hi", "max_output_tokens": 1},
        )
    ).json()
    instructed = (
        await client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": "hi",
                "instructions": "You are a helpful assistant who is very thorough.",
                "max_output_tokens": 1,
            },
        )
    ).json()
    assert instructed["usage"]["input_tokens"] > plain["usage"]["input_tokens"]


async def test_running_out_of_budget_reports_incomplete(client):
    """`length` is the one finish reason that makes a response incomplete rather than
    complete, and it carries a reason a client can branch on."""
    body = (
        await client.post(
            "/v1/responses",
            json={"model": MODEL, "input": "hi", "max_output_tokens": 2},
        )
    ).json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    assert body["output"][0]["status"] == "incomplete"


# --- the event stream (R2.3, C5) --------------------------------------------


async def test_the_stream_is_the_nine_documented_events_in_order(client):
    response = await client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "hi", "stream": True, "max_output_tokens": 3},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = sse_frames(response.text)
    types = [event_type for event_type, _ in frames]

    assert types[:4] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert types[-4:] == [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.incomplete",
    ]
    # Everything between the prologue and the epilogue is a delta, one per step.
    assert set(types[4:-4]) == {"response.output_text.delta"}
    assert types.count("response.output_text.delta") == 3


async def test_the_stream_sends_no_done_sentinel(client):
    """Chat completions ends with `data: [DONE]`; this does not. A client that waits
    for one against a real vLLM waits forever, so pvllm must not send one either."""
    response = await client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "hi", "stream": True, "max_output_tokens": 2},
    )
    assert "[DONE]" not in response.text
    assert response.text.rstrip().endswith("}")


async def test_the_streaming_item_id_is_bare_and_differs_from_the_final_one(client):
    """An upstream wart, pinned so a later cleanup cannot quietly "fix" it.

    The streamed item carries a bare id; the same message in the terminal event's
    response body carries a fresh `msg_`-prefixed one. A client correlating the two
    finds they never match -- against real vLLM too, which is the point.
    """
    response = await client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "hi", "stream": True, "max_output_tokens": 2},
    )
    frames = sse_frames(response.text)
    added = next(p for t, p in frames if t == "response.output_item.added")
    terminal = frames[-1][1]["response"]

    streamed_id = added["item"]["id"]
    final_id = terminal["output"][0]["id"]
    assert not streamed_id.startswith("msg_")
    assert final_id.startswith("msg_")
    assert streamed_id != final_id
    # Every mid-stream event agrees with the streamed id, whatever the final one says.
    for _, payload in frames:
        if "item_id" in payload:
            assert payload["item_id"] == streamed_id


async def test_sequence_numbers_are_global_and_monotonic(client):
    """One counter for the whole response, not one per item -- a client uses it to
    detect a dropped frame."""
    response = await client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "hi", "stream": True, "max_output_tokens": 3},
    )
    numbers = [payload["sequence_number"] for _, payload in sse_frames(response.text)]
    assert numbers == list(range(len(numbers)))


async def test_every_frame_names_its_type_in_both_places(client):
    """The `event:` line is the dispatch key, and the payload repeats it. A client may
    read either."""
    response = await client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "hi", "stream": True, "max_output_tokens": 2},
    )
    for event_type, payload in sse_frames(response.text):
        assert payload["type"] == event_type


async def test_the_streamed_text_matches_the_final_response(client):
    """The deltas concatenate to what the terminal event reports, which is the one
    invariant a streaming client actually depends on."""
    response = await client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "hi", "stream": True, "max_output_tokens": 4},
    )
    frames = sse_frames(response.text)
    streamed = "".join(
        payload["delta"]
        for event_type, payload in frames
        if event_type == "response.output_text.delta"
    )
    terminal = frames[-1][1]["response"]
    assert terminal["output"][0]["content"][0]["text"] == streamed
    assert terminal["usage"]["output_tokens"] == 4


# --- refusals, by name (C7) -------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"tools": [{"type": "function", "name": "f"}]}, "tools"),
        ({"reasoning": {"effort": "low"}}, "reasoning"),
        ({"include": ["reasoning.encrypted_content"]}, "reasoning"),
        ({"prompt": {"id": "pmpt_1"}}, "prompt"),
        ({"include": ["message.output_text.logprobs"]}, "include"),
        ({"previous_input_messages": [{"role": "user"}]}, "previous_input_messages"),
        ({"enable_response_messages": True}, "enable_response_messages"),
    ],
)
async def test_an_unmodelled_feature_is_refused_by_name(client, payload, param):
    """R-discipline: a dropped upstream path raises rather than silently no-opping.
    Answering a tool-calling request with a plain assistant message is the kind of
    plausible wrong answer that costs more than an error would."""
    response = await client.post(
        "/v1/responses", json={"model": MODEL, "input": "hi", **payload}
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "NotImplementedError"
    assert error["param"] == param


# --- the store, in both positions -------------------------------------------


async def test_store_is_off_by_default_and_the_request_still_succeeds(client):
    """The silent downgrade. The OpenAI SDK sends `store=true` by default, so
    rejecting it would break every unmodified client -- upstream accepts it and
    quietly drops it, and this is the one place a silent no-op is the correct port."""
    response = await client.post(
        "/v1/responses", json={"model": MODEL, "input": "hi", "store": True}
    )
    assert response.status_code == 200
    # Nothing on the wire reveals the downgrade: there is no `store` field on the
    # response at all.
    assert "store" not in response.json()
    # And the response is not retrievable, exactly as on a stock vLLM.
    stored = await client.get(f"/v1/responses/{response.json()['id']}")
    assert stored.status_code == 404


async def test_an_unknown_previous_response_id_is_a_404(client):
    response = await client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "hi", "previous_response_id": "resp_nope"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["param"] == "previous_response_id"


async def test_background_is_refused_when_the_store_is_off(client):
    """Upstream 400s this too when the store is off, so refusing is parity rather
    than a limitation."""
    response = await client.post(
        "/v1/responses", json={"model": MODEL, "input": "hi", "background": True}
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "background"


async def test_with_the_store_on_a_response_is_retrievable(stored_client):
    created = await stored_client.post(
        "/v1/responses", json={"model": MODEL, "input": "hi", "max_output_tokens": 2}
    )
    response_id = created.json()["id"]
    fetched = await stored_client.get(f"/v1/responses/{response_id}")
    assert fetched.status_code == 200
    assert fetched.json() == created.json()


async def test_with_the_store_on_a_prior_turn_is_replayed_into_the_prompt(
    stored_client,
):
    """The point of `previous_response_id`: the follow-up carries the conversation, so
    its prompt is strictly longer than the same input sent cold."""
    first = await stored_client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "the first turn", "max_output_tokens": 4},
    )
    first_id = first.json()["id"]

    cold = await stored_client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "the second turn", "max_output_tokens": 1},
    )
    warm = await stored_client.post(
        "/v1/responses",
        json={
            "model": MODEL,
            "input": "the second turn",
            "previous_response_id": first_id,
            "max_output_tokens": 1,
        },
    )
    assert warm.status_code == 200
    assert warm.json()["previous_response_id"] == first_id
    assert warm.json()["usage"]["input_tokens"] > cold.json()["usage"]["input_tokens"]


async def test_instructions_do_not_carry_across_turns(stored_client):
    """Per the OpenAI spec, and the opposite of what replaying the stored messages
    would do: only the *current* request's instructions become a system turn."""
    instruction = "A very long system instruction that costs a great many tokens."

    async def follow_up_after(first_payload: dict) -> int:
        first = await stored_client.post("/v1/responses", json=first_payload)
        follow_up = await stored_client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": "two",
                "previous_response_id": first.json()["id"],
                "max_output_tokens": 1,
            },
        )
        return int(follow_up.json()["usage"]["input_tokens"])

    base = {"model": MODEL, "input": "one", "max_output_tokens": 1}
    without = await follow_up_after(base)
    with_instructions = await follow_up_after({**base, "instructions": instruction})

    # Two turns identical but for a system message on the *first* one. If the stored
    # system turn carried over, the second follow-up would be longer by the length of
    # that instruction; the whole point is that it is not.
    assert with_instructions == without


async def test_a_streamed_response_is_stored_too(stored_client):
    response = await stored_client.post(
        "/v1/responses",
        json={"model": MODEL, "input": "hi", "stream": True, "max_output_tokens": 2},
    )
    terminal = sse_frames(response.text)[-1][1]["response"]
    fetched = await stored_client.get(f"/v1/responses/{terminal['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["output"] == terminal["output"]


async def test_cancelling_a_finished_response_is_refused(stored_client):
    created = await stored_client.post(
        "/v1/responses", json={"model": MODEL, "input": "hi", "max_output_tokens": 2}
    )
    cancelled = await stored_client.post(f"/v1/responses/{created.json()['id']}/cancel")
    assert cancelled.status_code == 400
    assert "Cannot cancel" in cancelled.json()["error"]["message"]


async def test_cancelling_an_unknown_response_is_a_404(stored_client):
    response = await stored_client.post("/v1/responses/resp_nope/cancel")
    assert response.status_code == 404


# --- C6: this endpoint adds no metric ---------------------------------------


async def test_the_responses_endpoint_adds_no_prometheus_metric(client):
    """C6 is a golden test, and a new endpoint is exactly the kind of change that
    tempts a new counter. Upstream declares none for this route, so a metric here
    would be a divergence rather than an improvement."""
    before = {
        line.split("{")[0].split(" ")[0]
        for line in (await client.get("/metrics")).text.splitlines()
        if line and not line.startswith("#")
    }
    await client.post(
        "/v1/responses", json={"model": MODEL, "input": "hi", "max_output_tokens": 2}
    )
    after = {
        line.split("{")[0].split(" ")[0]
        for line in (await client.get("/metrics")).text.splitlines()
        if line and not line.startswith("#")
    }
    assert after == before
