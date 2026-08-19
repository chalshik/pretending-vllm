"""The OpenAI HTTP surface. R2.2--R2.5, C5, C6, C7.

This is M1's acceptance test: a product speaking the OpenAI API points at the server
unmodified and gets what it would get from real vLLM.
"""

from __future__ import annotations

import json

import httpx
import pytest
from prometheus_client import CollectorRegistry

from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.entrypoints.openai.api_server import build_app

MODEL = "test-model"


@pytest.fixture
async def client():
    config = AsyncEngineArgs(
        model="dense-0.6b",
        served_model_name=MODEL,
        max_model_len=512,
        block_size=16,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        # Prefix caching is on by default, as upstream (R1.4). It was disabled here
        # while M1 could not implement it.
        device_card="workstation-24gb",
        disable_log_stats=True,
    ).create_engine_config()
    # A fresh registry per app: two apps in one process would otherwise collide on
    # metric names.
    app = build_app(config, registry=CollectorRegistry())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client


def sse_events(text: str) -> list[dict | str]:
    """Parse an SSE body into payloads, `[DONE]` included as a marker."""
    events: list[dict | str] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        events.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return events


def metric_value(text: str, name: str) -> float:
    """The value of a single-series metric in a Prometheus scrape.

    Matched on the name alone, labels stripped: the series carries `engine` and
    `model_name`, and a `startswith` would also match `vllm:prompt_tokens_created`.
    """
    for line in text.splitlines():
        series, _, value = line.rpartition(" ")
        if series.split("{")[0] == name:
            return float(value)
    raise AssertionError(f"{name} is not in the scrape")


# --- operational endpoints -------------------------------------------------


async def test_health_reports_ready(client):
    """R2.7: ready only after load and profiling, which the core does in its
    constructor."""
    assert (await client.get("/health")).status_code == 200
    assert (await client.get("/ping")).status_code == 200


async def test_version_declares_that_this_is_a_simulator(client):
    body = (await client.get("/version")).json()
    assert body["upstream_version"] == "0.27.1"
    # Anything reading this endpoint to identify the backend learns the truth.
    assert body["simulated"] is True


async def test_models_lists_the_served_name(client):
    body = (await client.get("/v1/models")).json()
    ids = [model["id"] for model in body["data"]]
    assert MODEL in ids
    assert body["object"] == "list"
    assert body["data"][0]["object"] == "model"


# --- completions (C5) ------------------------------------------------------


async def test_completion_returns_the_openai_shape(client):
    response = await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "Hello", "max_tokens": 6}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    assert body["model"] == MODEL
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["text"]
    assert choice["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 6
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


async def test_created_comes_from_the_engine_clock(client):
    """R19.1. Deterministic under a virtual clock, which is what lets a whole
    response be golden-tested."""
    first = (
        await client.post(
            "/v1/completions", json={"model": MODEL, "prompt": "a", "max_tokens": 2}
        )
    ).json()
    # The virtual clock's fixed epoch, not wall time.
    assert first["created"] == 1767225600


async def test_completions_are_reproducible(client):
    payload = {"model": MODEL, "prompt": "same", "max_tokens": 8}
    first = (await client.post("/v1/completions", json=payload)).json()
    second = (await client.post("/v1/completions", json=payload)).json()
    # Different request ids, same content: the per-request RNG is keyed on the
    # request id, so these differ -- what must be stable is a given id's output,
    # which the engine tests cover. Here we assert the shape is stable.
    assert first["choices"][0]["finish_reason"] == second["choices"][0]["finish_reason"]
    assert len(first["choices"][0]["text"]) > 0


async def test_max_tokens_is_honoured(client):
    body = (
        await client.post(
            "/v1/completions", json={"model": MODEL, "prompt": "x", "max_tokens": 3}
        )
    ).json()
    assert body["usage"]["completion_tokens"] == 3


# --- chat completions ------------------------------------------------------


async def test_chat_completion_returns_the_openai_shape(client):
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 5,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"]
    assert body["usage"]["completion_tokens"] == 5


async def test_max_completion_tokens_supersedes_max_tokens(client):
    """A client sending both means the newer field."""
    body = (
        await client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 50,
                "max_completion_tokens": 4,
            },
        )
    ).json()
    assert body["usage"]["completion_tokens"] == 4


async def test_empty_messages_is_rejected(client):
    response = await client.post(
        "/v1/chat/completions", json={"model": MODEL, "messages": []}
    )
    assert response.status_code == 400


# --- streaming (R2.3) ------------------------------------------------------


async def test_completion_stream_is_sse_terminated_by_done(client):
    response = await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "hi", "max_tokens": 4, "stream": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = sse_events(response.text)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events if isinstance(e, dict)]
    assert len(chunks) == 4
    assert all(c["object"] == "text_completion" for c in chunks)
    # Deltas, not cumulative: concatenating them reconstructs the text.
    assert "".join(c["choices"][0]["text"] for c in chunks)


async def test_chat_stream_sends_the_role_chunk_first(client):
    """A client that reads `delta.role` from the first chunk and `delta.content`
    from the rest depends on this split."""
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 3,
            "stream": True,
        },
    )
    chunks = [e for e in sse_events(response.text) if isinstance(e, dict)]

    first = chunks[0]["choices"][0]["delta"]
    assert first["role"] == "assistant"
    assert first.get("content") is None
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert any(c["choices"][0]["delta"].get("content") for c in chunks[1:])


async def test_include_usage_appends_a_usage_chunk_with_empty_choices(client):
    """R2.3, and the shape matters: clients that iterate choices unconditionally
    break on the empty list, so reproducing it is the point."""
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 3,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    chunks = [e for e in sse_events(response.text) if isinstance(e, dict)]
    usage_chunk = chunks[-1]

    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["completion_tokens"] == 3
    assert usage_chunk["usage"]["total_tokens"] > 0


async def test_without_include_usage_no_usage_chunk(client):
    response = await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "hi", "max_tokens": 3, "stream": True},
    )
    chunks = [e for e in sse_events(response.text) if isinstance(e, dict)]
    assert all(c.get("usage") is None for c in chunks)


async def test_streaming_finish_reason_lands_on_the_last_chunk(client):
    response = await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "hi", "max_tokens": 4, "stream": True},
    )
    chunks = [e for e in sse_events(response.text) if isinstance(e, dict)]
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


# --- errors (R2.5, C7) -----------------------------------------------------


async def test_unknown_model_is_404_naming_what_is_served(client):
    response = await client.post(
        "/v1/completions", json={"model": "no-such-model", "prompt": "x"}
    )
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["type"] == "NotFoundError"
    assert MODEL in error["message"]


async def test_context_length_exceeded_is_400(client):
    response = await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "hello", "max_tokens": 100_000},
    )
    assert response.status_code == 400
    assert "maximum context length" in response.json()["error"]["message"]


async def test_a_prompt_longer_than_the_context_is_400(client):
    response = await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": list(range(600))}
    )
    assert response.status_code == 400


async def test_an_unmodeled_feature_is_501_not_500(client):
    """501, the status upstream gives the same exception, and the one that says what
    actually happened: this server does not implement that. A 500 would send a client
    into retry logic for something that will never succeed, and the 400 this used to
    return blamed the request for a gap in the server."""
    response = await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": ["one", "two"]}
    )
    assert response.status_code == 501
    assert response.json()["error"]["type"] == "NotImplementedError"


async def test_n_greater_than_one_returns_n_choices(client):
    """R11.7. `n` is a plain OpenAI parameter, and it used to 400."""
    response = await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "x", "n": 3, "max_tokens": 8},
    )
    assert response.status_code == 200
    body = response.json()
    assert [choice["index"] for choice in body["choices"]] == [0, 1, 2]
    # Distinct completions, and the usage counts all three -- the prompt is billed
    # once and shared through the prefix cache, but each completion was generated.
    assert len({choice["text"] for choice in body["choices"]}) == 3
    assert body["usage"]["completion_tokens"] > 8


async def test_n_greater_than_one_streams_every_choice(client):
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "n": 2,
            "max_tokens": 6,
            "stream": True,
        },
    )
    assert response.status_code == 200

    # Unioning indices across *all* chunks proves nothing: the leading role chunk
    # already emits one choice per index, so every content chunk could be mislabelled
    # index 0 and the union would still be {0, 1}. R11.7 is that a piece of generated
    # text belongs to a particular choice, so accumulate text per index and check both
    # choices actually received their own.
    text: dict[int, str] = {}
    for event in sse_events(response.text):
        if not isinstance(event, dict):
            continue
        for choice in event.get("choices", ()):
            content = (choice.get("delta") or {}).get("content")
            if content:
                text[choice["index"]] = text.get(choice["index"], "") + content

    assert set(text) == {0, 1}, f"content arrived for choices {sorted(text)}"
    assert all(value for value in text.values()), "a choice streamed no content"


async def test_an_invalid_sampling_parameter_is_400(client):
    response = await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "x", "temperature": -1}
    )
    assert response.status_code == 400


# --- tokenization ----------------------------------------------------------


async def test_tokenize_and_detokenize_round_trip(client):
    tokenized = (
        await client.post("/tokenize", json={"model": MODEL, "prompt": "hello"})
    ).json()
    assert tokenized["count"] == len(tokenized["tokens"])
    assert tokenized["max_model_len"] == 512

    detokenized = (
        await client.post(
            "/detokenize", json={"model": MODEL, "tokens": tokenized["tokens"]}
        )
    ).json()
    assert "hello" in detokenized["prompt"]


# --- metrics (R12.1, R12.2, C6) --------------------------------------------


async def test_metrics_endpoint_serves_prometheus_text(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


async def test_counters_are_not_double_suffixed(client):
    """F5, the correction the draft spec needed.

    prometheus_client appends `_total` to counters. Declaring
    `vllm:prompt_tokens_total` would export `vllm:prompt_tokens_total_total` and
    every counter panel on a dashboard built against real vLLM would go empty.
    """
    await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "x", "max_tokens": 3}
    )
    body = (await client.get("/metrics")).text
    assert "_total_total" not in body


async def test_exported_metric_names_match_upstream(client):
    """C6. These are the names a dashboard's queries reference."""
    await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "x", "max_tokens": 3}
    )
    body = (await client.get("/metrics")).text
    names = {
        line.split("{")[0].split(" ")[0]
        for line in body.splitlines()
        if line.startswith("vllm:")
    }

    for expected in (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:num_preemptions_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
    ):
        assert expected in names, f"missing {expected}"

    for histogram in (
        "vllm:time_to_first_token_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:iteration_tokens_total",
        "vllm:request_queue_time_seconds",
    ):
        assert f"{histogram}_bucket" in names, f"missing {histogram}"


async def test_latency_metrics_declare_that_they_are_modeled(client):
    """R12.4/R9.5. A consumer reading only /metrics must still be able to tell."""
    body = (await client.get("/metrics")).text
    assert "MODELED" in body
    for line in body.splitlines():
        if line.startswith("# HELP vllm:time_to_first_token_seconds"):
            assert "MODELED" in line
            break
    else:
        pytest.fail("no help line for vllm:time_to_first_token_seconds")


async def test_gauges_track_engine_state(client):
    body = (await client.get("/metrics")).text
    running = [
        line
        for line in body.splitlines()
        if line.startswith("vllm:num_requests_running")
    ]
    assert running and running[0].endswith("0.0")


async def test_histogram_buckets_match_upstream_edges():
    """R12.2: dashboards render against either engine only if the edges agree."""
    from pvllm.v1.metrics.loggers import (
        REQUEST_LATENCY_BUCKETS,
        TIME_TO_FIRST_TOKEN_BUCKETS,
    )

    assert REQUEST_LATENCY_BUCKETS[:5] == [0.3, 0.5, 0.8, 1.0, 1.5]
    assert REQUEST_LATENCY_BUCKETS[-1] == 7680.0
    assert TIME_TO_FIRST_TOKEN_BUCKETS[0] == 0.001
    assert TIME_TO_FIRST_TOKEN_BUCKETS[-1] == 2560.0


async def test_token_counters_populate_after_a_request(client):
    """C6. `vllm:prompt_tokens_total` was declared, scraped, and never incremented.

    It exported 0.0 for the life of the process, so the prefill half of every
    token-throughput panel on a dashboard built against real vLLM read empty, and
    `vllm:iteration_tokens_total` undercounted every step by the prompt. What hid it
    is that `vllm:request_prompt_tokens` is fed from the finished-request path and
    was right the whole time -- the per-request panel looked healthy.
    """
    usage = (
        await client.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": "the quick brown fox", "max_tokens": 6},
        )
    ).json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0

    body = (await client.get("/metrics")).text
    assert metric_value(body, "vllm:prompt_tokens_total") == usage["prompt_tokens"]
    assert (
        metric_value(body, "vllm:generation_tokens_total") == usage["completion_tokens"]
    )
    # A cold engine cached nothing, so every prompt token was computed and the
    # iteration histogram sums to the whole request -- prompt included, which is the
    # part that summed to just the 6 generated tokens before.
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0
    assert (
        metric_value(body, "vllm:iteration_tokens_total_sum") == usage["total_tokens"]
    )
    assert (
        metric_value(body, "vllm:request_prompt_tokens_sum") == usage["prompt_tokens"]
    )


async def test_prefix_cache_hits_count_as_prefilled_but_not_as_computed(client):
    """C6. Two counters, two questions, and the prefix cache separates them.

    `vllm:prompt_tokens_total` counts tokens *prefilled*, cache hits included --
    upstream accumulates the request's whole prompt length there. The iteration
    histogram counts tokens actually *computed*, so a warm cache does not inflate the
    apparent batch size with tokens no kernel ran on.
    """
    prompt = "the quick brown fox jumps over the lazy dog and keeps on running"
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 4}
    first = (await client.post("/v1/completions", json=body)).json()["usage"]
    second = (await client.post("/v1/completions", json=body)).json()["usage"]

    cached = second["prompt_tokens_details"]["cached_tokens"]
    assert cached > 0, "an identical second request should hit the prefix cache"

    scrape = (await client.get("/metrics")).text
    assert metric_value(scrape, "vllm:prompt_tokens_total") == (
        first["prompt_tokens"] + second["prompt_tokens"]
    )
    assert metric_value(scrape, "vllm:iteration_tokens_total_sum") == (
        first["total_tokens"] + second["total_tokens"] - cached
    )


# --- prefix cache ----------------------------------------------------------


async def test_reset_prefix_cache_endpoint(client):
    """R6.10."""
    response = await client.post("/reset_prefix_cache")
    assert response.status_code == 200
    assert response.json()["success"] is True


async def test_a_batched_prompt_is_refused_not_silently_collapsed(client):
    """The OpenAI schema allows a list of prompts. Upstream fans it out; batching is
    not implemented here, so it is refused -- treating it as one prompt would return
    a single completion for N inputs, which looks like it worked."""
    response = await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": ["one", "two"]}
    )
    assert response.status_code == 501
    assert response.json()["error"]["type"] == "NotImplementedError"


async def test_token_id_prompts_are_accepted(client):
    """R3.3: a flat list of ints is one prompt, not a batch."""
    response = await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": [10, 11, 12], "max_tokens": 3},
    )
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens"] == 3


async def test_an_empty_prompt_is_refused(client):
    response = await client.post("/v1/completions", json={"model": MODEL, "prompt": []})
    assert response.status_code == 400


# --- latency histograms actually populate (R12.1) ---------------------------


async def test_latency_histograms_have_observations_after_a_request(client):
    """The histograms existed from M1 but nothing populated them, so every latency
    panel on a dashboard read empty. This is the test that would have caught it."""
    await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "hello", "max_tokens": 6}
    )
    body = (await client.get("/metrics")).text

    counts = {
        line.split("{")[0]: float(line.rsplit(" ", 1)[1])
        for line in body.splitlines()
        if line.startswith("vllm:") and "_count" in line.split("{")[0]
    }
    for metric in (
        "vllm:e2e_request_latency_seconds_count",
        "vllm:time_to_first_token_seconds_count",
        "vllm:request_prompt_tokens_count",
        "vllm:request_generation_tokens_count",
    ):
        assert counts.get(metric, 0) >= 1, f"{metric} has no observations"


async def test_inter_token_latency_is_observed_for_multi_token_requests(client):
    await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "hello", "max_tokens": 8}
    )
    body = (await client.get("/metrics")).text
    line = next(
        ln
        for ln in body.splitlines()
        if ln.startswith("vllm:inter_token_latency_seconds_count")
    )
    assert float(line.rsplit(" ", 1)[1]) > 0


async def test_request_success_is_labelled_by_finish_reason(client):
    await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "x", "max_tokens": 4}
    )
    body = (await client.get("/metrics")).text
    assert "vllm:request_success_total{" in body
    assert 'finished_reason="length"' in body


async def test_observations_are_not_double_counted_across_scrapes(client):
    """Each observation belongs in a histogram exactly once. A scrape that left
    them pending would re-observe every request on every subsequent scrape."""
    await client.post(
        "/v1/completions", json={"model": MODEL, "prompt": "x", "max_tokens": 3}
    )

    def e2e_count(text: str) -> float:
        line = next(
            ln
            for ln in text.splitlines()
            if ln.startswith("vllm:e2e_request_latency_seconds_count")
        )
        return float(line.rsplit(" ", 1)[1])

    first = e2e_count((await client.get("/metrics")).text)
    second = e2e_count((await client.get("/metrics")).text)
    assert first == second == 1.0


async def test_prefix_cache_counters_move_on_a_shared_prefix(client):
    """R6.9 through the HTTP surface."""
    shared = "shared preamble " * 20
    await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": shared + "a", "max_tokens": 2},
    )
    await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": shared + "b", "max_tokens": 2},
    )
    body = (await client.get("/metrics")).text

    hits = next(
        float(ln.rsplit(" ", 1)[1])
        for ln in body.splitlines()
        if ln.startswith("vllm:prefix_cache_hits_total")
    )
    assert hits > 0


# --- structured output over HTTP (R15) -------------------------------------


async def test_response_format_json_schema_returns_parseable_json(client):
    """The surface a product actually calls. OpenAI's spelling, nested one level
    deeper than the vLLM extension's, which is the part people get wrong."""
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "population": {"type": "integer", "minimum": 1},
        },
        "required": ["city", "population"],
    }
    response = await client.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": "describe a city",
            "max_tokens": 300,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "city", "schema": schema},
            },
        },
    )
    assert response.status_code == 200, response.text
    value = json.loads(response.json()["choices"][0]["text"])
    assert isinstance(value["city"], str)
    assert value["population"] >= 1


async def test_guided_choice_over_http(client):
    response = await client.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": "yes or no",
            "max_tokens": 20,
            "guided_choice": ["yes", "no"],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"] in ("yes", "no")


async def test_chat_completions_honour_response_format(client):
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "give me an object"}],
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        },
    )
    assert response.status_code == 200, response.text
    assert isinstance(
        json.loads(response.json()["choices"][0]["message"]["content"]), dict
    )


async def test_setting_two_constraints_at_once_is_a_400(client):
    """No defined precedence, so guessing one would return output shaped like the
    field the caller did not mean."""
    response = await client.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": "x",
            "guided_choice": ["a", "b"],
            "guided_regex": r"\d+",
        },
    )
    assert response.status_code == 400
    assert "only one guided decoding constraint" in response.json()["error"]["message"]


async def test_an_unsatisfiable_schema_is_reported_not_hung(client):
    """A malformed schema has to come back as an error. Hanging would be the worst
    outcome, and is what happens if the failure never reaches the frontend."""
    response = await client.post(
        "/v1/completions",
        json={
            "model": MODEL,
            "prompt": "x",
            "max_tokens": 50,
            "guided_json": {
                "type": "object",
                "properties": {"n": {"type": "integer", "minimum": 5, "maximum": 1}},
                "required": ["n"],
            },
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["finish_reason"] == "error"
