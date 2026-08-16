"""C5, C6, C7 -- the surfaces a product actually binds to. R21.3.

Different in kind from C1--C4. Those pin *behavior* and are recorded from runs; these
pin a *schema*, which is a static thing that either matches upstream or does not.

They matter more than they look. A product with a working dashboard, a working client
library, and working error handling depends on none of the scheduler's decisions and
all of these. F5 is the cautionary tale: the draft spec's metric names carried `_total`
suffixes that upstream declares without, `prometheus_client` appends on export, and
every counter would have exported as `vllm:prefix_cache_queries_total_total` --
silently emptying every panel built against real vLLM. That is a schema bug no
behavioral test would ever have caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from prometheus_client import CollectorRegistry

from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.entrypoints.openai.api_server import build_app

GOLDEN_DIR = Path(__file__).parent / "goldens"
MODEL = "conformance-model"


def make_app(registry: CollectorRegistry, **overrides):
    config = AsyncEngineArgs(
        model="tiny-test",
        served_model_name=MODEL,
        max_model_len=256,
        block_size=8,
        max_num_batched_tokens=64,
        max_num_seqs=2,
        device_card="tiny-2gb",
        num_gpu_blocks_override=32,
        disable_log_stats=True,
        **overrides,
    ).create_engine_config()
    return build_app(config, registry=registry)


@pytest.fixture
async def client():
    app = make_app(CollectorRegistry())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client


# --- C5: OpenAI request and response schema --------------------------------


async def test_completion_response_carries_every_required_field(client):
    response = await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "hello", "max_tokens": 4},
    )
    assert response.status_code == 200
    body = response.json()

    assert set(body) >= {"id", "object", "created", "model", "choices", "usage"}
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")

    choice = body["choices"][0]
    assert set(choice) >= {"index", "text", "finish_reason"}
    assert choice["finish_reason"] in ("stop", "length")

    usage = body["usage"]
    assert set(usage) >= {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


async def test_chat_completion_response_carries_every_required_field(client):
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert isinstance(message["content"], str)


async def test_streaming_chunks_match_the_streaming_schema(client):
    """A streaming client binds to a different object type and a `delta`, not a
    `message` -- getting that wrong breaks every SDK while non-streaming still
    works."""
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "stream": True,
        },
    )
    assert response.status_code == 200

    payloads = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert payloads
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in payloads)
    assert all("delta" in chunk["choices"][0] for chunk in payloads)
    assert payloads[-1]["choices"][0]["finish_reason"] is not None
    assert response.text.rstrip().endswith("data: [DONE]")


async def test_embedding_response_carries_every_required_field(client):
    """C5. R2.2. The vectors are synthetic; the envelope around them is the contract
    an embedding client is written against."""
    body = (
        await client.post(
            "/v1/embeddings", json={"model": MODEL, "input": ["one", "two"]}
        )
    ).json()
    assert set(body) >= {"id", "object", "created", "model", "data", "usage"}
    assert body["object"] == "list"
    for index, item in enumerate(body["data"]):
        assert set(item) == {"index", "object", "embedding"}
        assert item["index"] == index
        assert item["object"] == "embedding"
        assert item["embedding"] and all(
            isinstance(value, float) for value in item["embedding"]
        )
    # An embedding request generates nothing, so OpenAI's usage block for it has no
    # completion_tokens -- a client summing usage across endpoints depends on that.
    assert set(body["usage"]) == {"prompt_tokens", "total_tokens"}


async def test_models_endpoint_matches_the_list_schema(client):
    body = (await client.get("/v1/models")).json()
    assert body["object"] == "list"
    entry = body["data"][0]
    assert set(entry) >= {"id", "object", "created", "owned_by"}
    assert entry["object"] == "model"


# --- C6: Prometheus names, types, labels, bucket edges ---------------------


def scrape_families(registry: CollectorRegistry) -> list[dict]:
    """The metric surface as data, independent of any sample values.

    Values change every step and are not part of the contract; names, types, labels,
    and bucket edges are exactly what a dashboard binds to.
    """
    families = []
    for metric in registry.collect():
        entry: dict = {
            "name": metric.name,
            "type": metric.type,
            "labels": sorted(metric.samples[0].labels) if metric.samples else [],
        }
        if metric.type == "histogram":
            entry["buckets"] = sorted(
                {
                    sample.labels["le"]
                    for sample in metric.samples
                    if sample.name.endswith("_bucket")
                },
                key=lambda edge: float(edge),
            )
        families.append(entry)
    return sorted(families, key=lambda entry: entry["name"])


def test_metric_surface_matches_its_golden():
    """C6. Any name, type, label, or bucket edge that moves fails here.

    Recorded as a golden rather than asserted inline because the surface is 30-odd
    families: a hand-written list would drift out of date, and a test that only
    checks the handful someone remembered is worse than none.
    """
    registry = CollectorRegistry()
    make_app(registry)
    recorded = scrape_families(registry)

    path = GOLDEN_DIR / "metrics.json"
    if not path.exists():
        pytest.fail(
            "no metric-surface golden. Write one with:\n"
            "    python tools/capture_golden_trace.py --metrics"
        )
    golden = json.loads(path.read_text())

    recorded_names = {entry["name"] for entry in recorded}
    golden_names = {entry["name"] for entry in golden}
    assert not (golden_names - recorded_names), (
        f"metrics disappeared from the surface: {sorted(golden_names - recorded_names)}"
        f". A dashboard built against real vLLM would show empty panels."
    )
    assert not (recorded_names - golden_names), (
        f"undeclared metrics appeared: {sorted(recorded_names - golden_names)}. If "
        f"intended, re-record with tools/capture_golden_trace.py --metrics."
    )

    by_name = {entry["name"]: entry for entry in recorded}
    for entry in golden:
        assert by_name[entry["name"]] == entry, (
            f"{entry['name']} changed shape:\n"
            f"    recorded: {by_name[entry['name']]}\n"
            f"    golden:   {entry}"
        )


def test_counters_are_declared_without_a_total_suffix():
    """F5, asserted directly rather than only via the golden.

    `prometheus_client` appends `_total` to counters on export. Declaring
    `vllm:prefix_cache_queries_total` would export
    `vllm:prefix_cache_queries_total_total`, which matches nothing anyone's
    dashboard queries. This is the exact bug the draft spec would have shipped.
    """
    registry = CollectorRegistry()
    make_app(registry)
    offenders = [
        metric.name
        for metric in registry.collect()
        if metric.type == "counter" and metric.name.endswith("_total")
    ]
    assert not offenders, (
        f"counters declared with an explicit _total: {offenders}. Drop the suffix; "
        f"the client library adds it on export (F5)."
    )


async def test_modeled_latency_is_labeled_as_such(client):
    """R9.5, C6. Latency families carry the label in their HELP text, so a dashboard
    built on them cannot silently present modeled numbers as measured."""
    scrape = (await client.get("/metrics")).text
    help_lines = [line for line in scrape.splitlines() if line.startswith("# HELP")]
    latency = [line for line in help_lines if "seconds" in line.split()[2]]
    assert latency, "no latency families in the scrape"
    unlabeled = [line for line in latency if "MODELED" not in line.upper()]
    assert not unlabeled, "latency metrics without a modeled label:\n  " + "\n  ".join(
        unlabeled
    )


# --- C7: error codes and failure modes -------------------------------------


async def test_an_unknown_model_is_404(client):
    response = await client.post(
        "/v1/completions", json={"model": "not-served", "prompt": "x"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "NotFoundError"


async def test_an_overlong_prompt_is_400(client):
    """The capacity failure a product hits most often, and the one whose message
    people parse."""
    response = await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": "word " * 500, "max_tokens": 4},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "BadRequestError"
    assert "maximum context length" in error["message"]


async def test_a_malformed_request_is_422(client):
    """FastAPI's validation error, which is what upstream returns too -- a client
    distinguishing 400 from 422 is distinguishing 'your prompt is too long' from
    'your JSON is wrong'."""
    response = await client.post("/v1/completions", json={"model": MODEL})
    assert response.status_code == 422


async def test_error_bodies_are_shaped_like_openai_errors(client):
    """Every error, not just the convenient ones: SDKs read `error.message` and
    `error.type`, and one endpoint returning a bare string breaks them."""
    responses = [
        await client.post("/v1/completions", json={"model": "nope", "prompt": "x"}),
        await client.post(
            "/v1/chat/completions",
            json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
        ),
    ]
    for response in responses:
        body = response.json()
        assert set(body) == {"error"}, body
        assert set(body["error"]) >= {"message", "type"}


async def test_the_engine_survives_more_requests_than_it_can_hold(client):
    """C7 at capacity: queueing, not failing. A test double that 503s under load
    would teach a product the wrong lesson about its own backpressure."""
    import asyncio

    responses = await asyncio.gather(
        *(
            client.post(
                "/v1/completions",
                json={"model": MODEL, "prompt": f"prompt {i}", "max_tokens": 8},
            )
            for i in range(12)
        )
    )
    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["choices"][0]["text"] for response in responses)
