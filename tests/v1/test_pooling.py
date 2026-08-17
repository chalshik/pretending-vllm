"""Pooling requests and `/v1/embeddings`. R2.2.

A pooling request prefills and stops. There is no decode phase, no sampling, and no
`max_tokens` -- which is the whole of the difference from the engine's point of view,
and the reason an embedding workload's step count looks nothing like a generation
workload's.

The vectors are synthetic and carry no meaning; see `pvllm/pooling_params.py`. What
these tests pin is everything around them, which is real: the schema, the batching,
the token accounting, the context-length error, and the prefix-cache sharing between
documents with a common preamble.
"""

from __future__ import annotations

import math

import httpx
import pytest
from prometheus_client import CollectorRegistry

from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.entrypoints.llm import LLM
from pvllm.entrypoints.openai.api_server import build_app
from pvllm.pooling_params import PoolingParams
from pvllm.sampling_params import SamplingParams

BASE = {
    "model": "tiny-test",
    "max_model_len": 512,
    "block_size": 16,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 8,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
    "seed": 3,
}

MODEL = "test-model"


@pytest.fixture
async def client():
    config = AsyncEngineArgs(
        model="tiny-test",
        served_model_name=MODEL,
        max_model_len=512,
        block_size=16,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        device_card="tiny-2gb",
        disable_log_stats=True,
    ).create_engine_config()
    app = build_app(config, registry=CollectorRegistry())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client


# --- the vector ------------------------------------------------------------


def test_a_vector_is_returned_and_it_is_normalized():
    llm = LLM(**BASE)
    try:
        output = llm.embed(["a document"])[0]
        vector = output.outputs.data
        # The model card's hidden size, which is what a real pooler emits.
        assert len(vector) == 128
        assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)
        assert output.finished
    finally:
        llm.shutdown()


def test_the_same_text_embeds_the_same_and_different_text_does_not():
    """The property a product's caching and dedup logic actually depend on. It is
    also the *only* property these vectors have: they carry no semantic information,
    so cosine similarity between two of them says nothing about the texts."""
    llm = LLM(**BASE)
    try:
        first, second, third = llm.embed(["alpha", "beta", "alpha"])
        assert first.outputs.data == third.outputs.data
        assert first.outputs.data != second.outputs.data
    finally:
        llm.shutdown()


def test_the_vector_does_not_depend_on_the_run():
    """Derived from the content rather than drawn from a stream, because a pooling
    model's output depends on its input and nothing else."""

    def run() -> list[float]:
        llm = LLM(**BASE)
        try:
            return llm.embed(["stable"])[0].outputs.data
        finally:
            llm.shutdown()

    assert run() == run()


def test_dimensions_narrows_the_vector():
    llm = LLM(**BASE)
    try:
        assert len(llm.embed(["x"], PoolingParams(dimensions=64))[0].outputs.data) == 64
        assert len(llm.embed(["x"], PoolingParams(dimensions=8))[0].outputs.data) == 8
    finally:
        llm.shutdown()


def test_a_classification_task_says_it_has_no_counterpart():
    with pytest.raises(NotImplementedError, match="classification head"):
        PoolingParams(task="classify")


# --- what the engine does with it ------------------------------------------


def test_a_pooling_request_generates_nothing_and_stops_after_its_prefill():
    llm = LLM(**BASE)
    try:
        engine = llm.llm_engine
        engine.add_request("e0", "a short document", pooling_params=PoolingParams())
        request = engine.engine_core.engine_core.scheduler.requests["e0"]
        assert request.use_pooling
        assert request.max_tokens == 0

        steps = 0
        while engine.has_unfinished_requests():
            engine.step()
            steps += 1
        # One step: prefill lands and the request is done. A generation request of
        # the same prompt would take one step per output token after it.
        assert steps == 1
        assert request.num_output_tokens == 0
    finally:
        llm.shutdown()


def test_a_long_document_chunks_and_pools_only_when_the_prompt_is_whole():
    """Chunked prefill applies unchanged. The vector exists on the last chunk's step
    and on none of the ones before it -- pooling over half a document would be a
    vector for a document nobody sent."""
    llm = LLM(**{**BASE, "max_num_batched_tokens": 64, "max_model_len": 1024})
    try:
        engine = llm.llm_engine
        engine.add_request(
            "e0",
            "a document long enough to need several chunks " * 12,
            pooling_params=PoolingParams(),
        )
        emitted = []
        steps = 0
        while engine.has_unfinished_requests():
            emitted.extend(engine.step())
            steps += 1
        assert steps > 1, "this test needs the prompt to actually chunk"
        assert len(emitted) == 1
        assert emitted[0].finished
    finally:
        llm.shutdown()


def test_documents_with_a_shared_preamble_share_its_blocks():
    """The capacity effect worth having: a page of documents sharing a system
    preamble prefills the preamble once."""
    llm = LLM(**BASE, enable_prefix_caching=True)
    preamble = "a long shared preamble that fills more than one block of KV cache "
    try:
        llm.embed([preamble + suffix for suffix in ("alpha", "beta", "gamma")])
        stats = llm.llm_engine.make_stats()
        assert stats["prefix_cache_hits"] > 0
    finally:
        llm.shutdown()


def test_a_pooling_request_cannot_also_be_a_sampling_one():
    from pvllm.v1.request import Request

    with pytest.raises(ValueError, match="exactly one of"):
        Request(
            request_id="r",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(max_tokens=4),
            arrival_time=0.0,
            pooling_params=PoolingParams(),
        )


def test_a_prompt_longer_than_the_context_window_is_refused():
    llm = LLM(**BASE)
    try:
        with pytest.raises(ValueError, match="maximum context length"):
            llm.embed([[7] * 600])
    finally:
        llm.shutdown()


# --- the endpoint ----------------------------------------------------------


async def test_the_embeddings_endpoint_returns_one_vector_per_input(client):
    response = await client.post(
        "/v1/embeddings", json={"model": MODEL, "input": ["one", "two", "three"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [item["index"] for item in body["data"]] == [0, 1, 2]
    assert all(item["object"] == "embedding" for item in body["data"])
    assert len(body["data"][0]["embedding"]) == 128
    # An embedding request generates nothing, so usage carries no completion tokens.
    assert body["usage"]["prompt_tokens"] == body["usage"]["total_tokens"]
    assert "completion_tokens" not in body["usage"]


async def test_a_bare_string_is_one_input_not_a_batch(client):
    response = await client.post(
        "/v1/embeddings", json={"model": MODEL, "input": "just one"}
    )
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


async def test_a_flat_token_id_list_is_one_input(client):
    """The ambiguity is in the OpenAI schema itself: a flat list of ints is one
    prompt of token ids, not a batch of single-token prompts. Backwards would turn
    one document into hundreds."""
    response = await client.post(
        "/v1/embeddings", json={"model": MODEL, "input": [10, 11, 12, 13]}
    )
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


async def test_a_batch_of_token_id_lists_is_a_batch(client):
    response = await client.post(
        "/v1/embeddings", json={"model": MODEL, "input": [[10, 11], [12, 13]]}
    )
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2


async def test_dimensions_reaches_the_endpoint(client):
    response = await client.post(
        "/v1/embeddings", json={"model": MODEL, "input": "x", "dimensions": 32}
    )
    assert len(response.json()["data"][0]["embedding"]) == 32


async def test_base64_encoding_says_it_is_not_supported(client):
    response = await client.post(
        "/v1/embeddings",
        json={"model": MODEL, "input": "x", "encoding_format": "base64"},
    )
    assert response.status_code == 501
    assert response.json()["error"]["type"] == "NotImplementedError"


async def test_an_unknown_model_is_404(client):
    response = await client.post("/v1/embeddings", json={"model": "nope", "input": "x"})
    assert response.status_code == 404


async def test_an_empty_input_is_400(client):
    response = await client.post("/v1/embeddings", json={"model": MODEL, "input": []})
    assert response.status_code == 400


async def test_a_document_too_long_for_the_context_window_is_400(client):
    response = await client.post(
        "/v1/embeddings", json={"model": MODEL, "input": [list(range(600))]}
    )
    assert response.status_code == 400
    assert "context length" in response.json()["error"]["message"]
