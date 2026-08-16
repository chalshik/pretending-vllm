"""Multimodal inputs. R18.

No pixels ever enter this engine. What enters is what scheduling and caching actually
depend on: how many prompt tokens an image occupies, where they sit, and a hash saying
whether two requests refer to the same image. All three are available without seeing
the image, and all three change observable behaviour:

* the encoder budget throttles how many images one step may encode, independently of
  the token budget;
* the encoder cache makes the second request with an image cheaper than the first;
* the image's hash partitions the prefix cache, so two prompts differing only in an
  image do not share KV -- while the text before the image still does.

What this cannot tell you is whether an image is malformed, too large, or of a type the
real model rejects. It tells you your plumbing, scheduling, and caching behave.
"""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from pvllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from pvllm.entrypoints.openai.multimodal import (
    DEFAULT_IMAGE_TOKENS,
    build_multimodal_prompt,
    parse_content,
)
from pvllm.multimodal.inputs import (
    PLACEHOLDER_TOKEN_ID,
    MultiModalFeatureSpec,
    content_hash,
)
from pvllm.sampling_params import SamplingParams
from pvllm.v1.core.encoder_cache_manager import EncoderCacheManager
from pvllm.v1.engine.llm_engine import LLMEngine

BASE = {
    "model": "tiny-test",
    "max_model_len": 2048,
    "block_size": 16,
    "max_num_batched_tokens": 512,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
}


def image(
    index: int,
    num_tokens: int = 64,
    position: int = 0,
    num_embeds: int | None = None,
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        identifier=content_hash(f"http://x/{index}.png"),
        modality="image",
        position=position,
        length=num_tokens,
        # Embeddings need not equal prompt tokens -- a real projector may emit
        # more or fewer -- and separating them is what lets a test move the
        # encoder budget without moving the token budget.
        num_embeds=num_tokens if num_embeds is None else num_embeds,
    )


class FakeRequest:
    def __init__(self, request_id: str, features: list[MultiModalFeatureSpec]) -> None:
        self.request_id = request_id
        self.mm_features = features


# --- content parsing -------------------------------------------------------


def test_a_string_content_stays_a_string():
    """The common case must not go anywhere near the placeholder machinery."""
    assert parse_content("just text") == ("just text", [])


def test_image_parts_become_placeholders_at_the_right_position():
    text, features = parse_content(
        [
            {"type": "text", "text": "before "},
            {"type": "image_url", "image_url": {"url": "http://x/cat.png"}},
            {"type": "text", "text": " after"},
        ]
    )
    assert text == "before  after"
    assert len(features) == 1
    offset, feature = features[0]
    # The character offset is where the image sits in the *text*; the token position
    # is resolved later, by the caller that owns the tokenizer.
    assert offset == len("before ")
    assert feature.length == DEFAULT_IMAGE_TOKENS


def test_the_same_url_hashes_the_same_and_a_different_one_does_not():
    assert content_hash("http://x/a.png") == content_hash("http://x/a.png")
    assert content_hash("http://x/a.png") != content_hash("http://x/b.png")
    # Modality is part of the identity: the encoder produces different embeddings
    # for the same bytes as an image and as a video frame.
    assert content_hash("data", "image") != content_hash("data", "video")


def test_unsupported_modalities_name_themselves():
    with pytest.raises(NotImplementedError, match="input_audio"):
        parse_content([{"type": "input_audio", "input_audio": {"data": "x"}}])


def test_a_malformed_image_part_is_refused():
    with pytest.raises(ValueError, match=r"image_url\.url"):
        parse_content([{"type": "image_url", "image_url": {}}])


def test_a_prompt_with_an_image_carries_placeholder_tokens():
    from pvllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer("tiny-test", tokenizer_mode="mock", vocab_size=1024)
    token_ids, features = build_multimodal_prompt(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look:"},
                    {"type": "image_url", "image_url": {"url": "http://x/cat.png"}},
                ],
            }
        ],
        tokenizer,
    )
    assert token_ids is not None
    assert len(features) == 1
    feature = features[0]
    run = token_ids[feature.position : feature.position + feature.length]
    assert run == [PLACEHOLDER_TOKEN_ID] * DEFAULT_IMAGE_TOKENS


def test_a_text_only_chat_takes_the_old_path():
    """`None` back means "not multimodal", so a text request is byte-for-byte what
    it was before any of this existed."""
    from pvllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer("tiny-test", tokenizer_mode="mock", vocab_size=1024)
    token_ids, features = build_multimodal_prompt(
        [{"role": "user", "content": "hello"}], tokenizer
    )
    assert token_ids is None
    assert features == []


# --- the encoder cache -----------------------------------------------------


def test_the_same_image_in_two_requests_is_encoded_once():
    manager = EncoderCacheManager(cache_size=256)
    first, second = FakeRequest("r0", [image(1)]), FakeRequest("r1", [image(1)])

    assert not manager.has_cache(first, 0)
    assert manager.can_allocate(first, 0)
    manager.allocate(first, 0)

    assert manager.has_cache(second, 0)
    manager.allocate(second, 0)
    # One entry, one allocation's worth of space, two holders.
    assert manager.num_free_slots == 256 - 64


def test_eviction_is_deferred_until_the_space_is_needed():
    """A request arriving with an image another just finished with must still hit.
    Eager eviction would lose exactly the reuse a chat workload depends on."""
    manager = EncoderCacheManager(cache_size=100)
    request = FakeRequest("r0", [image(1, num_tokens=60)])
    manager.can_allocate(request, 0)
    manager.allocate(request, 0)
    manager.free(request)

    assert manager.has_cache(request, 0), "evicted eagerly"
    assert manager.num_freeable_slots == 100


def test_an_unreferenced_entry_is_evicted_when_room_is_needed():
    manager = EncoderCacheManager(cache_size=100)
    old = FakeRequest("r0", [image(1, num_tokens=60)])
    manager.can_allocate(old, 0)
    manager.allocate(old, 0)
    manager.free(old)

    new = FakeRequest("r1", [image(2, num_tokens=80)])
    assert manager.can_allocate(new, 0)
    manager.allocate(new, 0)

    assert not manager.has_cache(old, 0)
    # The worker has to be told, or it holds embeddings nobody will ever free.
    assert manager.get_freed_mm_hashes() == [old.mm_features[0].identifier]


def test_a_referenced_entry_is_never_evicted():
    manager = EncoderCacheManager(cache_size=100)
    held = FakeRequest("r0", [image(1, num_tokens=60)])
    manager.can_allocate(held, 0)
    manager.allocate(held, 0)

    # 80 will not fit beside a live 60, and the live one must not be taken.
    other = FakeRequest("r1", [image(2, num_tokens=80)])
    assert not manager.can_allocate(other, 0)
    assert manager.has_cache(held, 0)


def test_an_image_larger_than_the_cache_is_refused_rather_than_looped_on():
    manager = EncoderCacheManager(cache_size=64)
    huge = FakeRequest("r0", [image(1, num_tokens=128)])
    assert not manager.can_allocate(huge, 0)


def test_a_zero_sized_cache_is_refused():
    with pytest.raises(ValueError, match="encoder_cache_size must be positive"):
        EncoderCacheManager(cache_size=0)


# --- the prefix cache (R6.3, C3) -------------------------------------------


def make_request(request_id: str, tokens: list[int], features: list) -> object:
    from pvllm.v1.request import Request

    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=4),
        arrival_time=0.0,
        mm_features=features,
    )


def test_a_block_before_an_image_is_not_partitioned_by_it():
    """The subtle half. Folding every image into every block's key would partition
    the text *before* the first image too, so two prompts sharing a long system
    prompt and differing only in a later image would share nothing -- and the
    reported hit rate would fall far below a real deployment's."""
    from pvllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys

    tokens = list(range(200))
    request = make_request("r0", tokens, [image(1, num_tokens=64, position=100)])

    # A block wholly before the image sees no multimodal key.
    assert generate_block_hash_extra_keys(request, 0, 16) is None
    # A block overlapping it does -- carrying the item's offset within the block,
    # as upstream's key does, so two tilings of the same images cannot collide.
    keys = generate_block_hash_extra_keys(request, 96, 112)
    assert keys == ((request.mm_features[0].identifier, 4),)


def test_two_prompts_differing_only_in_an_image_do_not_share_its_blocks():
    from pvllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys

    tokens = list(range(200))
    first = make_request("r0", tokens, [image(1, num_tokens=64, position=100)])
    second = make_request("r1", tokens, [image(2, num_tokens=64, position=100)])

    assert generate_block_hash_extra_keys(first, 96, 112) != (
        generate_block_hash_extra_keys(second, 96, 112)
    )


async def test_the_shared_text_prefix_survives_a_different_image():
    """End to end through the real cache, over HTTP."""
    import httpx

    from pvllm.entrypoints.openai.api_server import build_app

    config = AsyncEngineArgs(
        **{**BASE, "served_model_name": "m"}
    ).create_engine_config()
    app = build_app(config, registry=CollectorRegistry(), enable_debug_endpoints=True)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            prefix = "you are a careful assistant. " * 8

            async def ask(url: str) -> None:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "max_tokens": 4,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prefix},
                                    {"type": "image_url", "image_url": {"url": url}},
                                ],
                            }
                        ],
                    },
                )
                assert response.status_code == 200, response.text

            async def hits() -> int:
                payload = (await client.get("/debug/prefix_cache")).json()
                return int(payload["prefix_cache_hits"])

            await ask("http://x/cat.png")
            after_first = await hits()
            await ask("http://x/dog.png")
            after_different = await hits()
            await ask("http://x/cat.png")
            after_same = await hits()

            # A different image still shares the text before it...
            assert after_different > after_first
            # ...and the same image again shares strictly more than that.
            assert after_same - after_different > after_different - after_first
    finally:
        app.state.server.shutdown()


# --- scheduling ------------------------------------------------------------


def test_an_image_request_completes_end_to_end():
    engine = LLMEngine.from_engine_args(EngineArgs(**BASE))
    try:
        engine.add_request(
            "r0",
            [10] * 20 + [PLACEHOLDER_TOKEN_ID] * 64 + [11] * 10,
            SamplingParams(max_tokens=8),
            mm_features=[image(1, num_tokens=64, position=20)],
        )
        final = None
        for _ in range(500):
            if not engine.has_unfinished_requests():
                break
            for output in engine.step():
                if output.finished:
                    final = output
        assert final is not None
        assert len(final.outputs[0].token_ids) == 8
    finally:
        engine.shutdown()


def test_the_encoder_budget_throttles_a_batch_of_images():
    """A separate budget from the token budget, because encoder work and decoder
    work do not trade against each other. Without it a burst of image requests would
    all encode in one step, which no real engine does.

    Six images of 200 embeddings each, but only 20 prompt tokens apiece: every
    request fits the 512-token step budget with room to spare, so anything the
    scheduler holds back it holds back on the *encoder* budget. Two images fit in
    512 embeddings; the rest wait.
    """
    engine = LLMEngine.from_engine_args(
        EngineArgs(**{**BASE, "max_num_batched_tokens": 512})
    )
    try:
        for index in range(6):
            engine.add_request(
                f"r{index}",
                [10] * 10 + [PLACEHOLDER_TOKEN_ID] * 20 + [11] * 5,
                SamplingParams(max_tokens=4),
                mm_features=[image(index, num_tokens=20, position=10, num_embeds=200)],
            )
        scheduler = engine.engine_core.engine_core.scheduler
        output = scheduler.schedule()

        encoded = sum(len(ids) for ids in output.scheduled_encoder_inputs.values())
        assert encoded == 512 // 200, output.scheduled_encoder_inputs
        # The token budget was nowhere near binding, so the encoder budget is the
        # only thing that could have held the other four back.
        assert sum(output.num_scheduled_tokens.values()) < 512
    finally:
        engine.shutdown()


def test_every_image_request_still_drains():
    """Throttling must not deadlock: an image that could not be encoded now has to
    be encoded once the budget frees."""
    engine = LLMEngine.from_engine_args(EngineArgs(**BASE))
    try:
        for index in range(6):
            engine.add_request(
                f"r{index}",
                [10] * 10 + [PLACEHOLDER_TOKEN_ID] * 200 + [11] * 5,
                SamplingParams(max_tokens=4),
                mm_features=[image(index, num_tokens=200, position=10)],
            )
        finished = set()
        for _ in range(1000):
            if not engine.has_unfinished_requests():
                break
            for output in engine.step():
                if output.finished:
                    finished.add(output.request_id)
        assert finished == {f"r{i}" for i in range(6)}
    finally:
        engine.shutdown()


def test_a_text_request_never_touches_the_encoder_cache():
    engine = LLMEngine.from_engine_args(EngineArgs(**BASE))
    try:
        engine.add_request("r0", "hello there", SamplingParams(max_tokens=4))
        while engine.has_unfinished_requests():
            engine.step()
        cache = engine.engine_core.engine_core.scheduler.encoder_cache_manager
        assert cache.num_queries == 0
        assert cache.num_free_slots == cache.cache_size
    finally:
        engine.shutdown()
