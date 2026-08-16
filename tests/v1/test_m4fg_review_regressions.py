"""Regressions for defects an adversarial review of M4f and M4g confirmed.

M4f is multimodal, M4g is KV disaggregation. Both shipped green. The worst two here
hang the engine outright -- one on an ordinary `--max-num-batched-tokens` value, the
other on a request carrying two images -- and neither logged anything while doing it.

Same discipline as the other two regression files: each test fails without its fix.
"""

from __future__ import annotations

import pytest

from pvllm.engine.arg_utils import EngineArgs
from pvllm.multimodal.inputs import MultiModalFeatureSpec
from pvllm.sampling_params import SamplingParams
from pvllm.sim.kv_store import get_store, reset_stores
from pvllm.v1.engine.llm_engine import LLMEngine

BASE = {
    "model": "tiny-test",
    "max_model_len": 1024,
    "block_size": 16,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
}

PROMPT = "a shared prefix long enough to fill several blocks of KV cache " * 4


def image(name: str, position: int, length: int = 256) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        identifier=name,
        modality="image",
        position=position,
        length=length,
        num_embeds=length,
    )


def engine(**overrides) -> LLMEngine:
    return LLMEngine(EngineArgs(**{**BASE, **overrides}).create_engine_config())


def drain(llm: LLMEngine, limit: int = 200) -> int:
    for step in range(limit):
        if not llm.has_unfinished_requests():
            return step
        llm.step()
    raise AssertionError(f"engine did not drain in {limit} steps")


# --- the two hangs ----------------------------------------------------------


def test_an_ordinary_token_budget_does_not_hang_an_image_request():
    """`--max-num-batched-tokens 128` sized the encoder cache at 128, below a single
    256-embedding image. `can_allocate` then answered "no" on every step for the rest
    of time: the request was admitted, trimmed to zero tokens, and retried forever
    with no error and no log line. Upstream floors both encoder budgets at the
    largest single item; now so does `SchedulerConfig`.
    """
    llm = engine(max_num_batched_tokens=128)
    try:
        config = llm.vllm_config.scheduler_config
        assert config.encoder_cache_size >= 256
        assert config.max_num_encoder_input_tokens >= 256
        llm.add_request(
            "r0",
            [7] * 10 + [4] * 256 + [7] * 5,
            SamplingParams(max_tokens=4),
            mm_features=[image("img", 10)],
        )
        drain(llm)
    finally:
        llm.shutdown()


def test_two_images_in_one_request_do_not_deadlock_against_each_other():
    """Encoder references were only released when the request *finished*, so a
    request whose images together exceeded the cache could never schedule its second
    one: the first pinned the space for the rest of the request's life, and the
    request waited for room that its own first image was holding. Upstream releases
    an entry as soon as its placeholder run is behind `num_computed_tokens`.
    """
    from pvllm.v1.core.encoder_cache_manager import EncoderCacheManager

    llm = engine(max_num_batched_tokens=1024)
    try:
        scheduler = llm.engine_core.engine_core.scheduler
        # Room for one image, not two.
        scheduler.encoder_cache_manager = EncoderCacheManager(300)
        scheduler.max_num_encoder_input_tokens = 300
        llm.add_request(
            "r0",
            [7] * 10 + [4] * 256 + [7] * 4 + [4] * 256 + [7] * 5,
            SamplingParams(max_tokens=4),
            mm_features=[image("a", 10), image("b", 270)],
        )
        drain(llm)
    finally:
        llm.shutdown()


def test_an_encoder_reference_is_released_once_its_placeholders_are_computed():
    """The same defect, stated directly: a decoding request kept its image's
    embeddings resident until it stopped, so the encoder cache behaved like a
    per-request reservation rather than a cache.
    """
    llm = engine(max_num_batched_tokens=1024)
    try:
        scheduler = llm.engine_core.engine_core.scheduler
        llm.add_request(
            "r0",
            [7] * 10 + [4] * 256 + [7] * 5,
            SamplingParams(max_tokens=30),
            mm_features=[image("c", 10)],
        )
        held = []
        while llm.has_unfinished_requests():
            llm.step()
            held.append(len(scheduler.encoder_cache_manager.request_cached_ids))
        # Released during the run, not at the end of it.
        assert held[-3] == 0, held
        # And the entry stays *resident* so the next request with the same image
        # still hits -- released is not evicted.
        assert "c" in scheduler.encoder_cache_manager.cached
    finally:
        llm.shutdown()


def test_an_item_that_could_never_be_scheduled_raises_instead_of_retrying():
    """The backstop behind the budget floor. The scheduler's answer to "this cannot
    be scheduled" is to try again next step, so an unsatisfiable item is a livelock
    rather than backpressure -- it has to be an error.
    """
    from pvllm.v1.core.encoder_cache_manager import EncoderCacheManager

    llm = engine(max_num_batched_tokens=1024)
    try:
        scheduler = llm.engine_core.engine_core.scheduler
        scheduler.encoder_cache_manager = EncoderCacheManager(64)
        scheduler.max_num_encoder_input_tokens = 64
        llm.add_request(
            "r0",
            [7] * 10 + [4] * 256 + [7] * 5,
            SamplingParams(max_tokens=4),
            mm_features=[image("huge", 10)],
        )
        with pytest.raises(ValueError, match="could never be scheduled"):
            drain(llm)
    finally:
        llm.shutdown()


def test_the_worker_drops_evicted_encoder_outputs():
    """`free_encoder_mm_hashes` was carried in every `SchedulerOutput` and read by
    nobody, so the eviction protocol notified a listener that did not exist and the
    worker's resident set only ever grew.
    """
    llm = engine(max_num_batched_tokens=512, max_num_seqs=2)
    try:
        runner = llm.engine_core.engine_core.executor.driver_worker.model_runner
        manager = llm.engine_core.engine_core.scheduler.encoder_cache_manager
        for index in range(8):
            llm.add_request(
                f"r{index}",
                [7] * 5 + [4] * 200 + [7] * 3,
                SamplingParams(max_tokens=4),
                mm_features=[image(f"img{index}", 5, length=200)],
            )
            drain(llm)
        assert runner.encoder_outputs == set(manager.cached)
        assert len(runner.encoder_outputs) < 8
    finally:
        llm.shutdown()


# --- prefix-cache fidelity --------------------------------------------------


def test_a_multimodal_prompt_shares_its_text_prefix_with_a_text_one():
    """The renderer built token ids with `add_special_tokens=False` throughout, so a
    multimodal prompt carried no BOS while the text path's did. Every token shifted
    by one, block 0 differed, and a conversation that mixed text turns with image
    turns shared *nothing* -- understating the prefix-cache hit rate for the whole
    class of mixed workloads, which is the headline number this simulator exists to
    produce.
    """
    from pvllm.entrypoints.openai.multimodal import build_multimodal_prompt
    from pvllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer("tiny-test", tokenizer_mode="mock", vocab_size=1024)
    messages = [
        {"role": "system", "content": "a long shared system prompt " * 6},
        {"role": "user", "content": "hello"},
    ]
    text_ids = tokenizer.encode(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    )
    with_image = [
        *messages[:-1],
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "http://x/c.png"}},
            ],
        },
    ]
    mm_ids, features = build_multimodal_prompt(with_image, tokenizer)
    assert mm_ids is not None

    shared = 0
    for text_token, mm_token in zip(text_ids, mm_ids, strict=False):
        if text_token != mm_token:
            break
        shared += 1
    # Identical right up to where the image begins.
    assert shared == features[0].position


def test_add_generation_prompt_is_honoured_on_the_multimodal_path():
    """It was honoured on the text path and ignored on the multimodal one, so the
    same flag meant two different things depending on whether a message carried an
    image."""
    from pvllm.entrypoints.openai.multimodal import build_multimodal_prompt
    from pvllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer("tiny-test", tokenizer_mode="mock", vocab_size=1024)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "http://x/c.png"}},
            ],
        }
    ]
    with_prompt, _ = build_multimodal_prompt(messages, tokenizer)
    without, _ = build_multimodal_prompt(
        messages, tokenizer, add_generation_prompt=False
    )
    assert with_prompt is not None and without is not None
    assert len(without) < len(with_prompt)
    assert with_prompt[: len(without)] == without


def test_two_image_layouts_over_identical_tokens_do_not_collide():
    """Placeholder blocks are byte-identical whatever produced them, so a block's
    hash key has to carry each item's offset *within the block* -- which upstream's
    does and ours did not. Without it, two different tilings of the same pair of
    images hashed the same, and the second request read KV computed for the first
    one's layout.
    """
    from pvllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
    from pvllm.v1.request import Request

    def request_with(first_length: int) -> Request:
        return Request(
            request_id="r",
            prompt_token_ids=[4] * 32,
            sampling_params=SamplingParams(max_tokens=4),
            arrival_time=0.0,
            mm_features=[
                image("A", 0, length=first_length),
                image("B", first_length, length=32 - first_length),
            ],
        )

    # One block, two different splits of the same two images across it.
    assert generate_block_hash_extra_keys(
        request_with(1), 0, 8
    ) != generate_block_hash_extra_keys(request_with(2), 0, 8)


def test_the_extra_key_order_and_values_are_upstreams():
    """C3 makes hash *values* part of the contract, and the tuple's order is part of
    its value. Ours was salt-lora-mm against upstream's lora-mm-salt, the adapter was
    keyed by id where upstream keys by name, and the salt was repeated on every block
    where upstream carries it only on the first.
    """
    from pvllm.lora.request import LoRARequest
    from pvllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
    from pvllm.v1.request import Request

    request = Request(
        request_id="r",
        prompt_token_ids=list(range(64)),
        sampling_params=SamplingParams(max_tokens=4),
        arrival_time=0.0,
        lora_request=LoRARequest("adapter-a", 3, "/a"),
        cache_salt="tenant-x",
        mm_features=[image("img", 16, length=16)],
    )
    assert generate_block_hash_extra_keys(request, 0, 16) == (
        "adapter-a",
        "tenant-x",
    )
    # A later block carries the item's offset but not the salt again.
    assert generate_block_hash_extra_keys(request, 16, 32) == ("adapter-a", ("img", 0))


# --- the encoder's price ----------------------------------------------------


def test_an_image_costs_more_than_a_rounding_error():
    """`ENCODER_PARAMS_PER_EMBED = 12` meant "12 times hidden_size", about 49k
    parameters -- so a 256-patch image was modeled at a tenth of a microsecond
    against a 20 ms step. An image was documented as expensive and priced at
    nothing, which is the one direction this can be wrong in when the question being
    asked is whether caching encoder output is worth it.
    """
    from pvllm.sim.cost_model import StepProfile, build_cost_model
    from pvllm.sim.hardware_db import load_device_card
    from pvllm.sim.model_db import load_model_card

    cost_model = build_cost_model(
        "roofline",
        load_model_card("dense-8b"),
        load_device_card("datacenter-80gb"),
        dtype="float16",
        kv_cache_dtype="float16",
    )
    shape = {
        "num_tokens": 256,
        "num_reqs": 1,
        "query_lens": [256],
        "seq_lens": [256],
    }
    plain = cost_model.step_cost(StepProfile(**shape))
    with_image = cost_model.step_cost(StepProfile(**shape, num_encoder_embeds=256))
    # At least a few percent of the step it rides on, not a rounding error.
    assert with_image.duration > plain.duration * 1.02
    assert with_image.encoder_seconds > 0


def test_the_encoder_term_does_not_decide_bound_by():
    """`compute_seconds` was reported as `t_compute + t_encoder`, and `bound_by`
    compares compute against memory -- so an image made the breakdown describe a
    decode step's compute cost as something it was not, and on a step where the two
    terms are close it flipped the verdict outright. It is its own field now, summed
    into `duration` and kept out of the comparison.
    """
    from pvllm.sim.cost_model import StepProfile, build_cost_model
    from pvllm.sim.hardware_db import load_device_card
    from pvllm.sim.model_db import load_model_card

    cost_model = build_cost_model(
        "roofline",
        load_model_card("dense-8b"),
        load_device_card("datacenter-80gb"),
        dtype="float16",
        kv_cache_dtype="float16",
    )
    shape = {
        "num_tokens": 1,
        "num_reqs": 1,
        "query_lens": [1],
        "seq_lens": [512],
    }
    plain = cost_model.step_cost(StepProfile(**shape))
    with_image = cost_model.step_cost(StepProfile(**shape, num_encoder_embeds=256))

    # The decode's own terms are untouched by the image: `compute_seconds` is what
    # the decode computed, which is what `bound_by` is entitled to compare.
    assert with_image.encoder_seconds > 0
    assert with_image.compute_seconds == pytest.approx(plain.compute_seconds)
    assert with_image.memory_seconds == pytest.approx(plain.memory_seconds)
    assert plain.as_dict()["bound_by"] == "memory"
    assert with_image.as_dict()["bound_by"] == "memory"
    # And it is still paid for.
    assert with_image.duration > plain.duration


# --- the KV connector -------------------------------------------------------


def connector_engine(role=None, store="s", **extra) -> LLMEngine:
    return engine(
        max_num_batched_tokens=512,
        enable_prefix_caching=True,
        kv_transfer_config={
            "kv_connector": "SimSharedStoreConnector",
            "kv_role": role,
            "kv_connector_extra_config": {"store_name": store, **extra},
        },
    )


@pytest.fixture(autouse=True)
def _clean_stores():
    reset_stores()
    yield
    reset_stores()


def test_a_request_that_never_ran_publishes_nothing():
    """`request_finished` wrote every hash in `request.block_hashes`, which covers
    the whole prompt from the moment the request is built. An aborted request -- or
    one that never got a step -- filled the store with hashes for KV that does not
    exist, and a consumer then "hit" on them and read uninitialised blocks.
    """
    llm = connector_engine()
    try:
        llm.add_request("r0", PROMPT, SamplingParams(max_tokens=8))
        llm.abort_request("r0")
        assert len(get_store("s").resident) == 0
    finally:
        llm.shutdown()


def test_a_request_aborted_mid_prefill_publishes_only_what_it_computed():
    llm = connector_engine()
    try:
        llm.engine_core.engine_core.scheduler.max_num_scheduled_tokens = 64
        llm.add_request("r0", PROMPT, SamplingParams(max_tokens=8))
        llm.step()
        computed = llm.engine_core.engine_core.scheduler.requests[
            "r0"
        ].num_computed_tokens
        llm.abort_request("r0")
        assert len(get_store("s").resident) <= computed // BASE["block_size"]
    finally:
        llm.shutdown()


def test_kv_role_decides_who_publishes_and_who_pulls():
    """Validated at config time and read by nothing, so both halves of a
    disaggregated pair behaved as `kv_both` -- and the prefill node reported hits on
    KV it had just written itself."""
    consumer = connector_engine(role="kv_consumer")
    try:
        consumer.add_request("r0", PROMPT, SamplingParams(max_tokens=8))
        drain(consumer)
        assert len(get_store("s").resident) == 0
    finally:
        consumer.shutdown()

    producer = connector_engine(role="kv_producer", store="p")
    try:
        producer.add_request("r0", PROMPT, SamplingParams(max_tokens=8))
        drain(producer)
        assert len(get_store("p").resident) > 0
    finally:
        producer.shutdown()

    # A second producer against the same store publishes and does not pull.
    second = connector_engine(role="kv_producer", store="p")
    try:
        second.add_request("r1", PROMPT, SamplingParams(max_tokens=8))
        drain(second)
        assert get_store("p").num_hits == 0
    finally:
        second.shutdown()

    reader = connector_engine(role="kv_consumer", store="p")
    try:
        reader.add_request("r2", PROMPT, SamplingParams(max_tokens=8))
        drain(reader)
        assert get_store("p").num_hits > 0
    finally:
        reader.shutdown()


def test_two_models_on_one_store_do_not_share_kv():
    """A block hash is over token ids and extra keys; it says nothing about which
    model computed the KV. Two engines pointed at one store matched each other's
    blocks with different models entirely, and the consumer "hit" on KV of a
    different shape.
    """
    first = engine(
        model="dense-0.6b",
        device_card="datacenter-80gb",
        max_num_batched_tokens=512,
        enable_prefix_caching=True,
        kv_transfer_config={
            "kv_connector": "SimSharedStoreConnector",
            "kv_connector_extra_config": {"store_name": "x"},
        },
    )
    try:
        first.add_request("r0", PROMPT, SamplingParams(max_tokens=8))
        drain(first)
    finally:
        first.shutdown()

    second = engine(
        model="dense-8b",
        device_card="datacenter-80gb",
        max_num_batched_tokens=512,
        enable_prefix_caching=True,
        kv_transfer_config={
            "kv_connector": "SimSharedStoreConnector",
            "kv_connector_extra_config": {"store_name": "x"},
        },
    )
    try:
        second.add_request("r1", PROMPT, SamplingParams(max_tokens=8))
        drain(second)
        assert get_store("x").num_hits == 0
    finally:
        second.shutdown()


def test_the_producer_pays_for_its_own_writes():
    """`request_finished` computed the write's modeled duration and banked it in a
    counter nothing read; `wait_for_save` returned a hard zero. A prefill node
    publishing over a slow link finished in exactly the same modeled time as one
    publishing over a fast one -- and "does publishing cost less than recomputing" is
    the only question a disaggregation experiment asks.
    """
    fast = connector_engine(store="fast", bandwidth=1e12, latency=0.0)
    try:
        fast.add_request("r0", PROMPT, SamplingParams(max_tokens=8))
        drain(fast)
        fast_elapsed = fast.engine_core.engine_core.clock.elapsed
    finally:
        fast.shutdown()

    slow = connector_engine(store="slow", bandwidth=1e3, latency=10.0)
    try:
        slow.add_request("r0", PROMPT, SamplingParams(max_tokens=8))
        drain(slow)
        slow_elapsed = slow.engine_core.engine_core.clock.elapsed
    finally:
        slow.shutdown()

    assert slow_elapsed > fast_elapsed * 10


def test_a_store_described_two_ways_is_a_configuration_error():
    """`get_store` returned the resident store on a name hit and dropped the second
    caller's parameters, so a sweep over store bandwidth reused the first cell's
    store: the knob under test moved nothing while the numbers looked like an answer.
    """
    first = connector_engine(store="z", bandwidth=1e10)
    first.shutdown()
    with pytest.raises(ValueError, match="different parameters"):
        connector_engine(store="z", bandwidth=1e6)


def test_a_connector_without_block_hashes_refuses_by_name():
    """The store is keyed by block hash, and no block hashes exist without prefix
    caching -- so the connector became a total no-op with no warning. A
    disaggregation experiment ran, reported zero hits, and looked like a measurement.
    """
    with pytest.raises(ValueError, match="prefix caching"):
        engine(
            enable_prefix_caching=False,
            kv_transfer_config={"kv_connector": "SimSharedStoreConnector"},
        )
    with pytest.raises(NotImplementedError, match="sliding-window"):
        engine(
            sliding_window=64,
            enable_prefix_caching=True,
            kv_transfer_config={"kv_connector": "SimSharedStoreConnector"},
        )


def test_the_scheduler_names_no_simulator_module():
    """`_build_connector` imported `SimSharedStoreConnector` by name from Tier A
    code, which is a boundary crossing the purity check cannot see because the import
    is inside a function -- and it meant the configured connector name was validated
    and then ignored. It goes through the platform now, as upstream's
    `KVConnectorFactory` does.
    """
    import inspect

    from pvllm.v1.core.sched import scheduler as scheduler_module

    source = inspect.getsource(scheduler_module)
    assert "pvllm.sim" not in source
    assert "SimSharedStoreConnector" not in source


def test_the_two_outer_caches_reach_the_metrics_surface():
    """Both hit-rate counters existed since M4 and were reported by nothing, so the
    two features whose whole point is a hit rate had none on the surface a dashboard
    reads. Upstream's names, per C6.
    """
    from prometheus_client import CollectorRegistry, generate_latest

    from pvllm.v1.metrics.loggers import PrometheusStatLogger
    from pvllm.v1.metrics.stats import SchedulerStats

    llm = connector_engine()
    try:
        llm.add_request("r0", PROMPT, SamplingParams(max_tokens=8))
        drain(llm)
        stats = llm.make_stats()
        assert stats["external_prefix_cache_queries"] > 0
        assert "mm_cache_queries" in stats

        registry = CollectorRegistry()
        logger = PrometheusStatLogger(llm.vllm_config, registry=registry)
        logger.record(
            SchedulerStats(
                external_prefix_cache_queries=int(
                    stats["external_prefix_cache_queries"]
                ),
                external_prefix_cache_hits=int(stats["external_prefix_cache_hits"]),
                mm_cache_queries=int(stats["mm_cache_queries"]),
                mm_cache_hits=int(stats["mm_cache_hits"]),
            ),
            None,
        )
        exported = generate_latest(registry).decode()
        for name in (
            "vllm:external_prefix_cache_queries",
            "vllm:external_prefix_cache_hits",
            "vllm:mm_cache_queries",
            "vllm:mm_cache_hits",
        ):
            assert name in exported
    finally:
        llm.shutdown()
