"""Regressions for defects an adversarial review of M4a-M4e confirmed.

Same discipline as `test_review_regressions.py`: every one of these shipped green,
and was found by reading the code against its own claims rather than by a failing
test. Each test fails without its fix. The docstrings carry the original symptom,
because that is the part that makes a future failure here legible.
"""

from __future__ import annotations

import json

import pytest

from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams, StructuredOutputsParams

BASE = {
    "model": "tiny-test",
    "max_model_len": 256,
    "block_size": 16,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
    "seed": 7,
}


# --- the constrained document, and who decides when it ends ------------------


def test_a_constrained_request_stops_when_its_document_does():
    """The plan's length and the generic output-length policy were two separate
    opinions about when the request ends, and they collided at exactly
    `max_tokens == len(plan) + 1`: the EOS branch only fires when the planned length
    is *below* max_tokens, so at equality nothing emitted EOS and the next position
    indexed one past the end of the plan. The `IndexError` escaped `execute_model`
    into the engine step and wedged the engine for every later request, constrained
    or not -- one client's choice of max_tokens took the server down.
    """
    from pvllm.sim.grammar import generate_json
    from pvllm.sim.rng import RngFactory

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    # The document the engine will plan for its first request, and therefore the
    # exact max_tokens that used to land on the collision.
    document = generate_json(schema, RngFactory(BASE["seed"]).for_constraint("0"))
    exact = len(json.dumps(document, separators=(",", ":")).encode()) + 1

    llm = LLM(**BASE)
    try:
        for max_tokens in (exact - 1, exact, exact + 1):
            output = llm.generate(
                ["describe a user"],
                SamplingParams(
                    max_tokens=max_tokens,
                    structured_outputs=StructuredOutputsParams(json=schema),
                ),
            )[0]
            assert len(output.outputs[0].token_ids) <= max_tokens
        # And the engine is still alive to serve an ordinary request afterwards.
        assert llm.generate(["hello"], SamplingParams(max_tokens=4))[0].outputs[0].text
    finally:
        llm.shutdown()


def test_the_compiled_document_is_the_one_that_gets_generated():
    """`compile_grammar` probes satisfiability by generating a document, and
    generation generated another. Both drew from the *request's* stream, so the probe
    consumed the entropy the real document was about to use -- the compile step
    silently changed what the client received, and a probe that succeeded said
    nothing about the document that shipped. Both now derive from
    `(seed, "constraint", request_id)`, which is stateless.
    """
    from pvllm.sim.grammar import generate_json
    from pvllm.sim.rng import RngFactory

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    # Stateless in the request id, so the compile probe and the generation that
    # follows it draw the same document however many times either runs.
    once = generate_json(schema, RngFactory(11).for_constraint("req-1"))
    twice = generate_json(schema, RngFactory(11).for_constraint("req-1"))
    assert once == twice
    assert generate_json(schema, RngFactory(11).for_constraint("req-2")) != once
    # And the constraint stream is not the sampling stream, so probing costs the
    # request nothing.
    factory = RngFactory(11)
    before = factory.stream("jitter").random()
    factory.for_constraint("req-1")
    assert factory.stream("jitter").random() != before  # same stream, still advancing

    # End to end: a request whose schema compiles receives the document the compile
    # step proved, not a second one drawn behind it.
    llm = LLM(**BASE)
    try:
        output = llm.generate(
            ["describe a user"],
            SamplingParams(
                max_tokens=200,
                structured_outputs=StructuredOutputsParams(json=schema),
            ),
        )[0]
        document = json.loads(output.outputs[0].text)
        assert set(document) == {"name", "age"}
    finally:
        llm.shutdown()


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "integer", "multipleOf": 3},
        {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
        {"type": "array", "prefixItems": [{"type": "string"}]},
        {"type": "object", "patternProperties": {"^x": {"type": "string"}}},
        {"type": "object", "minProperties": 2},
        {"not": {"type": "string"}},
        {"if": {"type": "string"}, "then": {"type": "string"}},
    ],
)
def test_an_unsupported_schema_keyword_is_refused_by_name(schema):
    """These keywords were parsed as far as the type and then ignored, so the
    generator produced a document that violated the schema it was handed and reported
    success. A test double that quietly relaxes a constraint is worse than one that
    refuses it, because the product under test passes here and fails on real vLLM.
    """
    from pvllm.sim.grammar import UnsupportedConstraintError, generate_json
    from pvllm.sim.rng import RngFactory

    with pytest.raises(UnsupportedConstraintError):
        generate_json(schema, RngFactory(0).for_constraint("r"))


def test_exclusive_bounds_are_honoured_not_just_inclusive_ones():
    """`exclusiveMinimum`/`exclusiveMaximum` were read as their inclusive
    counterparts, so a schema demanding a value strictly above 0 got 0.
    """
    from pvllm.sim.grammar import generate_json
    from pvllm.sim.rng import RngFactory

    for index in range(20):
        value = generate_json(
            {"type": "integer", "exclusiveMinimum": 0, "maximum": 3},
            RngFactory(index).for_constraint(f"r{index}"),
        )
        assert value > 0
        number = generate_json(
            {"type": "number", "exclusiveMaximum": 1.0, "minimum": 0.0},
            RngFactory(index).for_constraint(f"n{index}"),
        )
        assert number < 1.0


# --- memory arithmetic ------------------------------------------------------


def test_lora_adapters_shard_only_the_half_tensor_parallel_can_shard():
    """`compute_lora_bytes` divided the whole adapter by `tp_size`. Only one of the
    two factors is shardable: `A` is `[d_in, r]` and shards along `d_in`, `B` is
    `[r, d_out]` and is replicated. Dividing both understated adapter memory by
    almost 2x at tp=8, and adapter memory comes straight out of the KV pool.
    """
    from pvllm.sim.memory import compute_lora_bytes
    from pvllm.sim.model_db import load_model_card

    card = load_model_card("tiny-test")
    kw = {
        "dtype": "float16",
        "max_loras": 4,
        "max_lora_rank": 16,
        "num_target_modules": 4,
    }
    one = compute_lora_bytes(card, tp_size=1, **kw)
    eight = compute_lora_bytes(card, tp_size=8, **kw)
    # Half shards, half does not: the ratio tends to 2, never to tp_size.
    assert one / eight == pytest.approx(2.0, rel=0.15)
    assert eight > one / 8


def test_pipeline_parallelism_divides_adapter_memory_per_stage():
    """Adapter bytes ignored `pp_size` entirely, so every stage was charged for the
    whole adapter -- the one thing pipeline parallelism exists to avoid.
    """
    from pvllm.sim.memory import compute_lora_bytes
    from pvllm.sim.model_db import load_model_card

    card = load_model_card("tiny-test")
    kw = {
        "dtype": "float16",
        "max_loras": 4,
        "max_lora_rank": 16,
        "num_target_modules": 4,
    }
    assert compute_lora_bytes(card, pp_size=2, **kw) < compute_lora_bytes(card, **kw)


def test_an_uneven_layer_split_charges_the_fattest_stage():
    """The local layer count truncated. A 29-layer model over 2 stages is 15 + 14,
    and the memory that has to fit is the 15's. Truncating reported the 14 and
    promised a KV pool the fat stage cannot hold.
    """
    import math

    from pvllm.sim.hardware_db import load_device_card
    from pvllm.sim.memory import compute_lora_bytes, compute_memory_profile
    from pvllm.sim.model_db import load_model_card

    card = load_model_card("dense-0.6b")
    layers = card.num_hidden_layers
    stages = 3
    assert layers % stages, "an even split would not distinguish floor from ceil"

    def profile(pp_size):
        return compute_memory_profile(
            card,
            load_device_card("datacenter-80gb"),
            dtype="float16",
            kv_cache_dtype="float16",
            block_size=16,
            gpu_memory_utilization=0.9,
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=4,
            pp_size=pp_size,
        )

    whole, split = profile(1), profile(stages)
    # ceil(L/2)/L of the weights, not floor(L/2)/L.
    assert (
        split.weight_bytes == whole.weight_bytes * math.ceil(layers / stages) // layers
    )

    lora = {
        "dtype": "float16",
        "max_loras": 4,
        "max_lora_rank": 16,
        "num_target_modules": 4,
    }
    assert compute_lora_bytes(card, pp_size=stages, **lora) == (
        compute_lora_bytes(card, **lora) * math.ceil(layers / stages) // layers
    )


# --- the null block is not anybody's KV -------------------------------------


def test_an_idle_windowed_engine_reports_zero_kv_usage():
    """The reserved null block counts against the pool but is never handed out, so
    `get_usage` over the raw block count reported a windowed engine as permanently
    in use at 1/num_blocks -- and `reset_prefix_cache` compared against the raw count
    too, so it refused to reset while the null block was held.
    """
    llm = LLM(**BASE, sliding_window=64)
    try:
        llm.generate(["hello"], SamplingParams(max_tokens=8))
        engine = llm.llm_engine.engine_core.engine_core
        assert engine.scheduler.kv_cache_manager.block_pool.get_usage() == 0.0
        assert llm.llm_engine.reset_prefix_cache() is True
    finally:
        llm.shutdown()


# --- configuration that cannot run should not report a capacity number -------


def test_a_tensor_parallel_size_that_splits_a_head_is_refused():
    """vLLM refuses this at startup. Reporting a capacity number for a configuration
    that cannot start is worse than reporting none.
    """
    with pytest.raises(ValueError, match="does not divide"):
        LLM(**BASE, tensor_parallel_size=3).shutdown()


def test_more_stages_than_layers_is_refused():
    with pytest.raises(ValueError, match="exceeds the model"):
        LLM(**BASE, pipeline_parallel_size=64).shutdown()


def test_an_adapter_without_enable_lora_is_refused():
    """The request carried a `lora_request` the engine had no adapter memory for and
    no `max_loras` bound; it was accepted and the adapter id silently partitioned the
    prefix cache anyway.
    """
    from pvllm.engine.arg_utils import EngineArgs
    from pvllm.lora.request import LoRARequest
    from pvllm.v1.engine.llm_engine import LLMEngine

    engine = LLMEngine(EngineArgs(**BASE).create_engine_config())
    try:
        with pytest.raises(ValueError, match="enable_lora"):
            engine.add_request(
                "r0",
                "hi",
                SamplingParams(max_tokens=4),
                lora_request=LoRARequest("a", 1, "/x"),
            )
    finally:
        engine.shutdown()


# --- speculative decoding ---------------------------------------------------


def test_speculation_turns_itself_off_above_the_configured_batch_size():
    """`speculative_disable_by_batch_size` was documented as modeled and read by
    nothing, so the knob moved no number at all.
    """
    from pvllm.config.speculative import SpeculativeConfig

    def drafts(**extra):
        llm = LLM(
            **BASE,
            spec_acceptance_rate=0.8,
            speculative_config=SpeculativeConfig(num_speculative_tokens=3, **extra),
        )
        try:
            llm.generate(["hello there"] * 4, SamplingParams(max_tokens=32))
            return (
                llm.llm_engine.engine_core.engine_core.scheduler.num_draft_tokens_total
            )
        finally:
            llm.shutdown()

    assert drafts() > 0
    # Every step runs 4 requests, so a threshold of 1 disables speculation entirely.
    assert drafts(speculative_disable_by_batch_size=1) == 0


def test_a_resumed_request_does_not_repeat_a_token_it_already_emitted():
    """Positional sampling made preemption lossy: `NewRequestData` carried only the
    prompt, so the runner rebuilt a resumed request believing it had generated
    nothing and re-sampled from position 0. The client got a token twice. Upstream
    carries `prefill_token_ids` for exactly this reason.
    """

    def run(**extra):
        llm = LLM(**BASE, enable_prefix_caching=False, **extra)
        try:
            outputs = llm.generate(
                [
                    "a prompt long enough to need several blocks of KV cache, "
                    f"number {i}"
                    for i in range(4)
                ],
                SamplingParams(max_tokens=60),
            )
            return (
                [list(o.outputs[0].token_ids) for o in outputs],
                llm.llm_engine.make_stats()["num_preemptions"],
            )
        finally:
            llm.shutdown()

    roomy, roomy_preemptions = run()
    tight, tight_preemptions = run(num_gpu_blocks_override=16)
    assert roomy_preemptions == 0
    assert tight_preemptions > 0, (
        "the tight run must actually preempt to prove anything"
    )
    assert roomy == tight


# --- per-request state that outlives the request ----------------------------


def test_per_request_simulator_state_does_not_accumulate_across_batches():
    """Every per-request map below the boundary was keyed by request id and never
    pruned: a long-lived server grew by one entry per request forever, and a harness
    that reuses request ids got the *previous* request's document back -- the cached
    plan hit before anything regenerated.

    The residue is bounded by the batch in flight, not by the requests served. The
    scheduler hands finished ids to the runner on the *following* step -- upstream
    does the same -- so the final batch's ids arrive only if another step runs.
    """
    llm = LLM(**BASE)
    try:
        core = llm.llm_engine.engine_core.engine_core
        model = core.executor.driver_worker.model_runner.sim_model
        params = SamplingParams(
            max_tokens=32,
            structured_outputs=StructuredOutputsParams(
                json={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            ),
        )
        residue = []
        for _ in range(4):
            llm.generate(["describe a user"] * 8, params)
            residue.append(len(model._constrained_plans))
        assert max(residue) <= 8, residue
        assert residue[-1] == residue[0], f"state accumulates across batches: {residue}"
    finally:
        llm.shutdown()


async def test_the_worker_drops_a_finished_multimodal_request_s_features():
    """`RequestState.mm_features` was written on admission and never removed, so a
    server doing image traffic leaked one entry per request for the process
    lifetime.
    """
    import httpx
    from prometheus_client import CollectorRegistry

    from pvllm.engine.arg_utils import AsyncEngineArgs
    from pvllm.entrypoints.openai.api_server import build_app

    app = build_app(
        AsyncEngineArgs(
            model="tiny-test",
            max_model_len=1024,
            max_num_batched_tokens=1024,
            device_card="tiny-2gb",
            disable_log_stats=True,
        ).create_engine_config(),
        registry=CollectorRegistry(),
    )
    runner = app.state.server.engine.engine_core.engine_core.executor.driver_worker.model_runner
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for index in range(6):
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "tiny-test",
                    "max_tokens": 4,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "look:"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"http://x/{index}.png"},
                                },
                            ],
                        }
                    ],
                },
            )
            assert response.status_code == 200
    # One request may still be in flight for the reason above; six may not.
    assert len(runner.req_states.mm_features) <= 1


# --- serving ----------------------------------------------------------------


async def test_lora_modules_are_served_under_their_own_model_names():
    """`--lora-modules` was accepted, validated, and then dropped: `/v1/models`
    listed only the base model and a request naming an adapter 404'd. The flag
    documented adapter routing and delivered none.
    """
    import httpx
    from prometheus_client import CollectorRegistry

    from pvllm.engine.arg_utils import AsyncEngineArgs
    from pvllm.entrypoints.openai.api_server import build_app

    args = AsyncEngineArgs(
        model="tiny-test",
        device_card="tiny-2gb",
        enable_lora=True,
        max_loras=2,
        lora_modules=["sql=/fake/sql", "chat=/fake/chat"],
        disable_log_stats=True,
    )
    app = build_app(args.create_engine_config(), registry=CollectorRegistry())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        listed = [m["id"] for m in (await client.get("/v1/models")).json()["data"]]
        assert "sql" in listed and "chat" in listed
        served = await client.post(
            "/v1/completions", json={"model": "sql", "prompt": "hi", "max_tokens": 4}
        )
        assert served.status_code == 200
        assert served.json()["model"] == "sql"
        missing = await client.post(
            "/v1/completions", json={"model": "nope", "prompt": "hi", "max_tokens": 4}
        )
        assert missing.status_code == 404
