"""Regressions for defects an adversarial review of M5 and M6 confirmed.

Three of these are regressions in *fixes* made earlier in the same session, which is
the pattern worth naming: the metrics gate over-corrected while fixing an
over-counting bug, and the expert-parallel lockstep landed on the offline path while
the commit message said the gap was closed. A fix is code, and code gets reviewed.
"""

from __future__ import annotations

import json

import httpx
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.entrypoints.llm import LLM
from pvllm.entrypoints.openai.api_server import build_app
from pvllm.sampling_params import SamplingParams
from pvllm.sim.cost_model import StepProfile, build_cost_model
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.model_db import load_model_card
from pvllm.v1.engine.async_llm import AsyncLLM

# --- metrics: n observations of the record, one observation of n -------------


async def test_an_n_way_request_is_counted_n_times_and_reports_n_once():
    """C6. Upstream calls `_update_stats_from_finished` unconditionally per child and
    gates only `observe_finished_request`, so an `n=3` request contributes three
    records and one observation of `n`. Gating the record too made a dashboard built
    against real vLLM read a third of the true completion rate, and sampled every
    latency histogram from whichever child happened to finish last."""
    registry = CollectorRegistry()
    app = build_app(
        AsyncEngineArgs(
            model="tiny-test",
            served_model_name="m",
            max_model_len=512,
            device_card="tiny-2gb",
        ).create_engine_config(),
        registry=registry,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "m", "prompt": "hi", "n": 3, "max_tokens": 4},
        )
        assert len(response.json()["choices"]) == 3
        await client.get("/metrics")

    exported = generate_latest(registry).decode()

    def value(needle: str) -> float:
        for line in exported.splitlines():
            if line.startswith(needle):
                return float(line.rsplit(" ", 1)[1])
        raise AssertionError(f"{needle} not exported")

    # Three finished requests...
    assert value('vllm:request_success_total{engine="0",finished_reason="length"') == 3
    assert value("vllm:e2e_request_latency_seconds_count") == 3
    assert value("vllm:request_generation_tokens_count") == 3
    # ...reporting n once, as 3.
    assert value("vllm:request_params_n_count") == 1
    assert value("vllm:request_params_n_sum") == 3


# --- expert parallelism: the server path locksteps too -----------------------


def _moe_config(**overrides):
    return AsyncEngineArgs(
        model="moe-8x7b",
        device_card="datacenter-80gb",
        max_model_len=2048,
        block_size=16,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        tensor_parallel_size=1,
        data_parallel_size=2,
        enable_expert_parallel=True,
        disable_log_stats=True,
        **overrides,
    ).create_engine_config()


async def test_the_async_path_locksteps_like_the_offline_one():
    """R13.4. `pvllm serve` never takes the synchronous path, so lockstep landing on
    only one of the two meant the HTTP surface -- the one an operator benchmarks --
    reported an idle expert-parallel replica as free. Every stats field still said
    `lockstep: True`, so nothing gave the discrepancy away."""
    engine = AsyncLLM(_moe_config())
    try:
        async for _ in engine.generate("a prompt", SamplingParams(max_tokens=16), "r0"):
            pass
        stats = await engine.make_stats()
        assert stats["lockstep"] is True
        assert stats["num_dummy_steps"] == 16
        assert stats["dummy_step_seconds"] > 0
        assert sorted(stats["per_engine_dummy_steps"]) == [0, 16]
    finally:
        engine.shutdown()


def test_the_ep_collective_is_charged_in_full_when_there_is_no_tp_term():
    """The tensor-parallel term bills the MLP's all-reduce only when `tp > 1`, so at
    `tp == 1` the EP term is the *whole* collective rather than the extra. Charging
    `dp - 1` unconditionally billed half of it at dp=2 and made every sweep comparing
    EP against TP come out cheap for EP."""
    model = load_model_card("moe-8x7b")
    device = load_device_card("datacenter-80gb")

    def comm(**kw) -> float:
        return (
            build_cost_model(
                "roofline",
                model,
                device,
                dtype="bfloat16",
                kv_cache_dtype="bfloat16",
                **kw,
            )
            .step_cost(
                StepProfile(
                    num_tokens=512,
                    num_reqs=4,
                    query_lens=[128] * 4,
                    seq_lens=[512] * 4,
                )
            )
            .comm_seconds
        )

    two_collectives = comm(tp_size=8)
    # Two token-sets of EP traffic equals the two all-reduces tensor parallelism bills.
    assert comm(tp_size=1, dp_size=2, ep_size=2) == pytest.approx(two_collectives)
    # A single device runs no collective at all.
    assert comm(tp_size=1) == 0.0
    # And EP at one replica stays byte-identical to plain TP.
    assert comm(tp_size=8, dp_size=1, ep_size=8) == pytest.approx(two_collectives)


# --- the real tokenizer ------------------------------------------------------


@pytest.fixture(scope="module")
def real_tokenizer_dir(tmp_path_factory):
    pytest.importorskip("tokenizers", reason="needs the `realtok` extra")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    path = tmp_path_factory.mktemp("eos-tokenizer")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog"] * 40,
        trainers.BpeTrainer(
            vocab_size=600,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>"}),
        encoding="utf-8",
    )
    return path


def test_a_request_stops_on_the_tokenizer_s_own_eos(real_tokenizer_dir):
    """R11.5. The simulator emitted `MockTokenizer`'s hardcoded id 1 while the stop
    check compared against the real tokenizer's EOS, so under `slow` nothing ever
    stopped on EOS: every request ran to its length cap instead."""

    def run(**overrides) -> tuple[int, str | None]:
        llm = LLM(
            model="dense-0.6b",
            device_card="workstation-24gb",
            max_model_len=2048,
            output_length_policy="fixed",
            output_length_fixed=12,
            disable_log_stats=True,
            seed=3,
            **overrides,
        )
        try:
            completion = llm.generate(
                ["the quick brown fox"], SamplingParams(max_tokens=1024)
            )[0].outputs[0]
            return len(completion.token_ids), completion.finish_reason
        finally:
            llm.shutdown()

    assert run() == (12, "stop")
    assert run(tokenizer=str(real_tokenizer_dir), tokenizer_mode="slow") == (12, "stop")


def test_a_multimodal_chat_refuses_a_template_that_is_not_the_model_s(
    real_tokenizer_dir,
):
    """The renderer hand-writes `<|role|>` markup, which matched `MockTokenizer`'s
    template exactly and matches no real model's. Rendering it anyway put an image
    turn and the text turn beside it on different templates, sharing zero leading
    tokens -- the regression the BOS fix existed to remove."""
    from pvllm.entrypoints.openai.multimodal import build_multimodal_prompt
    from pvllm.tokenizers import get_tokenizer

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "http://x/c.png"}},
            ],
        }
    ]
    real = get_tokenizer(str(real_tokenizer_dir), tokenizer_mode="slow", vocab_size=600)
    with pytest.raises(NotImplementedError, match="chat template"):
        build_multimodal_prompt(messages, real)

    # Unchanged under the default tokenizer, which is what it was written for.
    mock = get_tokenizer("tiny-test", tokenizer_mode="mock", vocab_size=1024)
    token_ids, features = build_multimodal_prompt(messages, mock)
    assert token_ids is not None and len(features) == 1


# --- the CLI clients ---------------------------------------------------------


def test_a_mid_stream_error_is_not_reported_as_a_finished_answer():
    """A stream that fails after its headers cannot use a status code, so the server
    sends the error as an SSE payload. Dropping it printed a truncated answer and
    exited 0 -- a script could not tell a failed generation from a finished one."""
    from pvllm.entrypoints.cli.openai import StreamFailure, stream_events

    body = (
        'data: {"choices":[{"text":"Hel"}]}\n\n'
        'data: {"choices":[{"text":"lo"}]}\n\n'
        'data: {"error":{"message":"engine died mid-stream","code":500}}\n\n'
        "data: [DONE]\n\n"
    )

    class FakeResponse:
        def __iter__(self):
            return iter(body.encode().splitlines(keepends=True))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import pvllm.entrypoints.cli.openai as module

    original = module._request
    module._request = lambda *a, **k: FakeResponse()
    try:
        with pytest.raises(StreamFailure, match="engine died mid-stream"):
            list(stream_events("http://x/v1/completions", {}, None))
    finally:
        module._request = original


def test_every_prompt_gets_the_same_unreachable_handling(capsys):
    """Only the *first* request was guarded, so a server restarted between two piped
    prompts came out as a raw urllib traceback -- the thing the module says it turns
    into a message."""
    import socket

    from pvllm.entrypoints.cli.main import main

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead = sock.getsockname()[1]

    assert main(["complete", "--url", f"http://127.0.0.1:{dead}/v1", "-q", "x"]) == 1
    assert "cannot reach" in capsys.readouterr().err


# --- the parent map ----------------------------------------------------------


async def test_a_cancelled_fan_out_leaves_no_parent_behind():
    """R11.7. `generate`'s finally aborts by *child* id -- that is what it has to hand
    -- so popping only the id it was given left the parent in the map forever. Nothing
    else signalled it: `request_states` and the core both drained correctly."""
    engine = AsyncLLM(
        AsyncEngineArgs(
            model="tiny-test",
            max_model_len=512,
            device_card="tiny-2gb",
            disable_log_stats=True,
        ).create_engine_config()
    )
    try:
        for index in range(5):
            stream = engine.generate(
                "a prompt", SamplingParams(n=3, max_tokens=64), f"r{index}"
            )
            await anext(stream)
            await stream.aclose()
        assert engine.output_processor.parent_requests == {}
        assert engine.output_processor.request_states == {}
    finally:
        engine.shutdown()


# --- the six the review's top-2 cap left unverified ---------------------------


def test_the_purity_guards_catch_both_import_forms():
    """`from a.b import c` records only `a.b`, so the ordinary module-import form --
    `from pvllm.entrypoints.cli import openai` -- slipped past both exemption guards
    while the dotted form was caught. Each enforced its invariant against one of the
    two ways to write the same import."""
    import ast

    from tests.unit.test_purity import _full_dotted_imports

    tree = ast.parse("from pvllm.entrypoints.cli import openai\n")
    assert "pvllm.entrypoints.cli.openai" in _full_dotted_imports(tree)
    tree = ast.parse("from pvllm.entrypoints.cli.openai import stream_events\n")
    assert "pvllm.entrypoints.cli.openai" in _full_dotted_imports(tree)


def test_a_wall_clock_read_through_datetime_is_caught():
    """`datetime.datetime.now()` is a wall clock and `now` was not in the forbidden
    set, so a `strftime_now` added to the chat-template environment walked straight
    past the lint that exists to catch exactly that."""
    from tests.unit.test_purity import FORBIDDEN_TIME_ATTRS

    assert "now" in FORBIDDEN_TIME_ATTRS
    assert "utcnow" in FORBIDDEN_TIME_ATTRS


def test_lockstep_rounds_is_reported():
    """The fourth inert mechanism: incremented twice, read nowhere, under a comment
    saying it was counted for the stats -- shipped in the commit that named the
    pattern and deleted the third."""
    llm = LLM(
        model="moe-8x7b",
        device_card="datacenter-80gb",
        max_model_len=2048,
        block_size=16,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        tensor_parallel_size=1,
        data_parallel_size=2,
        enable_expert_parallel=True,
        disable_log_stats=True,
    )
    try:
        llm.generate(["a prompt"], SamplingParams(max_tokens=8))
        stats = llm.llm_engine.make_stats()
        assert stats["lockstep_rounds"] == 8
        # The ratio is the imbalance: one replica-step wasted per round here.
        assert stats["num_dummy_steps"] == stats["lockstep_rounds"]
    finally:
        llm.shutdown()


def test_a_dummy_step_lands_on_the_scheduler_s_step_axis(tmp_path):
    """Dummy records carried `EngineCore.step_index`, a different counter that never
    moves for a dummy batch -- so an idle replica wrote every record at step 0 and the
    viewer, which uses that field as its column index, rendered none of them."""
    import json

    from pvllm.trace_viewer import render_text, summarize

    llm = LLM(
        model="moe-8x7b",
        device_card="datacenter-80gb",
        max_model_len=2048,
        block_size=16,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        tensor_parallel_size=1,
        data_parallel_size=2,
        enable_expert_parallel=True,
        disable_log_stats=True,
        trace_path=str(tmp_path / "t.jsonl"),
    )
    try:
        llm.generate(["a prompt"], SamplingParams(max_tokens=8))
    finally:
        llm.shutdown()

    idle = tmp_path / "t.dp1.jsonl"
    steps = [
        json.loads(line)
        for line in idle.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("dummy")
    ]
    assert [record["step"] for record in steps] == list(range(1, 9))
    # And the viewer says what the replica spent, which a request timeline cannot.
    assert "8 dummy steps" in render_text(summarize(str(idle)))


def test_the_debug_surface_says_which_replica_it_describes():
    """A comment saying "replica 0, said rather than implied" said it only to whoever
    read the source. An operator chasing a request routed elsewhere got "not found"
    for a request the deployment was actively running."""
    import asyncio

    from prometheus_client import CollectorRegistry

    async def probe() -> dict:
        app = build_app(
            AsyncEngineArgs(
                model="tiny-test",
                served_model_name="m",
                max_model_len=512,
                device_card="tiny-2gb",
                data_parallel_size=2,
                disable_log_stats=True,
            ).create_engine_config(),
            registry=CollectorRegistry(),
            enable_debug_endpoints=True,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return (await client.get("/debug/scheduler")).json()

    body = asyncio.run(probe())
    assert body["data_parallel_rank"] == 0
    assert body["data_parallel_size"] == 2
    assert "replica 0 of 2" in body["scope"]


def test_a_seed_reproduces_the_completion_under_every_length_policy():
    """The seed threaded into the per-position draw but not the length draw, which
    stayed keyed on request id -- so two same-seeded requests agreed on their token
    stream and disagreed on where to stop. The test written at the time ran only
    under the default policy, which never takes that draw."""
    for policy in ("from_request", "uniform", "lognormal"):
        llm = LLM(
            model="tiny-test",
            device_card="tiny-2gb",
            max_model_len=512,
            output_length_policy=policy,
            disable_log_stats=True,
        )
        try:
            outputs = llm.generate(["x", "x"], SamplingParams(max_tokens=300, seed=1))
            first, second = outputs[0].outputs[0], outputs[1].outputs[0]
            assert first.text == second.text, policy
            assert len(first.token_ids) == len(second.token_ids), policy
        finally:
            llm.shutdown()


def test_a_real_chat_template_gets_the_environment_transformers_builds(tmp_path):
    """A bare jinja2 Environment is missing three things and each broke real models:
    `strftime_now` (every Llama-3.x template calls it -- an HTTP 500 on a request that
    can never succeed), `loopcontrols` (`{% break %}` dies at load), and a `tojson`
    that does not reorder keys or escape into `\\uXXXX`."""
    pytest.importorskip("jinja2")
    from pvllm.tokenizers.hf import HFTokenizer

    class Stub:
        def token_to_id(self, token):
            return 1

        def get_vocab_size(self):
            return 100

        def get_added_tokens_decoder(self):
            return {}

    def render(template: str) -> str:
        tokenizer = HFTokenizer(
            Stub(), name="probe", config={"chat_template": template}
        )
        return tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])

    assert render("{{ strftime_now('%Y') }}").isdigit()
    assert render("{% for m in messages %}{% break %}{% endfor %}ok") == "ok"
    # Key order preserved, `<` not escaped -- jinja2's builtin would do both.
    assert render("{{ {'b': 1, 'a': 'x<y'} | tojson }}") == '{"b": 1, "a": "x<y"}'
