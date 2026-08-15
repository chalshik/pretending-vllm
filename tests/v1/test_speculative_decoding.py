"""Speculative decoding. R14.

A draft model proposes `k` continuations, the target verifies them in one forward
pass, and every accepted draft is a token that cost no extra step. The whole point is
that it is **lossless**: the same sequence, produced in fewer steps. A simulator that
changed the output when speculation was enabled would be useless for the one question
anybody asks of it -- whether to turn it on.

What the simulator cannot supply is the acceptance itself: there is no draft model and
no target distribution to compare. `spec_acceptance_rate` stands in for their
agreement. Everything downstream of that number -- the scheduling, the token
accounting, the metrics -- is real.
"""

from __future__ import annotations

import pytest

from pvllm.config.speculative import SpeculativeConfig
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams

BASE = {
    "model": "tiny-test",
    "max_model_len": 512,
    "block_size": 16,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
    "seed": 4,
}


def run(prompts=("hello there",), max_tokens=40, **overrides):
    """Returns `(token_ids per request, steps, drafts, accepted)`."""
    engine = LLM(**{**BASE, **overrides})
    try:
        outputs = engine.generate(list(prompts), SamplingParams(max_tokens=max_tokens))
        scheduler = engine.llm_engine.engine_core.engine_core.scheduler
        return (
            [list(o.outputs[0].token_ids) for o in outputs],
            engine.llm_engine.make_stats()["step_index"],
            scheduler.num_draft_tokens_total,
            scheduler.num_accepted_tokens_total,
        )
    finally:
        engine.shutdown()


def speculating(k: int, rate: float) -> dict:
    return {
        "speculative_config": SpeculativeConfig(num_speculative_tokens=k),
        "spec_acceptance_rate": rate,
    }


# --- the property that makes it worth turning on ---------------------------


@pytest.mark.parametrize(
    ("k", "rate"), [(1, 0.9), (3, 0.9), (3, 0.5), (3, 0.1), (5, 0.95)]
)
def test_speculation_is_lossless(k, rate):
    """The same sequence, whatever the acceptance rate and however many drafts.

    This is the assertion that caught two real bugs while it was being written:
    sampling drew from a per-request *stream*, so drafting ahead consumed the
    entropy the real tokens were going to use; and the worker inferred its output
    position by counting appended tokens, which lags when a step schedules more
    than it accepts.
    """
    plain, _, _, _ = run()
    speculated, _, _, _ = run(**speculating(k, rate))
    assert speculated == plain


def test_speculation_takes_fewer_steps_when_acceptance_is_high():
    _, plain_steps, _, _ = run()
    _, spec_steps, _, _ = run(**speculating(3, 0.9))
    assert spec_steps < plain_steps / 2


def test_the_return_falls_off_as_acceptance_drops():
    """The curve a product tuning `num_speculative_tokens` needs to see. At a low
    acceptance rate verification is nearly all wasted work, which is why speculation
    is not free."""
    _, plain, _, _ = run()
    _, high, _, _ = run(**speculating(3, 0.9))
    _, mid, _, _ = run(**speculating(3, 0.5))
    _, low, _, _ = run(**speculating(3, 0.1))

    assert high < mid < low < plain
    # At 10% acceptance it barely beats no speculation at all.
    assert low > plain * 0.8


def test_more_drafts_help_when_acceptance_is_high():
    _, three, _, _ = run(**speculating(3, 0.95))
    _, five, _, _ = run(**speculating(5, 0.95))
    assert five < three


# --- the accounting --------------------------------------------------------


def test_observed_acceptance_follows_the_prefix_rule():
    """Drafts are accepted as a *prefix*: rejecting the second invalidates the third
    whatever the target thought of it. So the expected accepted fraction of `k`
    drafts at rate `p` is `sum(p**i for i in 1..k) / k`, not `p` -- which is exactly
    why the return on more drafts falls off so fast.
    """
    k, rate = 3, 0.5
    _, _, drafts, accepted = run(max_tokens=200, **speculating(k, rate))

    expected = sum(rate**i for i in range(1, k + 1)) / k
    assert drafts > 100, "not enough drafts for the rate to be meaningful"
    assert accepted / drafts == pytest.approx(expected, abs=0.08)


def test_no_drafts_are_counted_without_speculation():
    _, _, drafts, accepted = run()
    assert drafts == 0
    assert accepted == 0


def test_rejected_drafts_are_rolled_back():
    """A step schedules `1 + k` tokens and may get back one. The difference was
    computed against drafts the target rejected, and if it is not taken back off
    `num_computed_tokens` the next step schedules a negative number -- a loop that
    never terminates rather than an error that says anything. The scheduler asserts
    it; this drives the assertion at a rate that rejects nearly everything.
    """
    tokens, _, _, _ = run(max_tokens=60, **speculating(5, 0.05))
    assert len(tokens[0]) == 60


def test_concurrent_requests_all_speculate():
    prompts = [f"prompt number {i}" for i in range(4)]
    plain, _, _, _ = run(prompts=prompts, max_tokens=20)
    speculated, steps, drafts, _ = run(
        prompts=prompts, max_tokens=20, **speculating(3, 0.9)
    )

    assert speculated == plain
    assert drafts > 0
    assert steps < 20


def test_preemption_discards_stale_drafts():
    """Drafts were proposed against a KV cache the request no longer has. Verifying
    them after a recompute would be verifying against different state."""
    tokens, _, _, _ = run(
        prompts=[
            f"a prompt long enough to need several blocks, number {i}" for i in range(4)
        ],
        max_tokens=24,
        # Tight enough to force preemption, and still able to hold one request at
        # max_model_len -- R10.6 refuses a pool that cannot.
        max_model_len=256,
        num_gpu_blocks_override=20,
        enable_prefix_caching=False,
        **speculating(3, 0.9),
    )
    assert all(len(t) == 24 for t in tokens)


# --- the metrics -----------------------------------------------------------


async def test_the_spec_decode_counters_reach_metrics():
    import httpx
    from prometheus_client import CollectorRegistry

    from pvllm.engine.arg_utils import AsyncEngineArgs
    from pvllm.entrypoints.openai.api_server import build_app

    config = AsyncEngineArgs(
        **{**BASE, "served_model_name": "m"}, **speculating(3, 0.9)
    ).create_engine_config()
    app = build_app(config, registry=CollectorRegistry())
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/completions",
                json={"model": "m", "prompt": "hello", "max_tokens": 30},
            )
            scrape = (await client.get("/metrics")).text

        assert "vllm:spec_decode_num_draft_tokens_total" in scrape
        assert "vllm:spec_decode_num_accepted_tokens_total" in scrape
        # R9.5: the acceptance is the simulator's knob, and the help text says so.
        assert "MODELED acceptance" in scrape
    finally:
        app.state.server.shutdown()


# --- the config surface ----------------------------------------------------


def test_speculation_is_off_unless_configured():
    engine = LLM(**BASE)
    try:
        assert engine.llm_engine.vllm_config.speculative_config is None
        assert engine.llm_engine.engine_core.engine_core.scheduler.num_spec_tokens == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_speculative_tokens": 0}, "at least 1"),
        ({"num_speculative_tokens": 32}, "beyond what any real draft model"),
        ({"num_speculative_tokens": 2, "method": "telepathy"}, "unknown speculative"),
    ],
)
def test_nonsense_speculation_configs_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SpeculativeConfig(**kwargs)


def test_an_out_of_range_acceptance_rate_is_refused():
    from pvllm.config.device import SimConfig

    with pytest.raises(ValueError, match="spec_acceptance_rate must be in"):
        SimConfig(spec_acceptance_rate=1.5)
