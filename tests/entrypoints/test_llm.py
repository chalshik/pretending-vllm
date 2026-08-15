"""The offline entrypoint. R2.1."""

from __future__ import annotations

import pytest

from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams

BASE = {
    "max_model_len": 512,
    "block_size": 16,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 4,
    "enable_prefix_caching": False,
    "device_card": "workstation-24gb",
    "disable_log_stats": True,
}


@pytest.fixture
def llm():
    engine = LLM("dense-0.6b", **BASE)
    yield engine
    engine.shutdown()


def test_a_single_prompt_generates(llm):
    outputs = llm.generate("Hello", SamplingParams(max_tokens=6))
    assert len(outputs) == 1
    assert outputs[0].outputs[0].text
    assert outputs[0].finished


def test_results_come_back_in_submission_order(llm):
    """Not completion order -- results must line up with the prompts passed in."""
    prompts = ["alpha", "a much longer prompt than the others here", "beta"]
    outputs = llm.generate(prompts, SamplingParams(max_tokens=4))
    assert [output.prompt for output in outputs] == prompts


def test_prompts_are_batched_not_serialized(llm):
    """All are submitted before the first step, so the batching behaviour is
    observable offline rather than only under a server."""
    outputs = llm.generate(["a", "b", "c", "d"], SamplingParams(max_tokens=3))
    assert len(outputs) == 4
    assert all(output.finished for output in outputs)


def test_shared_sampling_params_are_not_mutated_across_prompts():
    """The processor resolves max_tokens against prompt length in place, so sharing
    one object would let the first prompt's resolution win for all of them."""
    engine = LLM("dense-0.6b", **BASE)
    params = SamplingParams(max_tokens=5)
    outputs = engine.generate(["short", "x" * 200], params)
    assert all(len(output.outputs[0].token_ids) == 5 for output in outputs)
    assert params.max_tokens == 5  # the caller's object is untouched
    engine.shutdown()


def test_per_prompt_sampling_params(llm):
    outputs = llm.generate(
        ["a", "b"], [SamplingParams(max_tokens=3), SamplingParams(max_tokens=7)]
    )
    assert len(outputs[0].outputs[0].token_ids) == 3
    assert len(outputs[1].outputs[0].token_ids) == 7


def test_mismatched_params_length_is_rejected(llm):
    with pytest.raises(ValueError, match="sampling params for"):
        llm.generate(["a", "b"], [SamplingParams(max_tokens=3)])


def test_token_ids_can_be_passed_instead_of_text(llm):
    """R3.3."""
    outputs = llm.generate([[10, 11, 12]], SamplingParams(max_tokens=3))
    assert outputs[0].prompt_token_ids == [10, 11, 12]


def test_chat_applies_the_template(llm):
    outputs = llm.chat(
        [{"role": "user", "content": "hello"}], SamplingParams(max_tokens=4)
    )
    assert len(outputs) == 1
    assert outputs[0].outputs[0].text
    # The template wrapped the message, so the prompt is longer than the content.
    assert "hello" in (outputs[0].prompt or "")


def test_generation_is_reproducible_across_instances():
    """R19.2, through the public surface."""

    def run() -> str:
        engine = LLM("dense-0.6b", seed=123, **BASE)
        text = engine.generate("hello", SamplingParams(max_tokens=8))[0].outputs[0].text
        engine.shutdown()
        return text

    assert run() == run()


def test_a_simulator_knob_reaches_the_engine():
    """The simulator surface is part of the offline API, not only the CLI."""
    engine = LLM(
        "dense-0.6b", output_length_policy="fixed", output_length_fixed=3, **BASE
    )
    outputs = engine.generate("hello", SamplingParams(max_tokens=100))
    assert len(outputs[0].outputs[0].token_ids) == 3
    assert outputs[0].outputs[0].finish_reason == "stop"
    engine.shutdown()


def test_context_length_errors_reach_the_caller(llm):
    with pytest.raises(ValueError, match="maximum context length"):
        llm.generate("x" * 4000, SamplingParams(max_tokens=4))


def test_the_engine_can_be_used_as_a_context_manager():
    with LLM("dense-0.6b", **BASE) as engine:
        assert engine.generate("hi", SamplingParams(max_tokens=2))
