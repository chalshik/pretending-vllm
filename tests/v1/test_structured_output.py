"""Structured output, end to end. R15.

The contract a product depends on is narrow and absolute: **if you ask for JSON
matching a schema, `json.loads()` on the result must succeed and the value must match
the schema.** Almost every test here is a variation on that, because a structured
output feature that produces almost-valid JSON is worse than none -- it fails in the
consumer, at parse time, with an error that points at the consumer.

The other half is the scheduler-side interaction R15.1 requires to be real: a request
whose grammar is still compiling is not admitted, a request whose grammar fails to
compile fails alone, and neither disturbs the requests around it.
"""

from __future__ import annotations

import json
import re

import pytest

from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams, StructuredOutputsParams

BASE = {
    "model": "tiny-test",
    "max_model_len": 1024,
    "block_size": 16,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
    },
    "required": ["sentiment", "confidence"],
}


@pytest.fixture
def engine():
    llm = LLM(**BASE)
    try:
        yield llm
    finally:
        llm.shutdown()


def generate(engine: LLM, params: StructuredOutputsParams, max_tokens: int = 400):
    outputs = engine.generate(
        ["classify this"],
        SamplingParams(max_tokens=max_tokens, structured_outputs=params),
    )
    return outputs[0].outputs[0]


# --- the contract ----------------------------------------------------------


def test_a_json_schema_request_produces_parseable_json(engine):
    completion = generate(engine, StructuredOutputsParams(json=SCHEMA))
    value = json.loads(completion.text)

    assert value["sentiment"] in ("positive", "negative", "neutral")
    assert 0 <= value["confidence"] <= 1
    assert 2 <= len(value["keywords"]) <= 4
    assert all(isinstance(word, str) for word in value["keywords"])
    # Terminated through the ordinary stop path, so a client sees `stop` rather than
    # a truncation it would need to retry.
    assert completion.finish_reason == "stop"


def test_a_json_object_request_produces_an_object(engine):
    value = json.loads(generate(engine, StructuredOutputsParams(json_object=True)).text)
    assert isinstance(value, dict)


def test_a_choice_request_returns_one_of_the_choices(engine):
    completion = generate(
        engine, StructuredOutputsParams(choice=["yes", "no", "maybe"])
    )
    assert completion.text in ("yes", "no", "maybe")


def test_a_regex_request_matches_the_pattern(engine):
    pattern = r"\d{3}-\d{3}-\d{4}"
    completion = generate(engine, StructuredOutputsParams(regex=pattern))
    assert re.fullmatch(pattern, completion.text), completion.text


@pytest.mark.parametrize(
    "pattern",
    [
        r"[A-Z]{2}\d{4}",
        r"(cat|dog|bird)",
        r"v\d+\.\d+\.\d+",
        r"[a-z]+@[a-z]+\.(com|org)",
        r"#[0-9a-f]{6}",
        r"(yes|no)!?",
    ],
)
def test_regex_constraints_hold_across_common_patterns(engine, pattern):
    completion = generate(engine, StructuredOutputsParams(regex=pattern))
    assert re.fullmatch(pattern, completion.text), (
        f"{completion.text!r} does not match {pattern!r}"
    )


def test_two_requests_against_one_schema_get_different_values(engine):
    """Constrained does not mean constant. A product testing its parser against a
    single frozen document would not be testing much."""
    outputs = engine.generate(
        [f"classify item {i}" for i in range(6)],
        SamplingParams(
            max_tokens=400, structured_outputs=StructuredOutputsParams(json=SCHEMA)
        ),
    )
    documents = [output.outputs[0].text for output in outputs]
    assert all(json.loads(document) for document in documents)
    assert len(set(documents)) > 1


def test_constrained_output_is_reproducible():
    """B4 applies to structured output too, or a golden test over an API that
    returns JSON could never be written."""

    def run() -> str:
        llm = LLM(**BASE, seed=11)
        text = (
            llm.generate(
                ["classify this"],
                SamplingParams(
                    max_tokens=400,
                    structured_outputs=StructuredOutputsParams(json=SCHEMA),
                ),
            )[0]
            .outputs[0]
            .text
        )
        llm.shutdown()
        return text

    assert run() == run()


def test_a_schema_that_does_not_fit_truncates_and_says_so(engine):
    """What real vLLM does, and what a client needs to be able to observe: the
    grammar did not finish, the output is a prefix, and `finish_reason` is `length`
    rather than `stop`. Quietly completing the document would hide a
    misconfiguration until production."""
    completion = generate(engine, StructuredOutputsParams(json=SCHEMA), max_tokens=10)

    assert completion.finish_reason == "length"
    assert len(completion.token_ids) == 10
    with pytest.raises(json.JSONDecodeError):
        json.loads(completion.text)


# --- nested and referential schemas ----------------------------------------


def test_nested_objects_and_refs_are_satisfied(engine):
    schema = {
        "type": "object",
        "$defs": {
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string", "pattern": r"\d{5}"},
                },
                "required": ["city", "zip"],
            }
        },
        "properties": {
            "name": {"type": "string"},
            "home": {"$ref": "#/$defs/address"},
            "offices": {"type": "array", "items": {"$ref": "#/$defs/address"}},
        },
        "required": ["name", "home"],
    }
    value = json.loads(generate(engine, StructuredOutputsParams(json=schema)).text)

    assert isinstance(value["name"], str)
    assert re.fullmatch(r"\d{5}", value["home"]["zip"])
    assert all(re.fullmatch(r"\d{5}", office["zip"]) for office in value["offices"])


def test_const_and_enum_are_honoured(engine):
    schema = {
        "type": "object",
        "properties": {
            "version": {"const": "v2"},
            "status": {"enum": ["ok", "error"]},
        },
        "required": ["version", "status"],
    }
    value = json.loads(generate(engine, StructuredOutputsParams(json=schema)).text)
    assert value["version"] == "v2"
    assert value["status"] in ("ok", "error")


# --- the scheduler-side interaction (R15.1) --------------------------------


def test_a_constrained_request_waits_for_its_grammar():
    """R15.1's core requirement. The request is not admissible until compilation
    finishes, and it reaches the scheduler in that state rather than being briefly
    admissible before anyone looked."""
    from pvllm.sampling_params import SamplingParams as Params
    from pvllm.v1.request import Request, RequestStatus

    request = Request(
        request_id="r0",
        prompt_token_ids=[1, 2, 3],
        sampling_params=Params(
            max_tokens=8, structured_outputs=StructuredOutputsParams(json=SCHEMA)
        ),
        arrival_time=0.0,
    )
    assert request.use_structured_output
    assert request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR


def test_an_unconstrained_request_skips_the_whole_path():
    from pvllm.sampling_params import SamplingParams as Params
    from pvllm.v1.request import Request, RequestStatus

    request = Request(
        request_id="r0",
        prompt_token_ids=[1, 2, 3],
        sampling_params=Params(max_tokens=8),
        arrival_time=0.0,
    )
    assert not request.use_structured_output
    assert request.structured_output_request is None
    assert request.status == RequestStatus.WAITING


def test_a_bad_schema_fails_its_own_request_and_nothing_else(engine):
    """The property that matters when a client sends something malformed: one
    request errors, every other request in flight completes."""
    outputs = engine.generate(
        ["good one", "bad one", "another good one"],
        [
            SamplingParams(
                max_tokens=400, structured_outputs=StructuredOutputsParams(json=SCHEMA)
            ),
            SamplingParams(
                max_tokens=400,
                # minimum above maximum: no integer satisfies it.
                structured_outputs=StructuredOutputsParams(
                    json={
                        "type": "object",
                        "properties": {
                            "n": {"type": "integer", "minimum": 10, "maximum": 1}
                        },
                        "required": ["n"],
                    }
                ),
            ),
            SamplingParams(
                max_tokens=400, structured_outputs=StructuredOutputsParams(json=SCHEMA)
            ),
        ],
    )
    assert json.loads(outputs[0].outputs[0].text)
    assert json.loads(outputs[2].outputs[0].text)
    assert outputs[1].outputs[0].finish_reason == "error"


def test_the_engine_keeps_serving_after_a_grammar_failure(engine):
    """A bad schema must not wedge the engine: the next request still works."""
    engine.generate(
        ["bad"],
        SamplingParams(
            max_tokens=64,
            structured_outputs=StructuredOutputsParams(regex=r"(?=lookahead)x"),
        ),
    )
    assert json.loads(generate(engine, StructuredOutputsParams(json=SCHEMA)).text)


# --- unsupported paths name themselves -------------------------------------


def test_an_ebnf_grammar_says_it_is_not_supported(engine):
    completion = generate(engine, StructuredOutputsParams(grammar='root ::= "a" | "b"'))
    assert completion.finish_reason == "error"


def test_an_upstream_backend_is_refused_by_name():
    from pvllm.v1.structured_output import StructuredOutputManager

    llm = LLM(**BASE)
    try:
        manager = StructuredOutputManager(llm.llm_engine.vllm_config)
        manager.set_tokenizer(llm.llm_engine.tokenizer)
        params = StructuredOutputsParams(json=SCHEMA)
        params._backend = "xgrammar"

        from pvllm.sampling_params import SamplingParams as Params
        from pvllm.v1.request import Request

        request = Request(
            request_id="r0",
            prompt_token_ids=[1],
            sampling_params=Params(max_tokens=4, structured_outputs=params),
            arrival_time=0.0,
        )
        with pytest.raises(NotImplementedError, match="xgrammar"):
            manager.grammar_init(request)
        manager.shutdown()
    finally:
        llm.shutdown()


def test_no_bitmask_is_produced_and_the_reason_is_stated():
    """The absence is deliberate. If someone later makes `grammar_bitmask` return
    an array, this fails and they have to think about whether anything consumes it."""
    from pvllm.v1.structured_output import StructuredOutputManager

    llm = LLM(**BASE)
    try:
        manager = StructuredOutputManager(llm.llm_engine.vllm_config)
        assert manager.grammar_bitmask({}, {"r0": 0}) is None
    finally:
        llm.shutdown()
