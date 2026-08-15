"""Turning OpenAI and vLLM request fields into a grammar constraint. R15.

Upstream: vllm/entrypoints/openai/engine/protocol.py
Tier: B

(Upstream split the monolithic `openai/protocol.py` into per-endpoint modules with a
shared `engine/protocol.py`; `structured_outputs_from_response_format` there is this
module's counterpart. `tools/spec_sync.py` caught the stale path.)

Two vocabularies reach the same place. OpenAI's `response_format` carries
`{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {...}}`; vLLM's
own extensions carry `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`.
Products use both, often in the same codebase, so both are accepted -- and asking for
two at once is refused rather than silently resolved, because there is no defined
precedence and guessing one would produce output shaped like the field the caller
did not mean.
"""

from __future__ import annotations

from typing import Any

from pvllm.sampling_params import StructuredOutputsParams

#: The `response_format` types OpenAI defines. `text` is the default and means
#: unconstrained, which is why it maps to no params rather than to an error.
_RESPONSE_FORMAT_TYPES = ("text", "json_object", "json_schema")


def build_structured_outputs(
    response_format: dict[str, Any] | None = None,
    guided_json: str | dict[str, Any] | None = None,
    guided_regex: str | None = None,
    guided_choice: list[str] | None = None,
    guided_grammar: str | None = None,
    structural_tag: str | None = None,
    guided_whitespace_pattern: str | None = None,
    guided_decoding_backend: str | None = None,
) -> StructuredOutputsParams | None:
    """The constraint a request asked for, or `None` if it asked for none."""
    from_format = _from_response_format(response_format)
    guided = {
        "json": guided_json,
        "regex": guided_regex,
        "choice": guided_choice,
        "grammar": guided_grammar,
        "structural_tag": structural_tag,
    }
    named = [name for name, value in guided.items() if value is not None]

    if from_format is not None and named:
        raise ValueError(
            f"a request may set response_format or the guided_* extensions, not "
            f"both; this one set response_format and {named}"
        )
    if len(named) > 1:
        raise ValueError(
            f"only one guided decoding constraint may be set, but {named} were"
        )

    if from_format is not None:
        params = from_format
    elif named:
        # Built by name rather than by `**{name: value}` so each field keeps its own
        # type. The splat form type-checks as a union against every parameter at
        # once, which is a real looseness and not just a mypy complaint: a caller
        # passing a string where a list belongs would reach the grammar backend.
        chosen = named[0]
        if chosen == "json":
            params = StructuredOutputsParams(json=guided_json)
        elif chosen == "regex":
            params = StructuredOutputsParams(regex=guided_regex)
        elif chosen == "choice":
            params = StructuredOutputsParams(choice=guided_choice)
        elif chosen == "grammar":
            params = StructuredOutputsParams(grammar=guided_grammar)
        else:
            params = StructuredOutputsParams(structural_tag=structural_tag)
    else:
        return None

    if guided_whitespace_pattern is not None:
        params.whitespace_pattern = guided_whitespace_pattern
    if guided_decoding_backend is not None:
        params._backend = guided_decoding_backend
    return params


def _from_response_format(
    response_format: dict[str, Any] | None,
) -> StructuredOutputsParams | None:
    if not response_format:
        return None
    kind = response_format.get("type")
    if kind is None:
        raise ValueError("response_format must have a 'type'")
    if kind not in _RESPONSE_FORMAT_TYPES:
        raise ValueError(
            f"unknown response_format type {kind!r}; expected one of "
            f"{list(_RESPONSE_FORMAT_TYPES)}"
        )
    if kind == "text":
        return None
    if kind == "json_object":
        return StructuredOutputsParams(json_object=True)

    schema_wrapper = response_format.get("json_schema")
    if not schema_wrapper:
        raise ValueError(
            "response_format type 'json_schema' requires a 'json_schema' object"
        )
    schema = schema_wrapper.get("schema")
    if schema is None:
        raise ValueError(
            "response_format.json_schema requires a 'schema'. Note that OpenAI nests "
            "it one level deeper than the guided_json extension does: "
            "{'type': 'json_schema', 'json_schema': {'name': ..., 'schema': {...}}}"
        )
    return StructuredOutputsParams(json=schema)
