"""Per-request structured output state. R15.

Upstream: vllm/v1/structured_output/request.py
Tier: B

Ported closely, including the part that looks over-engineered until you see why it is
there: `_grammar` may be a `Future`, because compilation happens on a thread pool and
the scheduler must not block on it. A schema that takes 200 ms to compile would
otherwise stall every other request in the engine for 200 ms.

The reasoning-parser fields upstream carries are dropped -- reasoning models are not
simulated -- and the dropped path raises where it would be reached.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from concurrent.futures import Future, TimeoutError
from typing import cast

from pvllm.sampling_params import SamplingParams, StructuredOutputsParams
from pvllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputKey,
    StructuredOutputOptions,
)


@dataclasses.dataclass
class StructuredOutputRequest:
    """One request's constraint and its compiled grammar, once ready."""

    params: StructuredOutputsParams
    _grammar: (
        Future[StructuredOutputGrammar] | StructuredOutputGrammar | Exception | None
    ) = None

    @staticmethod
    def from_sampling_params(
        sampling_params: SamplingParams | None,
    ) -> StructuredOutputRequest | None:
        """`None` when the request is unconstrained, which is the common case."""
        if sampling_params is None:
            return None
        params = sampling_params.structured_outputs
        if not params or params.all_constraints_none():
            return None
        return StructuredOutputRequest(params=params)

    def _check_grammar_completion(self) -> bool:
        if isinstance(self._grammar, Future):
            try:
                # A tiny timeout rather than zero: `result(0)` on a future that just
                # completed can still raise, and the scheduler asks this once per
                # step, so a hundred microseconds of patience saves a whole step of
                # delay for a grammar that is essentially ready.
                self._grammar = self._grammar.result(timeout=0.0001)
            except TimeoutError:
                return False
            except Exception as exc:
                # Held rather than raised: a schema that fails to compile must fail
                # *its own request*, not the engine step that happened to notice.
                self._grammar = exc
        return True

    @property
    def is_grammar_ready(self) -> bool:
        return self._check_grammar_completion()

    @property
    def grammar(self) -> StructuredOutputGrammar | Exception | None:
        if not self._check_grammar_completion():
            return None
        return cast("StructuredOutputGrammar | Exception | None", self._grammar)

    @grammar.setter
    def grammar(
        self, grammar: StructuredOutputGrammar | Future[StructuredOutputGrammar]
    ) -> None:
        self._grammar = grammar

    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey:
        return get_structured_output_key(self.params)


def get_structured_output_key(params: StructuredOutputsParams) -> StructuredOutputKey:
    """`(kind, spec)` for a constraint. The compiled-grammar cache key."""
    if params.json is not None:
        spec = params.json if isinstance(params.json, str) else json.dumps(params.json)
        return StructuredOutputOptions.JSON, spec
    if params.json_object:
        return StructuredOutputOptions.JSON_OBJECT, ""
    if params.regex is not None:
        return StructuredOutputOptions.REGEX, params.regex
    if params.choice is not None:
        # Always JSON-encoded, unlike upstream, which also accepts a pre-encoded
        # string because its field is loosely typed. Here `choice` is `list[str]`,
        # so there is one representation and the cache key cannot depend on which
        # form the caller happened to use.
        return StructuredOutputOptions.CHOICE, json.dumps(params.choice)
    if params.grammar is not None:
        return StructuredOutputOptions.GRAMMAR, params.grammar
    if params.structural_tag is not None:
        return StructuredOutputOptions.STRUCTURAL_TAG, params.structural_tag
    raise ValueError("no structured output constraint is set")
