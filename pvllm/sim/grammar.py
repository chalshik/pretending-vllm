"""Generating strings that satisfy a constraint. R15.1, R11.3.

Upstream: (none -- simulator)
Tier: D

This is where pretending-vllm departs from upstream's *mechanism* in order to keep its
*contract*, and the reasoning is worth stating plainly because the shortcut is
tempting in both directions.

Upstream constrains structured output by masking the sampler: at every step the
grammar marks which vocabulary tokens are legal, and the model picks among them.
Reproducing that here would be easy and useless. The model is a token generator with
no language model behind it and the tokenizer is a mock whose vocabulary is synthetic
pseudowords, so a JSON grammar over that vocabulary would admit sequences that are
"legal" in the emulated bitmask and detokenize to nothing resembling JSON. A product
that called `json.loads()` on the result would fail -- against the one feature it
uses structured output *for*.

So the constraint is satisfied at the level a consumer can observe: this module
generates a string that really does conform, and `SimModel` emits that string's
tokens. What R15.1 requires to be real -- the scheduler-side interaction -- is real
and lives elsewhere: asynchronous compilation, the grammar-wait status, admission
gating, per-request compile failure. No token bitmask is computed at all; see
`pvllm/sim/structured_output.py` for why a mask nothing consumes would be worse than
none.

Everything here is deterministic given a seeded generator, so a request's structured
output is reproducible like every other part of a run (B4).

**Self-checking where it can be.** A generated JSON value is validated against the
schema it came from, and a generated regex match is checked with `re.fullmatch`,
before either is returned. A generator that drifts from its own specification fails
here rather than shipping a plausible string that does not actually conform.
"""

from __future__ import annotations

import json
import re
import string
from collections.abc import Callable
from typing import Any

import numpy as np

#: Words drawn on for generated string values. Fixed and boring: the content is not
#: the point, and R11.3 wants text that is stable across runs so golden tests work.
_WORDS = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
)

#: Keywords a real grammar backend honours and this generator cannot. Listed
#: explicitly so adding support is a deletion from this set rather than a hunt.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "multipleOf",
        "uniqueItems",
        "prefixItems",
        "contains",
        "minContains",
        "maxContains",
        "not",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "patternProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
    }
)

#: How deep a schema may nest before this refuses. Schemas are user input, and a
#: recursive `$ref` would otherwise recurse until the interpreter gives up.
MAX_DEPTH = 12


class UnsupportedConstraintError(NotImplementedError):
    """A constraint this generator cannot satisfy.

    `NotImplementedError` rather than `ValueError` on purpose: it names a feature
    that upstream supports and this does not, which is the project's standing rule
    for a dropped path. Silently emitting a string that does not conform would be
    much worse -- the product would parse garbage and blame itself.
    """


# --- JSON schema -----------------------------------------------------------


def generate_json(
    schema: dict[str, Any] | None, rng: np.random.Generator, depth: int = 0
) -> Any:
    """A value conforming to `schema`. `None` schema means any object.

    Supports the subset of JSON Schema that shows up in practice around LLM output:
    types, `enum`, `const`, `properties`/`required`/`additionalProperties`, `items`,
    `minItems`/`maxItems`, `minimum`/`maximum`, `minLength`/`maxLength`, `anyOf`,
    `oneOf`, and `$ref` into `$defs`/`definitions`. Anything else raises rather than
    being approximated.
    """
    if depth > MAX_DEPTH:
        raise UnsupportedConstraintError(
            f"schema nests deeper than {MAX_DEPTH} levels; a recursive $ref cannot "
            f"be satisfied by a generator that must terminate"
        )
    if schema is None:
        schema = {"type": "object"}
    return _generate(schema, schema, rng, depth)


def _generate(
    schema: dict[str, Any],
    root: dict[str, Any],
    rng: np.random.Generator,
    depth: int,
) -> Any:
    if depth > MAX_DEPTH:
        raise UnsupportedConstraintError(f"schema nests deeper than {MAX_DEPTH} levels")
    if not isinstance(schema, dict):
        raise UnsupportedConstraintError(
            f"a schema must be an object, got {type(schema).__name__}"
        )

    if "$ref" in schema:
        return _generate(_resolve_ref(schema["$ref"], root), root, rng, depth + 1)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        options = schema["enum"]
        if not options:
            raise UnsupportedConstraintError("an empty enum admits no value")
        return options[int(rng.integers(len(options)))]
    for combinator in ("anyOf", "oneOf"):
        if combinator in schema:
            branches = schema[combinator]
            if not branches:
                raise UnsupportedConstraintError(
                    f"an empty {combinator} admits no value"
                )
            # The first branch, not a random one: `anyOf: [T, null]` is how optional
            # fields are spelled, and picking randomly would make a field appear and
            # vanish between runs of the same seed-stable workload.
            return _generate(branches[0], root, rng, depth + 1)
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for branch in schema["allOf"]:
            value = _generate(branch, root, rng, depth + 1)
            if not isinstance(value, dict):
                raise UnsupportedConstraintError(
                    "allOf is only supported over object schemas"
                )
            merged.update(value)
        return merged

    # Constructs this generator does not honour. Raised rather than ignored: a
    # schema asking for `multipleOf: 10` and getting 85 back is worse than an error,
    # because the product only finds out when its own validator rejects the engine's
    # output and it goes looking in the wrong place.
    unsupported = _UNSUPPORTED_KEYWORDS & schema.keys()
    if unsupported:
        raise UnsupportedConstraintError(
            f"JSON Schema keyword(s) {sorted(unsupported)} are not honoured by the "
            f"simulated grammar backend. Generating a value that satisfies them "
            f"needs a constraint solver; supported: type, enum, const, properties, "
            f"required, items, minItems/maxItems, minimum/maximum, "
            f"exclusiveMinimum/exclusiveMaximum, minLength/maxLength, pattern, "
            f"format, anyOf, oneOf, allOf, $ref."
        )

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        # A union of types. First non-null, same reasoning as anyOf.
        schema_type = next((t for t in schema_type if t != "null"), "null")
    if schema_type is None:
        # No type and no combinator: JSON Schema calls this "any". An object is the
        # useful answer, since that is what a caller who omitted the type meant.
        schema_type = "object"

    if schema_type == "object":
        return _generate_object(schema, root, rng, depth)
    if schema_type == "array":
        return _generate_array(schema, root, rng, depth)
    if schema_type == "string":
        return _generate_string(schema, rng)
    if schema_type == "integer":
        return _generate_integer(schema, rng)
    if schema_type == "number":
        return _generate_number(schema, rng)
    if schema_type == "boolean":
        return bool(rng.integers(2))
    if schema_type == "null":
        return None
    raise UnsupportedConstraintError(f"unsupported JSON schema type {schema_type!r}")


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise UnsupportedConstraintError(
            f"only local $ref is supported, got {ref!r}; a remote reference would "
            f"need a fetch this engine will not make"
        )
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            raise UnsupportedConstraintError(f"$ref {ref!r} does not resolve")
        node = node[part]
    if not isinstance(node, dict):
        raise UnsupportedConstraintError(f"$ref {ref!r} does not point at a schema")
    return node


def _generate_object(
    schema: dict[str, Any],
    root: dict[str, Any],
    rng: np.random.Generator,
    depth: int,
) -> dict[str, Any]:
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or ())

    result: dict[str, Any] = {}
    for name, subschema in properties.items():
        # Required always; optional included too. A schema-conforming object may
        # omit optional fields, but a product testing its parser wants to see them
        # populated -- an engine that silently dropped them would make the test
        # double weaker than the thing it stands in for.
        result[name] = _generate(subschema, root, rng, depth + 1)

    missing = required - set(result)
    if missing:
        raise UnsupportedConstraintError(
            f"schema requires {sorted(missing)} but declares no properties for them"
        )
    if not properties:
        # `{"type": "object"}` with nothing else. Produce something rather than an
        # empty object, so a caller eyeballing the output can tell it worked.
        result = {"result": _WORDS[int(rng.integers(len(_WORDS)))]}
    return result


def _generate_array(
    schema: dict[str, Any],
    root: dict[str, Any],
    rng: np.random.Generator,
    depth: int,
) -> list[Any]:
    items = schema.get("items")
    low = int(schema.get("minItems", 1))
    high = int(schema.get("maxItems", max(low, 3)))
    if high < low:
        raise UnsupportedConstraintError(
            f"maxItems ({high}) is below minItems ({low}); no array satisfies this"
        )
    count = int(rng.integers(low, high + 1))
    if items is None:
        return [_WORDS[int(rng.integers(len(_WORDS)))] for _ in range(count)]
    if isinstance(items, list):
        raise UnsupportedConstraintError(
            "tuple-style `items` (a list of schemas) is not supported"
        )
    return [_generate(items, root, rng, depth + 1) for _ in range(count)]


def _generate_string(schema: dict[str, Any], rng: np.random.Generator) -> str:
    if "pattern" in schema:
        return generate_regex_match(schema["pattern"], rng)
    fmt = schema.get("format")
    if fmt == "date-time":
        return "2026-01-01T00:00:00Z"
    if fmt == "date":
        return "2026-01-01"
    if fmt == "email":
        return "someone@example.com"
    if fmt == "uuid":
        return "00000000-0000-4000-8000-000000000000"

    minimum = int(schema.get("minLength", 0))
    maximum = int(schema.get("maxLength", max(minimum, 24)))
    if maximum < minimum:
        raise UnsupportedConstraintError(
            f"maxLength ({maximum}) is below minLength ({minimum})"
        )
    value = _WORDS[int(rng.integers(len(_WORDS)))]
    while len(value) < minimum:
        value += " " + _WORDS[int(rng.integers(len(_WORDS)))]
    return value[:maximum]


def _integer_bounds(schema: dict[str, Any]) -> tuple[int | None, int | None]:
    """`(low, high)` inclusive, or `None` where the schema sets no bound.

    `None` rather than a default, because defaulting the *missing* side to 0 made a
    schema whose only bound was a negative maximum -- `{"maximum": -5}`, a perfectly
    ordinary temperature delta -- come back as "maximum (-5) is below minimum (0);
    no integer satisfies this", which is false. Defaults are applied afterwards,
    relative to whichever bound exists.
    """
    low: int | None = None
    high: int | None = None
    if "minimum" in schema:
        low = int(schema["minimum"])
    if "exclusiveMinimum" in schema:
        # Draft 2020-12 spells these as numbers, which is what Pydantic emits for
        # `Field(gt=..., lt=...)`. Ignoring them produced values outside the range
        # a product had asked for and believed it had.
        bound = int(schema["exclusiveMinimum"]) + 1
        low = bound if low is None else max(low, bound)
    if "maximum" in schema:
        high = int(schema["maximum"])
    if "exclusiveMaximum" in schema:
        bound = int(schema["exclusiveMaximum"]) - 1
        high = bound if high is None else min(high, bound)
    return low, high


def _generate_integer(schema: dict[str, Any], rng: np.random.Generator) -> int:
    low, high = _integer_bounds(schema)
    if low is None:
        low = 0 if high is None else min(0, high - 100)
    if high is None:
        high = low + 100
    if high < low:
        raise UnsupportedConstraintError(
            f"maximum ({high}) is below minimum ({low}); no integer satisfies this"
        )
    return int(rng.integers(low, high + 1))


#: How far inside an exclusive bound a generated float lands. Small enough not to
#: distort a narrow range, large enough to survive the rounding below -- a value
#: exactly *on* an exclusive bound violates the schema.
_EXCLUSIVE_EPSILON = 1e-3


def _generate_number(schema: dict[str, Any], rng: np.random.Generator) -> float:
    low: float | None = None
    high: float | None = None
    if "minimum" in schema:
        low = float(schema["minimum"])
    if "exclusiveMinimum" in schema:
        bound = float(schema["exclusiveMinimum"]) + _EXCLUSIVE_EPSILON
        low = bound if low is None else max(low, bound)
    if "maximum" in schema:
        high = float(schema["maximum"])
    if "exclusiveMaximum" in schema:
        bound = float(schema["exclusiveMaximum"]) - _EXCLUSIVE_EPSILON
        high = bound if high is None else min(high, bound)

    if low is None:
        low = 0.0 if high is None else min(0.0, high - 100.0)
    if high is None:
        high = low + 100.0
    if high < low:
        raise UnsupportedConstraintError(
            f"maximum ({high}) is below minimum ({low}); no number satisfies this"
        )
    # Rounded so the value round-trips through JSON without a long float tail, which
    # would make golden output noisy for no benefit. Clamped afterwards because
    # rounding can push a value back onto an exclusive bound it had cleared.
    value = round(float(rng.uniform(low, high)), 4)
    return min(max(value, low), high)


def validate_against_schema(value: Any, schema: dict[str, Any] | None) -> None:
    """Check a generated value really does conform. Raises `AssertionError` if not.

    Deliberately a *self*-check rather than a full JSON Schema validator: it verifies
    the constraints this generator claims to honour. A generator that drifts from its
    own specification is caught here, at the moment it drifts, instead of shipping a
    string that a product will fail to parse for reasons it cannot see.
    """
    if schema is None:
        return
    _validate(value, schema, schema, path="$")


def _validate(
    value: Any, schema: dict[str, Any], root: dict[str, Any], path: str
) -> None:
    if "$ref" in schema:
        _validate(value, _resolve_ref(schema["$ref"], root), root, path)
        return
    if "const" in schema:
        assert value == schema["const"], (
            f"{path}: {value!r} != const {schema['const']!r}"
        )
        return
    if "enum" in schema:
        assert value in schema["enum"], f"{path}: {value!r} not in enum"
        return
    for combinator in ("anyOf", "oneOf"):
        if combinator in schema:
            _validate(value, schema[combinator][0], root, path)
            return
    if "allOf" in schema:
        return

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((t for t in schema_type if t != "null"), "null")
    if schema_type in (None, "object"):
        assert isinstance(value, dict), f"{path}: expected object, got {type(value)}"
        for name in schema.get("required") or ():
            assert name in value, f"{path}: required property {name!r} is missing"
        for name, subschema in (schema.get("properties") or {}).items():
            if name in value:
                _validate(value[name], subschema, root, f"{path}.{name}")
    elif schema_type == "array":
        assert isinstance(value, list), f"{path}: expected array, got {type(value)}"
        if "minItems" in schema:
            assert len(value) >= schema["minItems"], f"{path}: too few items"
        if "maxItems" in schema:
            assert len(value) <= schema["maxItems"], f"{path}: too many items"
        items = schema.get("items")
        if isinstance(items, dict):
            for index, element in enumerate(value):
                _validate(element, items, root, f"{path}[{index}]")
    elif schema_type == "string":
        assert isinstance(value, str), f"{path}: expected string, got {type(value)}"
        if "minLength" in schema:
            assert len(value) >= schema["minLength"], f"{path}: too short"
        if "maxLength" in schema:
            assert len(value) <= schema["maxLength"], f"{path}: too long"
        if "pattern" in schema:
            assert re.fullmatch(schema["pattern"], value), f"{path}: pattern mismatch"
    elif schema_type == "integer":
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"{path}: expected integer, got {type(value)}"
        )
        _validate_bounds(value, schema, path)
    elif schema_type == "number":
        assert isinstance(value, int | float) and not isinstance(value, bool), (
            f"{path}: expected number, got {type(value)}"
        )
        _validate_bounds(value, schema, path)
    elif schema_type == "boolean":
        assert isinstance(value, bool), f"{path}: expected boolean, got {type(value)}"
    elif schema_type == "null":
        assert value is None, f"{path}: expected null, got {value!r}"


def _validate_bounds(value: float, schema: dict[str, Any], path: str) -> None:
    if "minimum" in schema:
        assert value >= schema["minimum"], f"{path}: {value} below minimum"
    if "maximum" in schema:
        assert value <= schema["maximum"], f"{path}: {value} above maximum"
    if "exclusiveMinimum" in schema:
        assert value > schema["exclusiveMinimum"], (
            f"{path}: {value} not above exclusiveMinimum"
        )
    if "exclusiveMaximum" in schema:
        assert value < schema["exclusiveMaximum"], (
            f"{path}: {value} not below exclusiveMaximum"
        )


# --- regex -----------------------------------------------------------------
#
# A generator over the subset of regex syntax that appears in guided-decoding
# requests: literals, classes, escapes, groups, alternation, and the quantifiers.
# Lookarounds, backreferences, and anchors inside the pattern are refused rather than
# ignored -- ignoring them would produce a string that does not match, which is the
# one outcome worse than an error.

_UNSUPPORTED_REGEX = {
    "(?=": "lookahead",
    "(?!": "negative lookahead",
    "(?<=": "lookbehind",
    "(?<!": "negative lookbehind",
    r"\b": "word boundary",
    r"\B": "non-word boundary",
}

#: What an unbounded quantifier expands to. Small: `a+` in a guided-decoding pattern
#: means "at least one", and generating four hundred is not more correct.
_STAR_MAX = 3


def generate_regex_match(pattern: str, rng: np.random.Generator) -> str:
    """A string matching `pattern`, verified with `re.fullmatch` before it returns."""
    for marker, name in _UNSUPPORTED_REGEX.items():
        if marker in pattern:
            raise UnsupportedConstraintError(
                f"regex {name} ({marker}) is not supported by the simulated grammar "
                f"backend; a generator cannot satisfy it without a solver"
            )
    if any(back in pattern for back in (r"\1", r"\2", r"\3")):
        raise UnsupportedConstraintError(
            "regex backreferences are not supported by the simulated grammar backend"
        )

    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc

    parser = _RegexGenerator(pattern, rng)
    value = parser.generate()

    if not re.fullmatch(pattern, value):
        # Never expected. Raised rather than returned because a string that does not
        # match is exactly the failure this whole module exists to prevent, and the
        # caller has no way to notice it.
        raise UnsupportedConstraintError(
            f"the simulated grammar backend generated {value!r}, which does not match "
            f"{pattern!r}. This is a bug in pvllm/sim/grammar.py, not in your pattern."
        )
    return value


class _RegexGenerator:
    """A recursive-descent generator over a regex pattern."""

    def __init__(self, pattern: str, rng: np.random.Generator) -> None:
        self.pattern = pattern
        self.rng = rng
        self.pos = 0

    def generate(self) -> str:
        # Anchors that wrap the whole pattern are redundant under `fullmatch` and
        # common in hand-written constraints, so they are stripped rather than
        # refused.
        if self.pattern.startswith("^"):
            self.pos = 1
        text = self._alternation()
        if self.pos < len(self.pattern) and self.pattern[self.pos] == "$":
            self.pos += 1
        if self.pos < len(self.pattern):
            raise UnsupportedConstraintError(
                f"unparsed regex remainder {self.pattern[self.pos :]!r} in "
                f"{self.pattern!r}"
            )
        return text

    def _alternation(self) -> str:
        branches = [self._sequence()]
        while self.pos < len(self.pattern) and self.pattern[self.pos] == "|":
            self.pos += 1
            branches.append(self._sequence())
        return branches[int(self.rng.integers(len(branches)))]

    def _sequence(self) -> str:
        parts: list[str] = []
        while self.pos < len(self.pattern) and self.pattern[self.pos] not in "|)$":
            parts.append(self._quantified())
        return "".join(parts)

    def _quantified(self) -> str:
        atom = self._atom()
        if self.pos >= len(self.pattern):
            return atom()
        char = self.pattern[self.pos]
        if char == "?":
            self.pos += 1
            self._maybe_lazy()
            return atom() if self.rng.integers(2) else ""
        if char == "*":
            self.pos += 1
            self._maybe_lazy()
            return "".join(atom() for _ in range(int(self.rng.integers(_STAR_MAX + 1))))
        if char == "+":
            self.pos += 1
            self._maybe_lazy()
            return "".join(atom() for _ in range(1 + int(self.rng.integers(_STAR_MAX))))
        if char == "{":
            low, high = self._repeat_bounds()
            return "".join(atom() for _ in range(int(self.rng.integers(low, high + 1))))
        return atom()

    def _maybe_lazy(self) -> None:
        if self.pos < len(self.pattern) and self.pattern[self.pos] in "?+":
            self.pos += 1

    def _repeat_bounds(self) -> tuple[int, int]:
        close = self.pattern.index("}", self.pos)
        body = self.pattern[self.pos + 1 : close]
        self.pos = close + 1
        if "," not in body:
            count = int(body)
            return count, count
        low_text, high_text = body.split(",", 1)
        low = int(low_text) if low_text else 0
        high = int(high_text) if high_text else low + _STAR_MAX
        if high < low:
            raise ValueError(f"regex repeat {{{body}}} has an inverted range")
        return low, high

    def _atom(self) -> Callable[[], str]:
        """Returns a *callable* producing one match of the next atom.

        A callable rather than a string because a quantifier repeats the atom, and
        `a{3}` over a character class must be free to pick a different character each
        time -- returning a string here would repeat one draw three times, which is a
        different language.
        """
        char = self.pattern[self.pos]

        if char == "(":
            self.pos += 1
            if self.pattern.startswith("?:", self.pos):
                self.pos += 2
            elif self.pattern.startswith("?", self.pos):
                raise UnsupportedConstraintError(
                    f"regex group flags are not supported: "
                    f"{self.pattern[self.pos : self.pos + 3]!r}"
                )
            start = self.pos
            text = self._alternation()
            if self.pos >= len(self.pattern) or self.pattern[self.pos] != ")":
                raise ValueError(f"unbalanced ( in regex {self.pattern!r}")
            end = self.pos
            self.pos += 1
            # Re-parsed per repetition, for the reason in the docstring above.
            inner = self.pattern[start:end]
            return lambda: (
                _RegexGenerator(inner, self.rng).generate() if inner else text
            )

        if char == "[":
            class_members = self._char_class()
            return lambda: class_members[int(self.rng.integers(len(class_members)))]

        if char == "\\":
            self.pos += 2
            escape = self.pattern[self.pos - 1]
            escape_members = _ESCAPE_CLASSES.get(escape)
            if escape_members is not None:
                return lambda: escape_members[
                    int(self.rng.integers(len(escape_members)))
                ]
            return lambda: escape

        if char == ".":
            self.pos += 1
            return lambda: _WORDS[int(self.rng.integers(len(_WORDS)))][0]

        self.pos += 1
        return lambda: char

    def _char_class(self) -> str:
        self.pos += 1
        negated = self.pattern[self.pos] == "^"
        if negated:
            self.pos += 1
        members: list[str] = []
        while self.pattern[self.pos] != "]":
            if self.pattern[self.pos] == "\\":
                escape = self.pattern[self.pos + 1]
                members.extend(_ESCAPE_CLASSES.get(escape, escape))
                self.pos += 2
                continue
            if (
                self.pos + 2 < len(self.pattern)
                and self.pattern[self.pos + 1] == "-"
                and self.pattern[self.pos + 2] != "]"
            ):
                low, high = self.pattern[self.pos], self.pattern[self.pos + 2]
                members.extend(chr(c) for c in range(ord(low), ord(high) + 1))
                self.pos += 3
                continue
            members.append(self.pattern[self.pos])
            self.pos += 1
        self.pos += 1

        if negated:
            excluded = set(members)
            members = [
                c for c in string.ascii_letters + string.digits if c not in excluded
            ]
        if not members:
            raise ValueError(f"character class in {self.pattern!r} admits nothing")
        return "".join(members)


_ESCAPE_CLASSES = {
    "d": string.digits,
    "w": string.ascii_letters + string.digits + "_",
    "s": " ",
    "D": string.ascii_letters,
    "W": " -",
    "S": string.ascii_letters + string.digits,
}


# --- the top-level entry point ---------------------------------------------


def generate_for_constraint(kind: str, spec: str, rng: np.random.Generator) -> str:
    """The text a constrained request should produce.

    `kind` is a `StructuredOutputOptions` name, lowercased -- taken as a string so
    this module, which is Tier D, does not import the control plane.
    """
    if kind == "json":
        schema = json.loads(spec) if spec else None
        value = generate_json(schema, rng)
        validate_against_schema(value, schema)
        return json.dumps(value)
    if kind == "json_object":
        return json.dumps(generate_json(None, rng))
    if kind == "choice":
        options = json.loads(spec)
        if not options:
            raise ValueError("a choice constraint must list at least one option")
        return str(options[int(rng.integers(len(options)))])
    if kind == "regex":
        return generate_regex_match(spec, rng)
    if kind == "grammar":
        raise UnsupportedConstraintError(
            "EBNF/Lark grammar constraints are not supported by the simulated "
            "grammar backend (R15). Generating a conforming string needs a real "
            "grammar engine; json, json_object, choice, and regex are supported."
        )
    if kind == "structural_tag":
        raise UnsupportedConstraintError(
            "structural tag constraints are not supported by the simulated grammar "
            "backend (R15). json, json_object, choice, and regex are supported."
        )
    raise ValueError(f"unknown structured output constraint kind {kind!r}")
