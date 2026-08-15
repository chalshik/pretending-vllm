"""The structured output backend interface. R15.

Upstream: vllm/v1/structured_output/backend_types.py
Tier: B

Ported with one substitution: `fill_bitmask` takes a numpy array rather than a
`torch.Tensor`, because NF1 forbids torch. The shape and the semantics are upstream's
-- one row per batch slot, one bit per vocabulary token, a set bit meaning "allowed".

The two levels are worth keeping straight, because they have different lifetimes.
A `StructuredOutputBackend` is engine-level and compiles grammars; a
`StructuredOutputGrammar` is request-level and tracks how far through the constraint
one request has got.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pvllm.config import VllmConfig
    from pvllm.tokenizers.protocol import TokenizerLike


class StructuredOutputOptions(enum.Enum):
    """Which kind of constraint a request carries."""

    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()


#: `(kind, spec)`. The cache key for a compiled grammar: two requests asking for the
#: same schema compile once, which is what makes a fleet serving one schema cheap.
StructuredOutputKey = tuple[StructuredOutputOptions, str]


class StructuredOutputGrammar(ABC):
    """One request's progress through its constraint."""

    @abstractmethod
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Advance the state machine. False if the tokens are not accepted."""

    @abstractmethod
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """The accepted prefix of `tokens`, without advancing."""

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        """Undo `num_tokens` accepted tokens. Used when speculation is rejected."""

    @abstractmethod
    def fill_bitmask(self, bitmask: np.ndarray, batch_index: int) -> None:
        """Set the allowed-token bits for one batch slot."""

    @abstractmethod
    def is_terminated(self) -> bool:
        """Whether the constraint has been fully satisfied."""

    @abstractmethod
    def reset(self) -> None:
        """Return to the start state."""


@dataclass
class StructuredOutputBackend(ABC):
    """Engine-level grammar compiler."""

    vllm_config: VllmConfig
    tokenizer: TokenizerLike
    vocab_size: int

    @abstractmethod
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        """Compile one constraint. Raises if the spec is unsupported or malformed."""

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> Any:
        """A `[max_num_seqs, ceil(vocab_size / 32)]` int32 bitmask."""

    @abstractmethod
    def destroy(self) -> None:
        """Backend-specific cleanup."""
