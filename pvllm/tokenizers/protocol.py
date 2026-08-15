"""The tokenizer interface.

Upstream: vllm/tokenizers/protocol.py
Tier: C

F2: upstream moved tokenizers out of `transformers_utils` into a top-level package
with a protocol and a registry. That is a better seam than the draft spec planned for,
because it means `MockTokenizer` is a first-class implementation rather than a
monkeypatch -- it satisfies the same protocol a real Hugging Face tokenizer does.

Only the surface pretending-vllm actually calls is declared. A real tokenizer has far
more; adding unused methods here would suggest they are supported.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TokenizerLike(Protocol):
    """What the engine needs from a tokenizer."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def max_token_id(self) -> int: ...

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def pad_token_id(self) -> int | None: ...

    @property
    def all_special_ids(self) -> list[int]: ...

    @property
    def is_fast(self) -> bool: ...

    def __len__(self) -> int: ...

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]: ...

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str: ...

    def convert_ids_to_tokens(
        self, ids: Sequence[int], skip_special_tokens: bool = False
    ) -> list[str]: ...

    def convert_tokens_to_string(self, tokens: list[str]) -> str: ...

    def get_vocab(self) -> dict[str, int]: ...

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> str | list[int]: ...
