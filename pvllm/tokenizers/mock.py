"""A deterministic, reversible tokenizer that needs no model files.

Upstream: (none -- pvllm addition)
Tier: D

R3.2/D3: the default, so the base install needs no `transformers` and no downloads.

**Reversibility is the design constraint.** `decode(encode(text)) == text` for any
input, exactly -- not approximately. Two things depend on it: R11.3 requires generated
content to detokenize to stable text so HTTP-level golden tests are possible, and R11.6
requires real incremental detokenization, which cannot be tested against a tokenizer
whose round trip loses information.

That is achieved by tokenizing **UTF-8 bytes**, not characters. Byte-level means every
possible string is representable, multi-byte characters split across token boundaries
the way a real BPE tokenizer's do -- so the partial-UTF-8 handling in the detokenizer
is genuinely exercised -- and there is no unknown token, ever.

What this deliberately is *not*: a realistic tokenizer. Token counts per word are far
higher than a real BPE tokenizer's, so prompt lengths differ from what a real model
would see. That matters for prefix cache hit rates on real text, which is why D3 makes
the `realtok` extra mandatory for conformance class C3.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: Reserved ids below this are special tokens. Chosen so the byte range that follows
#: starts at a round number, which makes traces easier to read by eye.
NUM_SPECIAL_TOKENS = 256

BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 2
UNK_TOKEN_ID = 3

#: Byte value `b` encodes to `BYTE_TOKEN_OFFSET + b`, so ids 256..511 are the bytes.
BYTE_TOKEN_OFFSET = NUM_SPECIAL_TOKENS

_SPECIAL_TOKEN_TEXT = {
    BOS_TOKEN_ID: "<s>",
    EOS_TOKEN_ID: "</s>",
    PAD_TOKEN_ID: "<pad>",
    UNK_TOKEN_ID: "<unk>",
}


class MockTokenizer:
    """A byte-level tokenizer with a model-card-sized vocabulary.

    Args:
        vocab_size: Taken from the model card, so `max_token_id` matches what the
            sampler and logprobs schema report. Ids above the byte range are never
            produced by `encode` but are valid sampler outputs -- `SimModel` emits
            them, and `decode` renders them as pseudo-words (R11.3).
        add_bos: Whether `encode` prepends BOS by default.
    """

    def __init__(self, vocab_size: int = 32_000, add_bos: bool = True) -> None:
        if vocab_size <= BYTE_TOKEN_OFFSET + 256:
            raise ValueError(
                f"vocab_size must exceed {BYTE_TOKEN_OFFSET + 256} to hold the "
                f"special tokens and the byte range, got {vocab_size}"
            )
        self._vocab_size = vocab_size
        self._add_bos = add_bos

    # --- properties ----------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def max_token_id(self) -> int:
        return self._vocab_size - 1

    @property
    def bos_token_id(self) -> int | None:
        return BOS_TOKEN_ID if self._add_bos else None

    @property
    def eos_token_id(self) -> int | None:
        return EOS_TOKEN_ID

    @property
    def pad_token_id(self) -> int | None:
        return PAD_TOKEN_ID

    @property
    def all_special_ids(self) -> list[int]:
        return list(_SPECIAL_TOKEN_TEXT)

    @property
    def is_fast(self) -> bool:
        return True

    def __len__(self) -> int:
        return self._vocab_size

    # --- encode / decode -----------------------------------------------------

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        ids = [BYTE_TOKEN_OFFSET + b for b in text.encode("utf-8")]
        if add_special_tokens and self._add_bos:
            ids.insert(0, BOS_TOKEN_ID)
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str:
        if isinstance(ids, int):
            ids = [ids]

        out: list[str] = []
        pending: bytearray = bytearray()

        def flush() -> None:
            if pending:
                # `errors="replace"` rather than raising: a *slice* of a token
                # sequence can legitimately end mid-character, and the incremental
                # detokenizer relies on decoding prefixes (R11.6).
                out.append(pending.decode("utf-8", errors="replace"))
                pending.clear()

        for token_id in ids:
            if token_id in _SPECIAL_TOKEN_TEXT:
                flush()
                if not skip_special_tokens:
                    out.append(_SPECIAL_TOKEN_TEXT[token_id])
            elif BYTE_TOKEN_OFFSET <= token_id < BYTE_TOKEN_OFFSET + 256:
                pending.append(token_id - BYTE_TOKEN_OFFSET)
            else:
                flush()
                out.append(self._pseudo_word(token_id))
        flush()
        return "".join(out)

    def convert_ids_to_tokens(
        self, ids: Sequence[int], skip_special_tokens: bool = False
    ) -> list[str]:
        tokens: list[str] = []
        for token_id in ids:
            if token_id in _SPECIAL_TOKEN_TEXT:
                if not skip_special_tokens:
                    tokens.append(_SPECIAL_TOKEN_TEXT[token_id])
            elif BYTE_TOKEN_OFFSET <= token_id < BYTE_TOKEN_OFFSET + 256:
                # The `<0xNN>` form is how byte-fallback tokens render in real
                # sentencepiece vocabularies, so consumers that special-case them
                # see what they expect.
                tokens.append(f"<0x{token_id - BYTE_TOKEN_OFFSET:02X}>")
            else:
                tokens.append(self._pseudo_word(token_id))
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        out = bytearray()
        text: list[str] = []
        for token in tokens:
            if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
                out.append(int(token[3:5], 16))
            else:
                if out:
                    text.append(out.decode("utf-8", errors="replace"))
                    out.clear()
                text.append(token)
        if out:
            text.append(out.decode("utf-8", errors="replace"))
        return "".join(text)

    def get_vocab(self) -> dict[str, int]:
        """The byte and special vocabulary.

        Deliberately excludes the pseudo-word range: materializing a dict with a
        150k-entry vocabulary on every call would dominate startup, and nothing reads
        those entries.
        """
        vocab = {text: token_id for token_id, text in _SPECIAL_TOKEN_TEXT.items()}
        vocab.update({f"<0x{b:02X}>": BYTE_TOKEN_OFFSET + b for b in range(256)})
        return vocab

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """A minimal, stable chat template.

        Not any real model's template. It exists so `/v1/chat/completions` has
        something deterministic to flatten messages with; a real template arrives with
        the `realtok` extra (R3.1, P2).
        """
        parts = [
            f"<|{message.get('role', 'user')}|>\n{message.get('content', '')}\n"
            for message in messages
        ]
        if kwargs.get("add_generation_prompt"):
            parts.append("<|assistant|>\n")
        return "".join(parts)

    def _pseudo_word(self, token_id: int) -> str:
        """Render an out-of-byte-range id as a stable pseudo-word.

        Deterministic in the id alone, so the same token always renders the same
        text -- which is what lets HTTP responses be golden-tested (R11.3).
        """
        consonants = "bdfgklmnprstvz"
        vowels = "aeiou"
        n = token_id
        syllables = []
        for _ in range(2):
            syllables.append(consonants[n % len(consonants)])
            n //= len(consonants)
            syllables.append(vowels[n % len(vowels)])
            n //= len(vowels)
        return "".join(syllables)

    def __repr__(self) -> str:
        return f"MockTokenizer(vocab_size={self._vocab_size})"
