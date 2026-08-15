"""Tokenizer selection.

Upstream: vllm/tokenizers/registry.py
Tier: C

R3.2/D3: `MockTokenizer` is the default so the base install pulls no `transformers`
and downloads nothing. A real Hugging Face tokenizer comes from the `realtok` extra
and is *mandatory* for conformance class C3, because prefix cache hit rates on real
text depend on the exact tokenization -- a byte-level mock will not reproduce them.
"""

from __future__ import annotations

from pvllm.logger import init_logger
from pvllm.tokenizers.mock import MockTokenizer
from pvllm.tokenizers.protocol import TokenizerLike

logger = init_logger(__name__)


def get_tokenizer(
    tokenizer_name: str,
    *,
    tokenizer_mode: str = "auto",
    vocab_size: int = 32_000,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> TokenizerLike:
    """Build the tokenizer for a model.

    Args:
        tokenizer_name: Model or tokenizer id. Only consulted in `slow` mode; the
            mock is architecture-agnostic.
        tokenizer_mode: `auto` and `mock` both give `MockTokenizer`. `slow` loads a
            real Hugging Face tokenizer and needs the `realtok` extra.
        vocab_size: From the model card, so `max_token_id` and the logprobs schema
            report what the model would.
    """
    if tokenizer_mode in ("auto", "mock"):
        return MockTokenizer(vocab_size=vocab_size)

    if tokenizer_mode == "slow":
        try:
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "tokenizer_mode='slow' needs a real tokenizer. Install the extra:\n"
                "    pip install 'pretending-vllm[realtok]'\n"
                "This is required for conformance class C3, since prefix cache hit "
                "rates on real text depend on exact tokenization."
            ) from exc
        raise NotImplementedError(
            "the real Hugging Face tokenizer path lands with the chat template work "
            "in M3 (requirement R3.1); only MockTokenizer is wired up so far"
        )

    raise ValueError(
        f"unknown tokenizer_mode {tokenizer_mode!r}; expected 'auto', 'mock', or 'slow'"
    )
