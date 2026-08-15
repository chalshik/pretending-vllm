"""MockTokenizer. R3.2, R11.3, R11.6."""

from __future__ import annotations

import pytest

from pvllm.tokenizers import MockTokenizer, get_tokenizer
from pvllm.tokenizers.mock import BYTE_TOKEN_OFFSET, EOS_TOKEN_ID
from pvllm.tokenizers.protocol import TokenizerLike


@pytest.fixture
def tokenizer() -> MockTokenizer:
    return MockTokenizer(vocab_size=32_000)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a",
        "hello world",
        "  leading and trailing  ",
        "héllo wörld",  # multi-byte Latin
        "日本語のテキスト",  # CJK
        "👋🏽 emoji with modifier",  # multi-codepoint grapheme
        "mixed 日本 and ASCII 123",
        "tabs\tand\nnewlines\r\n",
        '{"json": true, "n": 1.5}',
    ],
)
def test_round_trip_is_exact(tokenizer, text):
    """R11.3/R11.6 both depend on this being exact, not approximate.

    Generated content must detokenize to stable text for HTTP golden tests, and
    incremental detokenization cannot be tested against a lossy round trip.
    """
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids, skip_special_tokens=True) == text


def test_every_string_is_representable():
    """Byte-level means there is no unknown token, ever."""
    tokenizer = MockTokenizer(vocab_size=32_000)
    text = "".join(chr(c) for c in range(1, 0x2FF))
    assert tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True) == text


def test_multibyte_characters_split_across_tokens():
    """The property that makes partial-UTF-8 handling testable: a slice of a token
    sequence can legitimately end mid-character, as it does with a real BPE
    tokenizer."""
    tokenizer = MockTokenizer(vocab_size=32_000)
    ids = tokenizer.encode("日", add_special_tokens=False)
    assert len(ids) == 3  # three UTF-8 bytes, three tokens

    partial = tokenizer.decode(ids[:2], skip_special_tokens=True)
    assert partial == "�"  # replacement char, not a crash
    assert tokenizer.decode(ids, skip_special_tokens=True) == "日"


def test_bos_is_added_and_can_be_skipped(tokenizer):
    ids = tokenizer.encode("hi")
    assert ids[0] == tokenizer.bos_token_id
    assert tokenizer.encode("hi", add_special_tokens=False)[0] != tokenizer.bos_token_id


def test_special_tokens_render_or_are_skipped(tokenizer):
    ids = [EOS_TOKEN_ID, *tokenizer.encode("hi", add_special_tokens=False)]
    assert tokenizer.decode(ids, skip_special_tokens=False) == "</s>hi"
    assert tokenizer.decode(ids, skip_special_tokens=True) == "hi"


def test_truncation_respects_max_length(tokenizer):
    ids = tokenizer.encode("hello world", truncation=True, max_length=4)
    assert len(ids) == 4


def test_out_of_range_ids_render_as_stable_pseudowords(tokenizer):
    """R11.3: the same token always renders the same text, so an HTTP response can
    be golden-tested."""
    first = tokenizer.decode([5000], skip_special_tokens=True)
    second = tokenizer.decode([5000], skip_special_tokens=True)
    assert first == second
    assert first.isalpha()
    assert tokenizer.decode([5001], skip_special_tokens=True) != first


def test_byte_tokens_use_the_sentencepiece_rendering(tokenizer):
    """Consumers that special-case byte-fallback tokens see what they expect."""
    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer.encode("hi", add_special_tokens=False)
    )
    assert tokens == ["<0x68>", "<0x69>"]
    assert tokenizer.convert_tokens_to_string(tokens) == "hi"


def test_vocab_size_comes_from_the_model_card():
    """max_token_id must match what the sampler and the logprobs schema report."""
    tokenizer = MockTokenizer(vocab_size=128_256)
    assert tokenizer.vocab_size == 128_256
    assert tokenizer.max_token_id == 128_255
    assert len(tokenizer) == 128_256


def test_a_vocabulary_too_small_for_the_byte_range_is_rejected():
    with pytest.raises(ValueError, match="vocab_size must exceed"):
        MockTokenizer(vocab_size=256)


def test_get_vocab_covers_bytes_and_specials(tokenizer):
    vocab = tokenizer.get_vocab()
    assert vocab["<0x41>"] == BYTE_TOKEN_OFFSET + 0x41
    assert vocab["</s>"] == EOS_TOKEN_ID


def test_chat_template_is_deterministic(tokenizer):
    messages = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    assert rendered == tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    assert rendered.endswith("<|assistant|>\n")


def test_mock_satisfies_the_protocol(tokenizer):
    """F2: the mock is a first-class implementation, not a monkeypatch."""
    assert isinstance(tokenizer, TokenizerLike)


def test_registry_defaults_to_the_mock():
    """D3: the base install pulls no transformers and downloads nothing."""
    assert isinstance(get_tokenizer("any/model", vocab_size=32_000), MockTokenizer)
    assert isinstance(
        get_tokenizer("any/model", tokenizer_mode="mock", vocab_size=32_000),
        MockTokenizer,
    )


def test_unknown_tokenizer_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown tokenizer_mode"):
        get_tokenizer("m", tokenizer_mode="fast")
