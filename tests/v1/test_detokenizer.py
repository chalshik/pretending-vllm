"""Incremental detokenization and stop strings. R11.5, R11.6."""

from __future__ import annotations

import pytest

from pvllm.sampling_params import SamplingParams
from pvllm.tokenizers.mock import MockTokenizer
from pvllm.v1.engine.detokenizer import (
    SlowIncrementalDetokenizer,
    check_stop_strings,
)


def make_detokenizer(prompt: str = "", **params) -> SlowIncrementalDetokenizer:
    tokenizer = MockTokenizer(vocab_size=32_000)
    return SlowIncrementalDetokenizer(
        tokenizer,
        tokenizer.encode(prompt) if prompt else [],
        SamplingParams(**params),
    )


def feed(detok: SlowIncrementalDetokenizer, text: str) -> list[str]:
    """Feed a string one token at a time, collecting the streamed deltas."""
    tokenizer = MockTokenizer(vocab_size=32_000)
    deltas = []
    for token_id in tokenizer.encode(text, add_special_tokens=False):
        detok.update([token_id], stop_terminated=False)
        deltas.append(detok.get_next_output_text(finished=False, delta=True))
    return deltas


# --- incremental correctness (R11.6) ---------------------------------------


def test_streamed_deltas_reassemble_into_the_full_text():
    """The property every streaming client depends on."""
    detok = make_detokenizer()
    deltas = feed(detok, "hello world")
    assert "".join(deltas) == "hello world"


def test_incremental_output_matches_a_single_decode():
    """The leading-space rule: decoding a token alone gives different text than
    decoding it in context, which is why the prefix/read offsets exist."""
    tokenizer = MockTokenizer(vocab_size=32_000)
    text = "the quick brown fox"
    detok = make_detokenizer()
    feed(detok, text)
    assert detok.output_text == tokenizer.decode(
        tokenizer.encode(text, add_special_tokens=False), skip_special_tokens=True
    )


@pytest.mark.parametrize(
    "text",
    ["日本語のテキスト", "héllo wörld", "👋🏽 wave", "mixed 日本 ascii"],
)
def test_multibyte_text_streams_without_emitting_replacement_characters(text):
    """R11.6's core rule. A multi-byte character spans tokens, so a prefix decode
    ends mid-character; emitting U+FFFD then would send a client a character that
    never existed in the output, and no later token could retract it."""
    detok = make_detokenizer()
    deltas = feed(detok, text)
    assert "".join(deltas) == text
    assert "�" not in "".join(deltas)


def test_partial_bytes_are_held_then_released_together():
    """The held-back bytes must resume, not be skipped."""
    detok = make_detokenizer()
    deltas = feed(detok, "日")  # three UTF-8 bytes, three tokens
    assert deltas[0] == "" and deltas[1] == ""
    assert deltas[2] == "日"


def test_cumulative_output_grows_monotonically():
    detok = make_detokenizer()
    tokenizer = MockTokenizer(vocab_size=32_000)
    seen = ""
    for token_id in tokenizer.encode("abcdef", add_special_tokens=False):
        detok.update([token_id], stop_terminated=False)
        current = detok.get_next_output_text(finished=False, delta=False)
        assert current.startswith(seen)
        seen = current


def test_the_prompt_is_not_part_of_the_output():
    detok = make_detokenizer(prompt="a prompt")
    feed(detok, "xy")
    assert detok.output_text == "xy"
    assert detok.num_output_tokens() == 2


# --- stop strings (R11.5) --------------------------------------------------


def test_a_stop_string_ends_the_request_and_is_excluded():
    detok = make_detokenizer(stop=["STOP"])
    tokenizer = MockTokenizer(vocab_size=32_000)
    matched = None
    for token_id in tokenizer.encode("abSTOPcd", add_special_tokens=False):
        matched = detok.update([token_id], stop_terminated=False)
        if matched:
            break
    assert matched == "STOP"
    assert detok.output_text == "ab"


def test_include_stop_str_in_output_keeps_it():
    detok = make_detokenizer(stop=["STOP"], include_stop_str_in_output=True)
    tokenizer = MockTokenizer(vocab_size=32_000)
    for token_id in tokenizer.encode("abSTOPcd", add_special_tokens=False):
        if detok.update([token_id], stop_terminated=False):
            break
    assert detok.output_text == "abSTOP"


def test_text_is_held_back_while_a_stop_string_might_still_form():
    """Streaming a partial stop string and retracting it later is impossible over
    SSE, so the tail is withheld until it cannot become one."""
    detok = make_detokenizer(stop=["STOP"])
    assert detok.stop_buffer_length == 3

    deltas = feed(detok, "abcdefgh")
    streamed = "".join(deltas)
    assert streamed == "abcde"  # last 3 chars withheld
    # Once finished, nothing is held back.
    assert detok.get_next_output_text(finished=True, delta=True) == "fgh"


def test_no_stop_strings_means_no_holdback():
    detok = make_detokenizer()
    assert detok.stop_buffer_length == 0
    assert "".join(feed(detok, "abcdef")) == "abcdef"


def test_min_tokens_suppresses_an_early_stop_string():
    """Text below min_tokens is excluded from stop matching."""
    detok = make_detokenizer(stop=["ab"], min_tokens=10)
    tokenizer = MockTokenizer(vocab_size=32_000)
    matched = None
    for token_id in tokenizer.encode("abab", add_special_tokens=False):
        matched = detok.update([token_id], stop_terminated=False) or matched
    assert matched is None


def test_the_earliest_completing_stop_string_wins():
    """When several tokens arrive at once -- common with speculative decoding --
    the result must match appending one token at a time."""
    match = check_stop_strings(
        output_text="hello world END",
        new_char_count=15,
        stop=["END", "world"],
        include_in_output=False,
    )
    assert match is not None
    assert match[0] == "world"


def test_a_stop_string_straddling_the_search_boundary_is_found():
    """Searching only the newly added characters would miss a stop string that
    began in the previous chunk."""
    match = check_stop_strings(
        output_text="abcSTOP",
        new_char_count=2,  # only "OP" is new
        stop=["STOP"],
        include_in_output=False,
    )
    assert match is not None and match[0] == "STOP"


def test_no_match_returns_none():
    assert (
        check_stop_strings(
            output_text="hello", new_char_count=5, stop=["xyz"], include_in_output=False
        )
        is None
    )
    assert (
        check_stop_strings(
            output_text="hello",
            new_char_count=0,
            stop=["hello"],
            include_in_output=False,
        )
        is None
    )


def test_a_stop_terminated_token_is_recorded_but_not_rendered():
    """The token that triggered the stop counts toward usage but must not appear in
    the text."""
    detok = make_detokenizer()
    tokenizer = MockTokenizer(vocab_size=32_000)
    ids = tokenizer.encode("ab", add_special_tokens=False)
    detok.update(ids, stop_terminated=True)
    assert detok.num_output_tokens() == 2
    assert detok.output_text == "a"  # the last token was skipped
