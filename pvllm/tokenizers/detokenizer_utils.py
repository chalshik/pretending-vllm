"""Incremental detokenization. R11.6.

Upstream: vllm/tokenizers/detokenizer_utils.py
Tier: A

A real port, not a stub, because R11.6 calls this out as "a genuine source of bugs in
stream consuming products" -- and it is. Two rules do the work, and both are easy to
get subtly wrong:

**Partial UTF-8.** A multi-byte character can span token boundaries, so decoding a
prefix of the token stream can end mid-character. When it does, the decoded text ends
in U+FFFD and the *correct* behaviour is to emit nothing and not advance -- the bytes
are held until the character completes. Emitting the replacement character instead
sends a client a `�` that never existed in the output, and no later token can retract
it. Note the check is `endswith`, not `in`: a replacement character in the *middle*
is a genuinely invalid id the model produced, and holding back on that would stall the
stream forever.

**The leading-space rule.** Tokenizers decide whether to insert a space based on
surrounding tokens, so decoding token N alone gives different text than decoding it in
context. The prefix/read offsets exist to defeat that: text is decoded from a window
starting several tokens back, and only the delta beyond the previously-decoded prefix
is emitted. Without it, streamed output differs from the same text decoded in one go.

Offsets advance only when text is actually emitted. That is what makes held-back bytes
resume correctly rather than being skipped.
"""

from __future__ import annotations

from pvllm.tokenizers.protocol import TokenizerLike

#: How many tokens of context to decode behind the read position. Upstream's value.
#: Enough to cover any tokenizer's lookbehind for spacing decisions.
INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5


def convert_prompt_ids_to_tokens(
    tokenizer: TokenizerLike,
    prompt_ids: list[int],
    skip_special_tokens: bool = False,
) -> tuple[list[str], int, int]:
    """Seed the incremental state from a prompt.

    Only the tail of the prompt is converted: incremental detokenization needs a few
    tokens of context, not the whole prompt, and converting a 100k-token prompt to
    strings on every request would cost more than the generation.
    """
    window = prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2 :]
    new_tokens = tokenizer.convert_ids_to_tokens(
        window, skip_special_tokens=skip_special_tokens
    )
    read_offset = len(new_tokens)
    prefix_offset = max(read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
    return list(new_tokens), prefix_offset, read_offset


def detokenize_incrementally(
    tokenizer: TokenizerLike,
    all_input_ids: list[int],
    prev_tokens: list[str] | None,
    prefix_offset: int,
    read_offset: int,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> tuple[list[str], str, int, int]:
    """Decode the newest token in context.

    Returns `(new_tokens, new_text, prefix_offset, read_offset)`. `new_text` is empty
    when the newest token did not complete any emittable text -- a partial UTF-8
    sequence -- and the offsets are then returned unchanged so the next token retries
    from the same point.
    """
    new_token_id = all_input_ids[-1]
    is_first_iter = prev_tokens is None
    if is_first_iter:
        prev_tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
            tokenizer, all_input_ids[:-1], skip_special_tokens=skip_special_tokens
        )
    assert prev_tokens is not None

    if 0 <= new_token_id < len(tokenizer):
        # Converted as a one-element list so `skip_special_tokens` applies.
        new_tokens = list(
            tokenizer.convert_ids_to_tokens(
                [new_token_id], skip_special_tokens=skip_special_tokens
            )
        )
    else:
        # Out of vocabulary. Contributes no text rather than raising: a sampler bug
        # should not take down a stream mid-response.
        new_tokens = [""]

    output_tokens = prev_tokens + new_tokens
    if is_first_iter:
        new_tokens = output_tokens

    # Decode twice over overlapping windows: the prefix as it was, and the prefix
    # plus the new token. The difference is what this token actually contributed,
    # which is not the same as decoding the token alone (the leading-space rule).
    prefix_text = tokenizer.convert_tokens_to_string(
        output_tokens[prefix_offset:read_offset]
    )
    new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:])

    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        # Nothing new, or an unfinished multi-byte character. Hold the bytes and do
        # not advance -- the next token completes them.
        return new_tokens, "", prefix_offset, read_offset

    return new_tokens, new_text[len(prefix_text) :], read_offset, len(output_tokens)
