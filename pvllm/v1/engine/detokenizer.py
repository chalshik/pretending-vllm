"""Per-request incremental detokenization and stop strings.

Upstream: vllm/v1/engine/detokenizer.py
Tier: A

R11.5 and R11.6. Stop *strings* live here rather than in the scheduler's `check_stop`
because they need text, and text only exists once tokens have been detokenized -- which
is exactly why they are the fiddly half of stop handling.

Three behaviours a stream-consuming product will notice if they are wrong:

* **Held-back text.** With a stop string configured and `include_stop_str_in_output`
  off, the last `max(len(stop)) - 1` characters are withheld from the stream, because
  they might turn out to be the start of a stop string. Streaming them and retracting
  later is impossible over SSE.
* **`min_tokens` suppression.** Text produced below `min_tokens` is excluded from stop
  matching, so a stop string appearing early cannot end the request.
* **Earliest match wins.** When several tokens arrive at once and more than one stop
  string matches, the one completing earliest is chosen -- so the result matches what
  appending one token at a time would have produced.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from pvllm.sampling_params import SamplingParams
from pvllm.tokenizers.detokenizer_utils import (
    convert_prompt_ids_to_tokens,
    detokenize_incrementally,
)
from pvllm.tokenizers.protocol import TokenizerLike


class IncrementalDetokenizer(ABC):
    """Turns a growing token stream into a growing text stream."""

    def __init__(self) -> None:
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids)

    @abstractmethod
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """Absorb new tokens. Returns the matched stop string, if any."""

    @abstractmethod
    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """Text to hand the client. `delta` returns only what is new."""


class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):
    """Stop-string handling, independent of how tokens become text."""

    def __init__(self, sampling_params: SamplingParams) -> None:
        super().__init__()
        stop = sampling_params.stop
        self.stop: list[str] = list(stop) if stop else []
        self.min_tokens = sampling_params.min_tokens
        self.include_stop_str_in_output = sampling_params.include_stop_str_in_output

        # Characters to withhold from the stream: any tail shorter than the longest
        # stop string might still become one, and SSE cannot retract.
        self.stop_buffer_length = (
            max(len(s) for s in self.stop) - 1
            if self.stop and not self.include_stop_str_in_output
            else 0
        )
        self._last_output_text_offset = 0
        self.output_text = ""

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        if not new_token_ids:
            return None

        skipped_stop_token_id: int | None = None
        if stop_terminated and not self.include_stop_str_in_output:
            # The token that triggered the stop is recorded but not rendered.
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]

        stop_check_offset = len(self.output_text)
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)
            # Text produced below min_tokens is excluded from stop matching by
            # moving the search window past it.
            if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
                stop_check_offset = len(self.output_text)

        if skipped_stop_token_id is not None:
            self.token_ids.append(skipped_stop_token_id)

        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:
            match = check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - stop_check_offset,
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if match is not None:
                stop_string, truncate_to = match
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]
        return stop_string

    @abstractmethod
    def decode_next(self, next_token_id: int) -> str: ...

    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """Text for the client.

        Once finished nothing is held back, because no further token can extend a
        partial stop string.
        """
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            return (
                self.output_text[:-buffer_length] if buffer_length else self.output_text
            )

        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""


class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):
    """Detokenizes through the tokenizer protocol, one token at a time.

    Named for its upstream counterpart. Upstream also has a `Fast` variant backed by
    the Rust tokenizers library's own incremental decoder; there is no equivalent to
    wrap here, and the "slow" path is the one whose behaviour R11.6 specifies.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> None:
        super().__init__(sampling_params)
        self.tokenizer = tokenizer
        self.prompt_len = len(prompt_token_ids)
        self.skip_special_tokens = sampling_params.skip_special_tokens
        self.spaces_between_special_tokens = (
            sampling_params.spaces_between_special_tokens
        )

        self.tokens, self.prefix_offset, self.read_offset = (
            convert_prompt_ids_to_tokens(
                tokenizer,
                prompt_token_ids,
                skip_special_tokens=self.skip_special_tokens,
            )
        )
        self.token_ids.extend(prompt_token_ids)

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :] if self.prompt_len else self.token_ids

    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len

    def decode_next(self, next_token_id: int) -> str:
        new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=self.tokenizer,
            all_input_ids=self.token_ids,
            prev_tokens=self.tokens,
            prefix_offset=self.prefix_offset,
            read_offset=self.read_offset,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )
        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset
        return decoded_text


def check_stop_strings(
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """Find a stop string in the newly generated text.

    Returns `(stop_string, truncate_to)`, where `truncate_to` is -1 for no truncation.

    When several stop strings match at once -- which speculative decoding and
    multi-token steps make common -- the one completing **earliest** wins, so the
    result matches what appending one token at a time would have produced. Ties break
    on the order of the stop list.
    """
    if not new_char_count or not stop:
        return None

    best_stop_str: str | None = None
    best_stop_index = 0
    best_end = sys.maxsize

    for stop_str in stop:
        stop_string_len = len(stop_str)
        # Start the search far enough back that a stop string straddling the
        # boundary between old and new text is still found, without rescanning
        # text that was already checked.
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        if stop_index == -1:
            continue
        end = stop_index + stop_string_len
        if end < best_end:
            best_stop_str = stop_str
            best_stop_index = stop_index
            best_end = end

    if best_stop_str is None:
        return None

    if include_in_output:
        if best_end >= len(output_text):
            return best_stop_str, -1
        return best_stop_str, best_end
    return best_stop_str, best_stop_index
