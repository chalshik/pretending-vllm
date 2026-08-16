"""The real Hugging Face tokenizer. R3.1, C3.

Upstream: vllm/tokenizers/hf.py
Tier: C

`MockTokenizer` is byte-level: every byte is a token, so "hello world" is eleven
tokens where a real BPE tokenizer makes two or three. That is fine for exercising the
plumbing and wrong for everything a product actually reads off it.

Three things go wrong with a synthetic tokenization, and all three are numbers a
product depends on:

* **`usage.prompt_tokens` is not the number production reports.** A billing estimate
  or a context-budget check built against it is built against fiction.
* **The context-length error fires at the wrong prompt.** A prompt that fits in
  production is refused here, or worse, the reverse.
* **The prefix cache hit rate is measured on the wrong block boundaries.** Blocks are
  cut every `block_size` *tokens*, so a different tokenization puts the cut points
  somewhere else, and two prompts that share a prefix in production may not share one
  here. C3 calls hit rate exact; the registry's own error message has said since M1
  that this is the reason `slow` mode exists.

So `tokenizer_mode="slow"` loads the tokenizer your model actually ships, and every
token count downstream becomes the one you would see on real vLLM.

**What this does not make real.** The generated text is still synthetic -- ids are
drawn, not inferred -- so a real tokenizer makes the output *look* like language
without making it mean anything (NG3). And the base install still pulls no tokenizer
library: this module is imported only under `slow` mode, behind the `realtok` extra
(D3).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pvllm.logger import init_logger

logger = init_logger(__name__)

#: Read from `tokenizer_config.json`, which is where a model declares them -- the
#: `tokenizers` library carries the vocabulary and the merges, and nothing else.
_SPECIAL_FIELDS = ("bos_token", "eos_token", "pad_token", "unk_token")


def _token_text(value: Any) -> str | None:
    """A special token as declared: either a string or an `AddedToken` dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return content if isinstance(content, str) else None
    return None


class HFTokenizer:
    """A real tokenizer, satisfying `TokenizerLike`.

    Built over `tokenizers.Tokenizer`, which is the fast tokenizer vLLM itself uses.
    `transformers` is deliberately *not* a dependency: upstream's counterpart wraps a
    `PreTrainedTokenizer`, and pulling that in would pull torch -- the one thing this
    project promises never to need. So the vocabulary and merges come from
    `tokenizers`, and the pieces `transformers` would have supplied (the special
    tokens, the chat template) are read from `tokenizer_config.json` directly.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        name: str = "",
        config: dict[str, Any] | None = None,
        model_vocab_size: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self.name = name
        config = config or {}
        self._chat_template = config.get("chat_template")

        self._special_ids: dict[str, int | None] = {}
        for field in _SPECIAL_FIELDS:
            text = _token_text(config.get(field))
            self._special_ids[field] = (
                tokenizer.token_to_id(text) if text is not None else None
            )

        self._vocab_size = int(tokenizer.get_vocab_size())
        # The *card's* vocabulary drives the memory model and the sampler's range, and
        # a checkpoint's tokenizer is routinely a little smaller than the embedding
        # matrix it is paired with (padded to a multiple of 64, or 128). Reported
        # rather than reconciled silently: an id the sampler can emit and the
        # tokenizer cannot name would decode to nothing, and the user should know
        # which of the two numbers their capacity answer used.
        if model_vocab_size is not None and model_vocab_size != self._vocab_size:
            logger.info(
                "Tokenizer %r has %d tokens; the model card declares %d. The card's "
                "number sizes the embedding table and the sampler's range; the "
                "tokenizer's decides what text encodes to. Sampled ids above the "
                "tokenizer's range decode as empty.",
                name or "<local>",
                self._vocab_size,
                model_vocab_size,
            )

    # --- construction --------------------------------------------------------

    @classmethod
    def load(cls, name: str, model_vocab_size: int | None = None) -> HFTokenizer:
        """Load from a local `tokenizer.json`, a local directory, or a Hub id.

        A Hub id fetches over the network on first use, which is a thing an engine
        does not normally do at startup -- so it happens only under `slow` mode, and
        the local paths exist for a run that must not reach out.
        """
        from tokenizers import Tokenizer

        path = Path(name)
        if path.is_file():
            return cls(
                Tokenizer.from_file(str(path)),
                name=name,
                config=cls._read_config(path.parent / "tokenizer_config.json"),
                model_vocab_size=model_vocab_size,
            )
        if path.is_dir():
            tokenizer_file = path / "tokenizer.json"
            if not tokenizer_file.is_file():
                raise FileNotFoundError(
                    f"{path} has no tokenizer.json; tokenizer_mode='slow' needs the "
                    f"fast-tokenizer file, which is what `tokenizers` reads"
                )
            return cls(
                Tokenizer.from_file(str(tokenizer_file)),
                name=name,
                config=cls._read_config(path / "tokenizer_config.json"),
                model_vocab_size=model_vocab_size,
            )

        logger.info("Fetching tokenizer %r from the Hugging Face Hub", name)
        return cls(
            Tokenizer.from_pretrained(name),
            name=name,
            config=cls._fetch_config(name),
            model_vocab_size=model_vocab_size,
        )

    @staticmethod
    def _read_config(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _fetch_config(name: str) -> dict[str, Any]:
        """The model's `tokenizer_config.json`, for its specials and chat template.

        Optional: a repository without one still tokenizes. Only the chat template
        and the special ids are lost, and both refuse by name rather than guessing.
        """
        try:
            from huggingface_hub import hf_hub_download

            return HFTokenizer._read_config(
                Path(hf_hub_download(name, "tokenizer_config.json"))
            )
        except Exception as exc:
            logger.info(
                "No tokenizer_config.json for %r (%s); special tokens and the chat "
                "template are unavailable",
                name,
                type(exc).__name__,
            )
            return {}

    # --- the protocol --------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def max_token_id(self) -> int:
        return self._vocab_size - 1

    @property
    def bos_token_id(self) -> int | None:
        return self._special_ids.get("bos_token")

    @property
    def eos_token_id(self) -> int | None:
        return self._special_ids.get("eos_token")

    @property
    def pad_token_id(self) -> int | None:
        return self._special_ids.get("pad_token")

    @property
    def all_special_ids(self) -> list[int]:
        return sorted(
            token_id
            for token_id, token in self._tokenizer.get_added_tokens_decoder().items()
            if getattr(token, "special", False)
        )

    @property
    def is_fast(self) -> bool:
        return True

    def __len__(self) -> int:
        return self._vocab_size

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        ids: list[int] = self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).ids
        if truncation and max_length is not None:
            return ids[:max_length]
        return ids

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str:
        sequence = [ids] if isinstance(ids, int) else list(ids)
        # Ids the tokenizer cannot name -- which the sampler can produce when the
        # card's vocabulary is larger -- are dropped rather than raising. The engine
        # is entitled to sample anywhere in the card's range, and a decode is not the
        # place to relitigate that.
        in_range = [i for i in sequence if 0 <= i < self._vocab_size]
        return str(
            self._tokenizer.decode(in_range, skip_special_tokens=skip_special_tokens)
        )

    def convert_ids_to_tokens(
        self, ids: Sequence[int], skip_special_tokens: bool = False
    ) -> list[str]:
        special = set(self.all_special_ids) if skip_special_tokens else set()
        tokens: list[str] = []
        for token_id in ids:
            if token_id in special:
                continue
            token = self._tokenizer.id_to_token(token_id)
            tokens.append(token if token is not None else "")
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        decoder = self._tokenizer.decoder
        if decoder is not None:
            return str(decoder.decode(tokens))
        return "".join(tokens)

    def get_vocab(self) -> dict[str, int]:
        vocab: dict[str, int] = self._tokenizer.get_vocab()
        return vocab

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Render the model's own chat template. R3.1.

        The real one, from `tokenizer_config.json`. Substituting a house template
        would put the token counts back where `slow` mode exists to move them away
        from: a chat prompt's length is mostly its template, so rendering a different
        one is the same error as tokenizing with a different tokenizer.
        """
        if not self._chat_template:
            raise NotImplementedError(
                f"tokenizer {self.name!r} declares no chat_template, so "
                f"/v1/chat/completions cannot be rendered the way this model renders "
                f"it. /v1/completions, /tokenize and /v1/embeddings work unchanged. "
                f"Use tokenizer_mode='mock' for a stand-in template, knowing its "
                f"token counts are not your model's."
            )
        try:
            from jinja2 import Environment
        except ImportError as exc:
            raise ImportError(
                "rendering a model's chat template needs jinja2. Install the extra:\n"
                "    pip install 'pretending-vllm[realtok]'"
            ) from exc

        environment = Environment(trim_blocks=True, lstrip_blocks=True)
        environment.globals["raise_exception"] = _raise_template_error
        template = environment.from_string(self._chat_template)
        return template.render(
            messages=messages,
            add_generation_prompt=kwargs.get("add_generation_prompt", False),
            bos_token=self._token_for("bos_token"),
            eos_token=self._token_for("eos_token"),
            **{k: v for k, v in kwargs.items() if k != "add_generation_prompt"},
        )

    def _token_for(self, field: str) -> str:
        token_id = self._special_ids.get(field)
        if token_id is None:
            return ""
        return self._tokenizer.id_to_token(token_id) or ""

    def __repr__(self) -> str:
        return f"HFTokenizer(name={self.name!r}, vocab_size={self._vocab_size})"


def _raise_template_error(message: str) -> None:
    """`raise_exception` is what chat templates call to refuse a message list."""
    raise ValueError(message)


__all__ = ["HFTokenizer"]
