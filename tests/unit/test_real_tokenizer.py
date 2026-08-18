"""The real Hugging Face tokenizer. R3.1, C3.

`MockTokenizer` is byte-level, so it inflates every token count several-fold. That is
fine for exercising plumbing and wrong for the three numbers a product reads off the
engine: `usage.prompt_tokens`, the prompt length at which the context-length error
fires, and -- because blocks are cut every `block_size` *tokens* -- where the prefix
cache's block boundaries fall.

These tests build a **real** `tokenizers.Tokenizer` locally rather than downloading
one. The code path is identical to loading a published model's; only the vocabulary is
small and local, so the suite needs no network and stays inside its time budget.
"""

from __future__ import annotations

import json

import pytest

from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.tokenizers import get_tokenizer

pytest.importorskip("tokenizers", reason="needs the `realtok` extra")

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    """A real BPE tokenizer, trained here so the test needs no network."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    path = tmp_path_factory.mktemp("tokenizer")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog"] * 40
        + ["hello world how are you today"] * 40,
        trainers.BpeTrainer(
            vocab_size=512,
            special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "chat_template": CHAT_TEMPLATE,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def real(tokenizer_dir):
    return get_tokenizer(str(tokenizer_dir), tokenizer_mode="slow", vocab_size=512)


# --- the tokenizer ----------------------------------------------------------


def test_it_tokenizes_the_way_the_model_does_not_byte_by_byte(real):
    """The whole point. A byte-level stand-in inflates every count severalfold, and
    every number a product reads off the engine is downstream of that."""
    text = "the quick brown fox jumps over the lazy dog"
    mock = get_tokenizer("tiny-test", tokenizer_mode="mock", vocab_size=1024)
    assert len(real.encode(text)) < len(mock.encode(text)) / 3


def test_encode_and_decode_round_trip(real):
    text = "hello world how are you today"
    assert real.decode(real.encode(text)) == text


def test_the_special_tokens_come_from_the_model_s_config(real):
    """`tokenizers` carries the vocabulary and the merges and nothing else -- the
    specials are declared in `tokenizer_config.json`, which is where this reads
    them."""
    assert real.bos_token_id == real.get_vocab()["<s>"]
    assert real.eos_token_id == real.get_vocab()["</s>"]
    assert real.pad_token_id == real.get_vocab()["<pad>"]
    assert real.is_fast is True


def test_it_renders_the_model_s_own_chat_template(real):
    """Substituting a house template would put the token counts back where `slow`
    mode exists to move them away from: a chat prompt's length is mostly its
    template."""
    rendered = real.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    assert rendered == "<|user|>\nhi\n<|assistant|>\n"
    assert (
        real.apply_chat_template([{"role": "user", "content": "hi"}])
        == "<|user|>\nhi\n"
    )


def test_a_tokenizer_without_a_template_refuses_by_name(tmp_path, tokenizer_dir):
    """Rather than falling back to a stand-in, which would be the same error as
    tokenizing with a different tokenizer."""
    import shutil

    shutil.copy(tokenizer_dir / "tokenizer.json", tmp_path / "tokenizer.json")
    bare = get_tokenizer(str(tmp_path), tokenizer_mode="slow", vocab_size=512)
    with pytest.raises(NotImplementedError, match="no chat_template"):
        bare.apply_chat_template([{"role": "user", "content": "hi"}])
    # It still tokenizes -- only the chat surface is lost.
    assert bare.encode("hello world")


def test_ids_the_tokenizer_cannot_name_decode_to_nothing(real):
    """A checkpoint's tokenizer is routinely smaller than the embedding matrix it is
    paired with, and the sampler draws over the *card's* range. A decode is not the
    place to relitigate that, so out-of-range ids are dropped rather than raising."""
    assert real.decode([10**9]) == ""
    assert real.decode([*real.encode("hello"), 10**9]) == "hello"


def test_a_directory_without_a_tokenizer_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"tokenizer\.json"):
        get_tokenizer(str(tmp_path), tokenizer_mode="slow", vocab_size=512)


# --- through the engine -----------------------------------------------------


def test_the_engine_reports_the_real_prompt_token_count(tokenizer_dir):
    """`usage.prompt_tokens` is what a billing estimate or a context budget is built
    on, and under the mock it is several times the number production reports."""
    text = "the quick brown fox jumps over the lazy dog"

    def prompt_tokens(**overrides) -> int:
        llm = LLM(
            model="dense-0.6b",
            device_card="workstation-24gb",
            max_model_len=1024,
            block_size=16,
            disable_log_stats=True,
            seed=1,
            **overrides,
        )
        try:
            output = llm.generate([text], SamplingParams(max_tokens=4))[0]
            return len(output.prompt_token_ids or ())
        finally:
            llm.shutdown()

    mock_count = prompt_tokens()
    real_count = prompt_tokens(tokenizer=str(tokenizer_dir), tokenizer_mode="slow")
    assert real_count < mock_count / 3


def test_the_context_length_error_fires_on_the_real_length(tokenizer_dir):
    """A prompt that fits in production must fit here. Under the mock its token count
    is several times larger, so the refusal lands on the wrong prompt."""
    text = "the quick brown fox jumps over the lazy dog " * 6

    def fits(**overrides) -> bool:
        llm = LLM(
            model="dense-0.6b",
            device_card="workstation-24gb",
            max_model_len=128,
            block_size=16,
            disable_log_stats=True,
            **overrides,
        )
        try:
            llm.generate([text], SamplingParams(max_tokens=4))
            return True
        except ValueError:
            return False
        finally:
            llm.shutdown()

    assert fits(tokenizer=str(tokenizer_dir), tokenizer_mode="slow")
    assert not fits()


def test_generated_text_decodes_through_the_real_vocabulary(tokenizer_dir):
    """The ids are still drawn rather than inferred -- a real tokenizer makes the
    output *look* like language without making it mean anything (NG3). What matters
    is that the incremental detokenizer drives the real vocabulary without
    breaking."""
    llm = LLM(
        model="dense-0.6b",
        device_card="workstation-24gb",
        max_model_len=512,
        tokenizer=str(tokenizer_dir),
        tokenizer_mode="slow",
        disable_log_stats=True,
        seed=2,
    )
    try:
        output = llm.generate(["hello world"], SamplingParams(max_tokens=16))[0]
        assert len(output.outputs[0].token_ids) == 16
        # Text comes back; it is synthetic, and the docstring says so.
        assert isinstance(output.outputs[0].text, str)
    finally:
        llm.shutdown()


def test_the_mock_remains_the_default(tokenizer_dir):
    """D3: the base install pulls no tokenizer library, so `auto` must not reach for
    one."""
    from pvllm.tokenizers.mock import MockTokenizer

    assert isinstance(
        get_tokenizer("anything", tokenizer_mode="auto", vocab_size=1024), MockTokenizer
    )
    assert isinstance(
        get_tokenizer("anything", tokenizer_mode="mock", vocab_size=1024), MockTokenizer
    )
