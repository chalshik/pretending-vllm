"""Multi-head latent attention. R6.7, R10.2.

MLA is the one attention variant that changes the *shape* of the KV cache rather than
which parts of it get read, and both of its differences move a capacity answer by a
lot in directions a plan gets wrong from first principles:

* one compressed latent per token instead of a key and a value per KV head, so the
  factor of two disappears along with the head count;
* the latent is replicated on every tensor-parallel rank rather than sharded, so
  scaling TP buys weights and compute on a DeepSeek-class model and buys *nothing* on
  its KV cache.
"""

from __future__ import annotations

import pytest

from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.sim.model_db import load_model_card
from pvllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

BASE = {
    "model": "mla-16b",
    "device_card": "datacenter-80gb",
    "max_model_len": 8192,
    "block_size": 16,
    "max_num_batched_tokens": 512,
    "max_num_seqs": 4,
    "disable_log_stats": True,
}


def test_the_card_lands_on_the_family_it_claims():
    """Derived from the dimensions, not overridden: DeepSeek-V2-Lite is 15.7B total
    and ~2.4B active."""
    card = load_model_card("mla-16b")
    assert card.use_mla
    assert card.mla_head_size == 512 + 64
    assert card.num_parameters / 1e9 == pytest.approx(15.7, abs=0.6)
    assert card.num_active_parameters / 1e9 == pytest.approx(2.4, abs=0.4)


def test_the_latent_replaces_a_key_and_a_value_per_head():
    """Upstream models MLA as full attention with one factor of two removed:
    `AttentionSpec.real_page_size_bytes` is `2 * block * heads * head_size * dtype`
    and `MLAAttentionSpec` drops the two, because there is one latent rather than a
    key and a value."""
    shared = {
        "block_size": 16,
        "num_kv_heads": 1,
        "head_size": 576,
        "dtype": "bfloat16",
        "dtype_bytes": 2,
    }
    assert MLAAttentionSpec(**shared).page_size_bytes == 16 * 1 * 576 * 2
    assert FullAttentionSpec(**shared).page_size_bytes == 2 * 16 * 1 * 576 * 2


def test_tensor_parallelism_does_not_shrink_the_latent():
    """`get_num_kv_heads` returns 1 for MLA *before* dividing by the tensor-parallel
    size. A GQA model's KV per device divides by TP; MLA's does not, and a plan that
    assumes otherwise over-provisions context by the TP factor."""
    mla = load_model_card("mla-16b")
    gqa = load_model_card("dense-8b")
    assert mla.kv_bytes_per_token("bfloat16", 1) == mla.kv_bytes_per_token(
        "bfloat16", 8
    )
    assert (
        gqa.kv_bytes_per_token("bfloat16", 8)
        == gqa.kv_bytes_per_token("bfloat16", 1) // 8
    )


def test_the_engine_resolves_an_mla_spec_and_serves():
    llm = LLM(**BASE)
    try:
        core = llm.llm_engine.engine_core.engine_core
        spec = core.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        assert isinstance(spec, MLAAttentionSpec)
        assert spec.page_size_bytes == 16 * 576 * 2
        output = llm.generate(["a prompt"], SamplingParams(max_tokens=8))[0]
        assert len(output.outputs[0].token_ids) == 8
    finally:
        llm.shutdown()


def test_kv_per_block_is_unchanged_by_tensor_parallelism():
    """The engine-level statement of the same fact: more TP ranks free memory by
    sharding *weights*, and every byte of that goes to more blocks of the same size --
    not to smaller blocks."""

    def kv_bytes_per_block(tp_size: int) -> int:
        llm = LLM(**BASE, tensor_parallel_size=tp_size)
        try:
            profile = llm.llm_engine.engine_core.engine_core.executor.driver_worker.memory_profile
            assert profile is not None
            return profile.kv_bytes_per_block
        finally:
            llm.shutdown()

    assert kv_bytes_per_block(1) == kv_bytes_per_block(8)


def test_mla_holds_far_less_kv_than_the_attention_it_replaces():
    """The headline: 27 layers of a 576-wide latent against 32 layers of 8 KV heads at
    128 dimensions, key and value. A context length decision turns on this ratio."""
    mla = load_model_card("mla-16b")
    gqa = load_model_card("dense-8b")
    per_layer_mla = mla.kv_bytes_per_token("bfloat16") / mla.num_hidden_layers
    per_layer_gqa = gqa.kv_bytes_per_token("bfloat16") / gqa.num_hidden_layers
    assert per_layer_mla < per_layer_gqa / 3
