"""State-space (Mamba) KV cache groups. R6.7, R10.2.

A recurrent state is a different capacity shape from a KV cache: it is *constant* per
request where KV is linear in context. For a Nemotron-H-class hybrid that is 97 MiB of
state against 16 KiB/token of KV, so the two are equal at about 6,200 tokens.

The larger consequence is second-order and easy to miss. A state page cannot shrink, so
one pool cannot hold it beside a token-sized attention page -- upstream grows the
*attention* block size until its page covers the state, then pads the state page to
match. That moves the block size from 16 tokens to 1040 here, which changes how many
blocks a request holds (C2), how coarse the prefix cache is, and every block hash value
(C3).
"""

from __future__ import annotations

import pytest

from pvllm.engine.arg_utils import EngineArgs
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.sim.model_db import load_model_card
from pvllm.v1.core.kv_cache_utils import get_kv_cache_groups
from pvllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from pvllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

BASE = {
    "model": "hybrid-ssm-8b",
    "device_card": "datacenter-80gb",
    "max_model_len": 8192,
    "block_size": 16,
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 4,
    "disable_log_stats": True,
}


def test_the_state_is_constant_per_request():
    """Two tensors sized from the config alone: a conv ring buffer and an SSM temporal
    state. Neither grows with context, and the SSM half is 97% of it -- which is why
    its dtype matters so much."""
    card = load_model_card("hybrid-ssm-8b")
    assert card.is_state_space
    assert card.num_mamba_layers == 24
    assert card.num_attention_layers == 4
    assert card.mamba_state_bytes_per_layer() == 4_255_744
    assert card.mamba_state_bytes() / 2**20 == pytest.approx(97.41, abs=0.01)


def test_the_ssm_dtype_is_a_factor_of_two():
    """Nemotron-H keeps the SSM state in float32 while Bamba and Falcon-H1 leave it at
    the model dtype. Reading the wrong one is a 2x error in every downstream number."""
    from dataclasses import replace

    card = load_model_card("hybrid-ssm-8b")
    at_model_dtype = replace(card, mamba_ssm_cache_dtype=None)
    assert (
        card.mamba_state_bytes_per_layer()
        > at_model_dtype.mamba_state_bytes_per_layer()
    )


def test_kv_is_charged_for_attention_layers_only():
    """A hybrid's Mamba and MLP layers cache no KV at all. Counting every hidden layer
    would charge this model for 52 layers of KV where it has 4."""
    card = load_model_card("hybrid-ssm-8b")
    per_layer = 2 * card.num_key_value_heads * card.head_dim * 2
    assert card.kv_bytes_per_token("bfloat16") == per_layer * 4


def test_the_block_size_grows_until_a_page_covers_the_state():
    """The load-bearing consequence, and the one a byte-accurate implementation still
    gets wrong if it skips this step: block counts, prefix-cache granularity and every
    block hash value all move with the block size."""
    config = EngineArgs(**BASE).create_engine_config()
    # 4,255,744 B of state over 4,096 B/token of attention, rounded to the 16-token
    # alignment: 16 * ceil(4255744 / (16 * 4096)) = 1040.
    assert config.cache_config.block_size == 1040
    assert config.cache_config.mamba_page_size_padded == 1040 * 4096


def test_every_layer_ends_up_with_the_same_page():
    """One pool cannot hold two page sizes. The state pads up; attention grew."""
    config = EngineArgs(**BASE).create_engine_config()
    specs = get_kv_cache_spec(config)
    assert sum(isinstance(s, MambaSpec) for s in specs.values()) == 24
    assert sum(type(s) is FullAttentionSpec for s in specs.values()) == 4
    assert len({spec.page_size_bytes for spec in specs.values()}) == 1


def test_the_layers_split_into_equal_groups():
    config = EngineArgs(**BASE).create_engine_config()
    groups = get_kv_cache_groups(get_kv_cache_spec(config))
    assert {len(group.layer_names) for group in groups} == {4}
    assert sum(isinstance(g.kv_cache_spec, MambaSpec) for g in groups) == 6


def test_a_state_space_group_reports_a_snapshot_not_nothing():
    """The trap. A recurrent state is not a prefix, so the tempting answer is 'no hit'
    -- but the coordinator reconciles a hit across *every* group, so a group reporting
    zero drives the reconciled length to zero and throws away the attention groups'
    hits. A hybrid model's prefix cache hit rate would read exactly 0.0, and C3 calls
    that rate exact."""
    llm = LLM(**BASE, enable_prefix_caching=True)
    try:
        manager = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
        # The null block is reserved: a state-space group hands one back for every
        # boundary it did not snapshot.
        assert manager.block_pool.null_block is not None
        prompt = "a shared preamble " * 200
        llm.generate([prompt + f"q{i}" for i in range(3)], SamplingParams(max_tokens=8))
        stats = llm.llm_engine.make_stats()
    finally:
        llm.shutdown()

    # The *hits*, not the queries. Queries are counted before any group is consulted,
    # so they are bit-identical whether the state-space group answers with a snapshot
    # or with nothing -- which is exactly why asserting on them let the trap through.
    # The reconciled length is the min across groups, so a group returning `([], 0)`
    # drags the whole rate to 0.0 while every other number in this test stays put.
    assert stats["prefix_cache_queries"] > 0
    assert stats["prefix_cache_hits"] > 0
    hit_rate = stats["prefix_cache_hits"] / stats["prefix_cache_queries"]
    assert hit_rate > 0.4, hit_rate


def test_a_hybrid_state_space_model_serves():
    llm = LLM(**BASE, seed=1)
    try:
        core = llm.llm_engine.engine_core.engine_core
        assert core.kv_cache_config.num_groups == 7
        outputs = llm.generate(
            ["a prompt for the hybrid state-space model", "and another"],
            SamplingParams(max_tokens=16),
        )
        assert all(len(o.outputs[0].token_ids) == 16 for o in outputs)
    finally:
        llm.shutdown()


def test_a_model_without_state_space_layers_is_untouched():
    """The alignment must not move the block size for every other model."""
    config = EngineArgs(
        model="dense-8b",
        device_card="datacenter-80gb",
        max_model_len=2048,
        block_size=16,
    ).create_engine_config()
    assert config.cache_config.block_size == 16
    assert config.cache_config.mamba_page_size_padded is None
