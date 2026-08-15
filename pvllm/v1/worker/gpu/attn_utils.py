"""Building attention metadata and the KV cache spec.

Upstream: vllm/v1/worker/gpu/attn_utils.py
Tier: B

R8.2. The metadata is the cost model's input, so it has to be right for the *latency*
to be right -- which is what keeps it honest. Metadata that nothing reads can be wrong
indefinitely; metadata that feeds a number someone looks at cannot.
"""

from __future__ import annotations

import numpy as np

from pvllm.config import VllmConfig
from pvllm.v1.attention.backends.sim_attn import SimAttentionMetadata
from pvllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)
from pvllm.v1.worker.gpu.block_table import BlockTables
from pvllm.v1.worker.gpu.input_batch import InputBatch


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """One spec per attention layer.

    Every layer of a dense model shares a spec, so they collapse into one KV cache
    group. Hybrid models (R6.7) produce more than one; the per-layer shape is what
    makes that expressible without changing this function.
    """
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config

    kv_cache_dtype = cache_config.resolved_cache_dtype or model_config.resolved_dtype
    from pvllm.sim.model_db import DTYPE_BYTES

    # R6.7. A configured window makes every layer a windowed one -- which is what a
    # uniformly-windowed model (Mistral-style) looks like. Models that *mix* full and
    # windowed layers (Gemma-2 style) need two groups with a unified page size, and
    # `_initialize_kv_caches` refuses them by name rather than collapsing them into
    # one group, which would report the wrong capacity for both halves.
    if cache_config.sliding_window is not None:
        windowed = SlidingWindowSpec(
            block_size=cache_config.block_size,
            num_kv_heads=model_config.get_num_kv_heads(
                parallel_config.tensor_parallel_size
            ),
            head_size=model_config.get_head_size(),
            dtype=kv_cache_dtype,
            dtype_bytes=DTYPE_BYTES[kv_cache_dtype],
            sliding_window=cache_config.sliding_window,
        )
        num_layers = model_config.get_num_layers(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
        )
        return {f"layer.{i}": windowed for i in range(num_layers)}

    spec = FullAttentionSpec(
        block_size=cache_config.block_size,
        num_kv_heads=model_config.get_num_kv_heads(
            parallel_config.tensor_parallel_size
        ),
        head_size=model_config.get_head_size(),
        dtype=kv_cache_dtype,
        dtype_bytes=DTYPE_BYTES[kv_cache_dtype],
    )
    num_layers = model_config.get_num_layers(
        parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size
    )
    return {f"model.layers.{i}.self_attn": spec for i in range(num_layers)}


def build_attn_metadata(
    input_batch: InputBatch,
    block_tables: BlockTables,
    num_common_prefix_blocks: int = 0,
    decode_query_len: int = 1,
    group_id: int = 0,
) -> SimAttentionMetadata:
    """Assemble the metadata a real attention kernel would consume. R8.2.

    Computing the slot mapping here rather than lazily is deliberate: it is what
    validates the KV manager (R8.3), and a lazily-built mapping would only be checked
    on the paths that happened to read it.
    """
    slot_mapping = block_tables.compute_slot_mapping(
        input_batch.idx_mapping_np,
        input_batch.positions,
        input_batch.query_start_loc_np,
        group_id=group_id,
    )
    block_table = block_tables.gather(input_batch.idx_mapping_np, group_id=group_id)

    # A request contributing exactly one token is decoding; anything more is a
    # prefill or a prefill chunk. The split drives the cost model's compute term and
    # the graph-capture decision (R8.4).
    query_lens = input_batch.num_scheduled_tokens
    is_decode = query_lens <= decode_query_len
    num_decodes = int(np.count_nonzero(is_decode))
    num_decode_tokens = int(query_lens[is_decode].sum()) if num_decodes else 0

    return SimAttentionMetadata(
        query_start_loc=input_batch.query_start_loc_np,
        seq_lens=input_batch.seq_lens_np,
        slot_mapping=slot_mapping,
        block_table=block_table,
        num_reqs=input_batch.num_reqs,
        num_actual_tokens=input_batch.num_tokens,
        max_query_len=int(query_lens.max()) if input_batch.num_reqs else 0,
        max_seq_len=int(input_batch.seq_lens_np.max()) if input_batch.num_reqs else 0,
        num_prefill_tokens=input_batch.num_tokens - num_decode_tokens,
        num_decode_tokens=num_decode_tokens,
        num_prefills=input_batch.num_reqs - num_decodes,
        num_decodes=num_decodes,
        num_common_prefix_blocks=num_common_prefix_blocks,
    )
