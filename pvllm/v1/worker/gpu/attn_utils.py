"""Building attention metadata and the KV cache spec.

Upstream: vllm/v1/worker/gpu/attn_utils.py
Tier: B

R8.2. The metadata is the cost model's input, so it has to be right for the *latency*
to be right -- which is what keeps it honest. Metadata that nothing reads can be wrong
indefinitely; metadata that feeds a number someone looks at cannot.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pvllm.config import VllmConfig
from pvllm.v1.attention.backends.sim_attn import SimAttentionMetadata
from pvllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from pvllm.v1.worker.gpu.block_table import BlockTables
from pvllm.v1.worker.gpu.input_batch import InputBatch


def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """One spec per attention layer. R6.7.

    Three shapes, and the per-layer return is what makes all three expressible
    without a branch anywhere downstream:

    * every layer full attention -- the ordinary dense model, one group;
    * every layer windowed -- a uniformly-windowed model, or `--sliding-window`
      forcing one, still one group;
    * a repeating mix -- Gemma-3's five windowed layers to every full one, which
      becomes several groups with a unified page size.

    `--sliding-window` overrides the card, and it makes *every* layer windowed. That
    is what the flag means upstream too: it is a way to ask "what would this model
    cost with a window", not a way to add one to a hybrid pattern.

    Under pipeline parallelism the spec describes the stage that *binds* -- the one
    whose layers cost the most -- rather than stage 0. pvllm runs one worker standing
    for the whole deployment, so whichever stage it reports is the one the pool is
    sized from, and stage 0 is systematically the cheap one: upstream pushes the
    remainder onto the middle partitions, so 32 layers over 7 stages is `4,4,5,5,5,5,4`
    and sizing from the 4 promises a pool the 5-layer stages cannot build. On a
    hybrid it is worse than a fraction -- stage 0 of a block-ordered pattern can hold
    no attention layers at all and report no KV cache.
    """
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config

    kv_cache_dtype = cache_config.resolved_cache_dtype or model_config.resolved_dtype
    from pvllm.sim.model_db import DTYPE_BYTES

    card = resolve_model_card(vllm_config)
    block_size = cache_config.block_size
    num_kv_heads = model_config.get_num_kv_heads(parallel_config.tensor_parallel_size)
    head_size = model_config.get_head_size()
    dtype_bytes = DTYPE_BYTES[kv_cache_dtype]

    def full() -> FullAttentionSpec:
        if card is not None and card.use_mla:
            # R6.7. One compressed latent per token instead of a key and a value per
            # KV head, and `num_kv_heads=1` because upstream returns 1 for MLA
            # *before* dividing by the tensor-parallel size -- the latent is
            # replicated on every rank, so TP does not shrink it.
            return MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=card.mla_head_size,
                dtype=kv_cache_dtype,
                dtype_bytes=dtype_bytes,
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=kv_cache_dtype,
            dtype_bytes=dtype_bytes,
        )

    def windowed(window: int) -> SlidingWindowSpec:
        if card is not None and card.use_mla:
            # R6.7. Same latent as `full()`, same reason: MLA decides how big a page
            # is, the window decides how many of them a request holds, and the two
            # compose. Reading the ordinary `num_kv_heads`/`head_size` here made a
            # windowed MLA model report a page 7.1x too large -- and made it shard
            # under TP, which the whole point of `full()`'s branch is that it does
            # not.
            return SlidingWindowMLASpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=card.mla_head_size,
                dtype=kv_cache_dtype,
                dtype_bytes=dtype_bytes,
                sliding_window=window,
            )
        return SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=kv_cache_dtype,
            dtype_bytes=dtype_bytes,
            sliding_window=window,
        )

    def build(start: int, end: int) -> dict[str, KVCacheSpec]:
        """Every spec for the layers `[start, end)` -- one pipeline stage's worth."""
        if cache_config.sliding_window is not None:
            uniform: KVCacheSpec = windowed(cache_config.sliding_window)
            return {f"model.layers.{i}.self_attn": uniform for i in range(start, end)}

        if card is not None and card.is_state_space:
            # R6.7. Attention layers cache KV; Mamba layers hold a fixed recurrent
            # state; MLP layers hold neither and so produce no spec at all. The
            # pattern string is what the model publishes, and reading it here keeps
            # the layer identity in one place.
            assert card.hybrid_override_pattern is not None
            state_spec = MambaSpec(
                block_size=block_size,
                state_bytes=card.mamba_state_bytes_per_layer(
                    parallel_config.tensor_parallel_size
                ),
                page_size_padded=cache_config.mamba_page_size_padded,
            )
            attention_spec = full()
            layered: dict[str, KVCacheSpec] = {}
            for index in range(start, end):
                kind = card.hybrid_override_pattern[index]
                if kind == "M":
                    layered[f"model.layers.{index}.mixer"] = state_spec
                elif kind == "*":
                    layered[f"model.layers.{index}.self_attn"] = attention_spec
            return layered

        if card is not None and card.is_hybrid_attention:
            assert card.sliding_window is not None
            if scheduler_config is not None and getattr(
                scheduler_config, "disable_hybrid_kv_cache_manager", False
            ):
                # Upstream's escape hatch: promote every windowed layer to full
                # attention, giving up the memory saving and keeping one group. Worth
                # having as more than a compatibility switch -- the two runs side by
                # side *are* the capacity argument for hybrid attention.
                return {
                    f"model.layers.{i}.self_attn": full() for i in range(start, end)
                }
            full_spec = full()
            window_spec = windowed(card.sliding_window)
            return {
                f"model.layers.{i}.self_attn": (
                    full_spec if card.layer_is_full_attention(i) else window_spec
                )
                for i in range(start, end)
            }
        if card is not None and card.sliding_window is not None:
            uniform = windowed(card.sliding_window)
            return {f"model.layers.{i}.self_attn": uniform for i in range(start, end)}

        uniform = full()
        return {f"model.layers.{i}.self_attn": uniform for i in range(start, end)}

    stages = model_config.pipeline_stage_ranges(parallel_config.pipeline_parallel_size)
    if len(stages) == 1:
        return build(*stages[0])
    # The stage that binds: the one whose layers need the most cache. Sizing the
    # pool from any cheaper stage reports blocks the deployment cannot allocate on
    # every rank, which is the direction that turns a capacity plan into an outage.
    return max(
        (build(start, end) for start, end in stages),
        key=lambda specs: sum(spec.page_size_bytes for spec in specs.values()),
    )


def resolve_model_card(vllm_config: VllmConfig) -> Any:
    """The card this deployment resolved to, or `None` if it has no counterpart.

    Read here rather than passed down because the attention shape is a property of
    the *model*, and this is the one place that turns a model into layer specs.
    """
    from pvllm.sim.model_db import load_model_card

    sim_config = vllm_config.sim_config
    name = (sim_config.model_card if sim_config is not None else None) or (
        vllm_config.model_config.model
    )
    try:
        return load_model_card(name)
    except (KeyError, FileNotFoundError):
        return None


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
