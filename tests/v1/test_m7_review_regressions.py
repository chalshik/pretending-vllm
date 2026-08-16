"""Regressions for the thirteen defects the M7 adversarial review confirmed.

One test per defect, and each was checked against the pre-fix tree (a git worktree at
e7ec8d6) to be sure it fails there. That check is the whole discipline: the last four
reviews found that a fix plus a test asserting the reported *symptom* passes for both
the right fix and the wrong one, and four of the defects in the previous round were in
this session's own repairs.

The review's shape, for the record: five finders over M7a/M7b and the livelock fix,
then an adversarial refuter per finding that defaulted to "refuted" unless it
independently reproduced. Eight survived. Two of the two refutations dismantled their
target and surfaced a different real defect underneath -- those are covered here too
(`the binding pipeline stage`, `the published layer pattern`), which is why thirteen
tests answer ten confirmed findings.

Imports are local to each test so that a missing symbol fails one test rather than
collecting the file to nothing -- which is what running this against the pre-fix tree
depends on.
"""

from __future__ import annotations

import pytest

from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams

GIB = 1 << 30


def _profile(llm: LLM):
    worker = llm.llm_engine.engine_core.engine_core.executor.driver_worker
    assert worker.memory_profile is not None
    return worker.memory_profile


def _specs(llm: LLM) -> dict:
    return llm.llm_engine.engine_core.engine_core.executor.driver_worker.get_kv_cache_spec()


# --- MLA -------------------------------------------------------------------


def test_a_window_on_an_mla_model_keeps_the_latent_page():
    """`--sliding-window` used to route around every MLA branch.

    `get_kv_cache_spec` asked `card.use_mla` in `full()` and nowhere else, so a
    windowed MLA model was built as ordinary multi-head attention: 16 KV heads of 128
    instead of one latent of 576, a page 7.11x too large. Upstream carries a
    `SlidingWindowMLASpec` for exactly this overlap.
    """
    from pvllm.v1.kv_cache_interface import SlidingWindowMLASpec

    llm = LLM(
        model="mla-16b",
        device_card="datacenter-80gb",
        max_model_len=8192,
        block_size=16,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        sliding_window=1024,
        disable_log_stats=True,
    )
    try:
        spec = next(iter(_specs(llm).values()))
        assert isinstance(spec, SlidingWindowMLASpec)
        # 16 tokens * 1 latent * (512 + 64) * 2 bytes. The non-MLA answer is 131072.
        assert spec.page_size_bytes == 18432
        profile = _profile(llm)
        # The profile used to report two per-token figures 7.11x apart: this one from
        # the card (MLA-aware) and the block one from the spec (MLA-blind).
        assert profile.kv_bytes_per_block == profile.kv_bytes_per_token * 16
    finally:
        llm.shutdown()


def test_a_window_raises_an_mla_models_concurrency_rather_than_lowering_it():
    """The observable a capacity plan reads. Asking for a window used to *reduce*
    reported concurrency by ~2.7x, which is the opposite of what a window does."""

    def concurrency(**kw) -> float:
        llm = LLM(
            model="mla-16b",
            device_card="datacenter-80gb",
            max_model_len=8192,
            block_size=16,
            max_num_batched_tokens=2048,
            max_num_seqs=4,
            disable_log_stats=True,
            **kw,
        )
        try:
            return _profile(llm).max_concurrency
        finally:
            llm.shutdown()

    assert concurrency(sliding_window=1024) > concurrency()


def test_a_windowed_mla_page_does_not_shard_under_tensor_parallelism():
    """M7a's headline invariant, which the window silently broke. The page went
    131072 -> 16384 across tp=1..8, so at tp=8 it was *smaller* than the true MLA
    page and the profile flipped from under- to over-reporting capacity."""
    pages = []
    for tp in (1, 2, 8):
        llm = LLM(
            model="mla-16b",
            device_card="datacenter-80gb",
            max_model_len=8192,
            block_size=16,
            max_num_batched_tokens=2048,
            max_num_seqs=4,
            sliding_window=1024,
            tensor_parallel_size=tp,
            disable_log_stats=True,
        )
        try:
            pages.append(next(iter(_specs(llm).values())).page_size_bytes)
        finally:
            llm.shutdown()
    assert pages == [18432, 18432, 18432]


def test_a_card_that_declares_half_of_mla_is_refused_by_name():
    """`use_mla` was `kv_lora_rank is not None and qk_rope_head_dim is not None`, and
    the field comment enumerated two states where the code accepted three. A card with
    the second field misspelled took the ordinary-attention branch and was sized from
    `num_key_value_heads` -- a field an MLA architecture declares and does not use for
    KV -- reporting 7.11x the KV per token with the rank silently ignored."""
    import json
    from pathlib import Path

    from pvllm.sim.model_db import ModelCard

    card = json.loads(Path("pvllm/sim/models/mla-16b.json").read_text())
    card.pop("qk_rope_head_dim")
    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        ModelCard.from_dict(card)


def test_the_attention_backend_hook_refuses_by_name():
    """The guard M7a's commit says it removed lived in `get_attn_backend_cls`, which
    nothing called -- so the platform had never refused MLA, and the *other* check in
    that method (an unsupported attention backend) was inert for the same reason. The
    runner now resolves its backend through the platform, as upstream's
    `v1/attention/selector.py` does."""
    import pvllm.envs as envs
    from pvllm.platforms import current_platform

    assert envs.PVLLM_ATTENTION_BACKEND is None  # default: nothing pinned
    with pytest.raises(NotImplementedError, match="FLASH_ATTN"):
        current_platform.get_attn_backend_cls(
            selected_backend="FLASH_ATTN",
            head_size=128,
            dtype="bfloat16",
            kv_cache_dtype="bfloat16",
            block_size=16,
            use_mla=False,
            has_sink=False,
        )


# --- state-space groups -----------------------------------------------------

STATE_SPACE = {
    "model": "hybrid-ssm-8b",
    "device_card": "datacenter-80gb",
    "max_model_len": 8192,
    "block_size": 16,
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 4,
    "disable_log_stats": True,
}


def test_a_recurrent_state_is_one_page_however_long_the_context():
    """`MambaManager` inherited `FullAttentionManager` whole, including its "nothing is
    ever skipped" accounting, so a request pinned `ceil(tokens / block_size)` state
    pages -- 6 at 2.4k tokens, growing with context. Upstream keeps exactly one:
    "Mamba only need to keep the state of the last computed token". The card, the
    README and the commit message all claimed the constant, and the code did not."""
    from pvllm.v1.kv_cache_interface import MambaSpec

    llm = LLM(**STATE_SPACE, enable_prefix_caching=True)
    try:
        core = llm.llm_engine.engine_core.engine_core
        manager = core.scheduler.kv_cache_manager
        groups = manager.kv_cache_config.kv_cache_groups
        llm.llm_engine.add_request(
            "r0", " ".join(f"w{i}" for i in range(1200)), SamplingParams(max_tokens=4)
        )
        peak = {}
        for _ in range(60):
            llm.llm_engine.step()
            if not llm.llm_engine.has_unfinished_requests():
                break
            for index, single in enumerate(manager.coordinator.single_type_managers):
                real = sum(
                    1 for b in single.req_to_blocks.get("r0", ()) if not b.is_null
                )
                peak[index] = max(peak.get(index, 0), real)
        state_groups = [
            i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec)
        ]
        attention_groups = [i for i in peak if i not in state_groups]
        assert state_groups and attention_groups
        assert all(peak[i] == 1 for i in state_groups), peak
        # The contrast is the point: the attention group in the same model is linear.
        assert max(peak[i] for i in attention_groups) > 1
    finally:
        llm.shutdown()


def test_the_memory_profile_charges_one_state_per_request():
    """The same defect in the reported number. `compute_memory_profile` charged a
    Mamba group `ceil(max_model_len / block_size)` -- and did it in a branch whose body
    was byte-identical to the `else` two lines below, so the branch added for
    state-space groups changed nothing at all."""
    from pvllm.sim.memory import state_blocks_for_one_request

    # A recurrent state's peak does not grow with context; an attention group's does.
    short = state_blocks_for_one_request(1040, 8192, 512)
    long_ctx = state_blocks_for_one_request(1040, 65536, 512)
    assert short == long_ctx == 2

    def concurrency(max_model_len: int) -> float:
        llm = LLM(**{**STATE_SPACE, "max_model_len": max_model_len})
        try:
            return _profile(llm).max_concurrency
        finally:
            llm.shutdown()

    # Six of seven groups are recurrent, so doubling the context must cost far less
    # than half the concurrency. Under the old arithmetic it cost exactly half.
    assert concurrency(4096) / concurrency(8192) < 1.6


def test_a_state_space_cache_hit_survives_the_recompute_one_token_trim():
    """`get_computed_blocks` looked up at full length and then trimmed each group with
    a *prefix* slice. That is only valid for a downward-closed group: a state-space
    hit is `[null, ..., null, state]`, so the slice kept the placeholders and threw the
    state away while still advancing `num_computed_tokens`. Upstream caps the search at
    `num_tokens - 1` before the lookup instead, which is what this now does."""
    from pvllm.v1.core.kv_cache_manager import KVCacheManager
    from pvllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from pvllm.v1.request import Request

    block_size = 4
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=32,
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["attn.0"],
                    kv_cache_spec=FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=2,
                        head_size=32,
                        dtype="bfloat16",
                        dtype_bytes=2,
                    ),
                ),
                KVCacheGroupSpec(
                    layer_names=["mixer.0"],
                    kv_cache_spec=MambaSpec(block_size=block_size, state_bytes=4096),
                ),
            ],
        ),
        max_model_len=256,
        enable_caching=True,
    )

    def request(request_id: str, num_tokens: int) -> Request:
        made = Request(
            request_id=request_id,
            prompt_token_ids=list(range(num_tokens)),
            sampling_params=SamplingParams(max_tokens=4),
            arrival_time=0.0,
        )
        assert manager.block_hasher is not None
        made.attach_block_hasher(manager.block_hasher)
        return made

    # An exact multiple of the block size is what arms the trap: the reconciled hit
    # equals `request.num_tokens`, so the recompute-one-token rule fires.
    num_tokens = 4 * block_size
    first = request("first", num_tokens)
    assert manager.allocate_slots(first, num_new_tokens=num_tokens) is not None
    first.num_computed_tokens = num_tokens
    manager.free(first)

    blocks, num_computed = manager.get_computed_blocks(request("second", num_tokens))
    assert 0 < num_computed < num_tokens  # one token is always recomputed
    state_blocks = blocks.blocks[1]
    assert state_blocks
    # The newest entry is a real snapshot, not the shared placeholder. The prefix
    # slice left exactly the opposite: it kept the nulls and dropped the state, while
    # still telling the request those tokens were computed.
    assert not state_blocks[-1].is_null, state_blocks


def test_a_state_space_model_never_publishes_the_null_block_as_cached():
    """`cache_full_blocks` had no `is_null` guard and `_cache_full_blocks` computed
    "already cached" as a *count* of hashed blocks rather than a leading-prefix length.
    Between them the pinned null block got a real content hash and was registered in
    `cached_block_hash_to_block` -- permanently, since it can never be evicted -- so
    once the genuine block for that key went, `get_cached_block` handed later requests
    block 0 as their KV. C3 calls the hash-to-block map exact."""
    llm = LLM(**STATE_SPACE, enable_prefix_caching=True)
    try:
        pool = (
            llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
        )
        prompt = " ".join(f"w{i}" for i in range(1200))
        llm.generate([prompt], SamplingParams(max_tokens=4))
        llm.generate([prompt], SamplingParams(max_tokens=4))
        assert pool.null_block is not None
        assert pool.null_block.block_hash is None
        residents = [
            key
            for key, blocks in pool.cached_block_hash_to_block.items()
            if 0 in blocks
        ]
        assert residents == []
    finally:
        llm.shutdown()


def test_the_layer_pattern_is_the_one_the_family_publishes():
    """The card shipped `M*24 + '*'*4 + '-'*24`, a block-ordered string no model in the
    family publishes, while `model_db.py` called the field "the per-layer string these
    models publish". Nemotron-H interleaves the attention layers at 7, 18, 29 and 40 --
    which is what makes every pipeline stage hold some of each."""
    from pvllm.sim.model_db import load_model_card

    card = load_model_card("hybrid-ssm-8b")
    pattern = card.hybrid_override_pattern
    assert pattern == "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    assert [i for i, k in enumerate(pattern) if k == "*"] == [7, 18, 29, 40]
    assert card.num_mamba_layers == 24
    assert card.num_attention_layers == 4


# --- the windowed livelock, and the guard that was meant to have closed it ----


def test_a_prefix_cache_hit_does_not_livelock_a_windowed_model():
    """The pre-check subtracted only the *non-null* part of a cache hit while
    `adopt_cached_blocks` installed the whole list, so every null placeholder in a
    windowed hit was charged as a block that had to come out of the free queue. The
    result was `None` from `allocate_slots` forever, on a pool that was entirely free
    -- reached by *hitting* the cache, which is the one thing a hit must not cause."""
    llm = LLM(
        model="tiny-test",
        device_card="tiny-2gb",
        max_model_len=2048,
        block_size=16,
        max_num_batched_tokens=255,
        max_num_seqs=1,
        sliding_window=64,
        num_gpu_blocks_override=23,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    try:
        prompt = " ".join(f"w{i}" for i in range(250))
        # The first pass populates the cache; the second is the one that used to spin
        # forever, with `running=[]`, one request waiting, and the pool idle.
        assert llm.generate([prompt], SamplingParams(max_tokens=4))[0].outputs[0].text
        assert llm.generate([prompt], SamplingParams(max_tokens=4))[0].outputs[0].text
        stats = llm.llm_engine.make_stats()
        assert stats["prefix_cache_hits"] > 0
    finally:
        llm.shutdown()


def test_the_startup_guard_counts_the_reserved_null_block():
    """`windowed_blocks_for_one_request` was right, and the comparison against it was
    not: `num_gpu_blocks < blocks_for_one_request` ignored the null block that a
    windowed pool reserves. So a pool of exactly the size the refusal message names
    cleared startup while reporting `max_concurrency` below 1.0, and then hung -- the
    same class of silent hang the commit says it closed, one block wide."""
    from pvllm.sim.memory import SimOutOfMemoryError, windowed_blocks_for_one_request

    needed = windowed_blocks_for_one_request(64, 16, 2048, 255)

    def start(pool: int) -> float:
        llm = LLM(
            model="tiny-test",
            device_card="tiny-2gb",
            max_model_len=2048,
            block_size=16,
            max_num_batched_tokens=255,
            max_num_seqs=1,
            sliding_window=64,
            num_gpu_blocks_override=pool,
            disable_log_stats=True,
        )
        try:
            return _profile(llm).max_concurrency
        finally:
            llm.shutdown()

    # Exactly `needed` blocks is one short once block 0 is reserved, so it is refused.
    with pytest.raises(SimOutOfMemoryError, match="allocatable"):
        start(needed)
    # And one more starts, with a concurrency figure that is not a lie.
    assert start(needed + 1) >= 1.0


# --- blast radius -----------------------------------------------------------


def test_the_kv_spec_describes_the_pipeline_stage_that_binds():
    """`get_num_layers` returns `n // pp` -- upstream's stage 0, and upstream puts the
    remainder on the middle stages. The weights term already used the ceiling, arguing
    that the busiest stage is what must actually be buildable; the KV spec used the
    floor, so `num_gpu_blocks` promised a pool the binding stage could not allocate.
    dense-8b at pp=7 is `4,4,5,5,5,5,4`: the spec covered 4 layers, the pool had to fit
    5."""
    llm = LLM(
        model="dense-8b",
        device_card="datacenter-80gb",
        max_model_len=2048,
        block_size=16,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        pipeline_parallel_size=7,
        disable_log_stats=True,
    )
    try:
        assert len(_specs(llm)) == 5
    finally:
        llm.shutdown()


def test_every_pipeline_stage_of_a_hybrid_holds_both_kinds_of_layer():
    """With the published interleaved pattern *and* stage selection by cost, a
    state-space hybrid keeps its seven groups at every pipeline size. The block-ordered
    card sliced from the front reported `{MambaSpec: 13}` at pp=4 -- one group, zero
    attention layers, and no KV cache at all -- and started anyway."""
    from pvllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    for pp in (1, 2, 4):
        llm = LLM(**{**STATE_SPACE, "pipeline_parallel_size": pp})
        try:
            kinds = [type(s) for s in _specs(llm).values()]
            assert MambaSpec in kinds, pp
            assert FullAttentionSpec in kinds, pp
            groups = llm.llm_engine.engine_core.engine_core.kv_cache_config.num_groups
            assert groups == 7, (pp, groups)
        finally:
            llm.shutdown()


def test_the_block_size_warning_can_fire_for_the_models_whose_block_size_grows():
    """`VllmConfig.__post_init__` read `block_size` for the `max_model_len < block_size`
    warning and *then* called the platform hook that raises it from 16 to 1040. The
    condition the check exists to catch is the one the alignment creates, so it could
    never fire for the only models it applies to."""
    import logging

    from pvllm.engine.arg_utils import EngineArgs

    def build(caplog_level: int = logging.WARNING) -> tuple[object, list[str]]:
        records: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        logger = logging.getLogger("pvllm.config.vllm")
        handler = Capture(level=caplog_level)
        logger.addHandler(handler)
        try:
            config = EngineArgs(
                model="hybrid-ssm-8b",
                device_card="datacenter-80gb",
                max_model_len=1024,
                block_size=16,
                max_num_batched_tokens=1024,
            ).create_engine_config()
        finally:
            logger.removeHandler(handler)
        return config, records

    config, warnings = build()
    assert config.cache_config.block_size == 1040
    assert config.model_config.max_model_len == 1024
    assert any("smaller than block_size (1040)" in line for line in warnings), warnings
