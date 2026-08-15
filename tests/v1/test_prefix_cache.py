"""Prefix caching: hashing, lookup, sharing, eviction. R6.3--R6.5, R6.9, C3."""

from __future__ import annotations

import pytest

from pvllm.sampling_params import SamplingParams
from pvllm.v1.core.kv_cache_manager import KVCacheManager
from pvllm.v1.core.kv_cache_utils import (
    compute_none_hash,
    get_hash_fn_by_name,
    hash_block_tokens,
    make_block_hash_with_group_id,
)
from pvllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from pvllm.v1.request import Request

BLOCK_SIZE = 4


def make_manager(num_blocks: int = 32, seed: int = 0) -> KVCacheManager:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=32,
        dtype="bfloat16",
        dtype_bytes=2,
    )
    return KVCacheManager(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
        ),
        max_model_len=256,
        enable_caching=True,
        seed=seed,
    )


def make_request(
    manager: KVCacheManager,
    request_id: str,
    tokens: list[int],
    cache_salt: str | None = None,
) -> Request:
    request = Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=8),
        arrival_time=0.0,
        cache_salt=cache_salt,
    )
    assert manager.block_hasher is not None
    request.attach_block_hasher(manager.block_hasher)
    return request


def admit(manager: KVCacheManager, request: Request) -> int:
    """Do what the scheduler does on admission. Returns cached token count."""
    cached_blocks, num_cached = manager.get_computed_blocks(request)
    num_new = request.num_tokens - num_cached
    manager.allocate_slots(
        request,
        num_new,
        num_new_computed_tokens=num_cached,
        new_computed_blocks=cached_blocks,
    )
    request.num_computed_tokens = request.num_tokens
    return num_cached


# --- hashing (R6.3) --------------------------------------------------------


def test_hashes_are_chained_through_the_parent():
    """What makes this a *prefix* cache: block 3 matches only if 0-2 matched too.

    Hash a block's tokens alone and two requests sharing a middle passage but not a
    beginning would collide, and the second would read KV computed under different
    preceding context.
    """
    hash_fn = get_hash_fn_by_name("sha256")
    none_hash = compute_none_hash(hash_fn, 0)

    first = hash_block_tokens(hash_fn, None, [1, 2, 3, 4], none_hash)
    same_tokens_different_parent = hash_block_tokens(
        hash_fn, first, [1, 2, 3, 4], none_hash
    )
    assert first != same_tokens_different_parent


def test_identical_content_hashes_identically():
    hash_fn = get_hash_fn_by_name("sha256")
    none_hash = compute_none_hash(hash_fn, 0)
    assert hash_block_tokens(hash_fn, None, [1, 2], none_hash) == hash_block_tokens(
        hash_fn, None, [1, 2], none_hash
    )


def test_hashes_are_reproducible_across_runs():
    """B4. Upstream seeds this from os.urandom unless PYTHONHASHSEED is set, which
    would make every run's hashes differ and a recorded trace incomparable."""
    hash_fn = get_hash_fn_by_name("sha256")
    assert compute_none_hash(hash_fn, 7) == compute_none_hash(hash_fn, 7)
    assert compute_none_hash(hash_fn, 7) != compute_none_hash(hash_fn, 8)


def test_extra_keys_change_the_hash():
    """Cache poisoning is the failure this prevents: one tenant reading another's
    KV for the same tokens."""
    hash_fn = get_hash_fn_by_name("sha256")
    none_hash = compute_none_hash(hash_fn, 0)
    plain = hash_block_tokens(hash_fn, None, [1, 2], none_hash)
    salted = hash_block_tokens(hash_fn, None, [1, 2], none_hash, extra_keys=("tenant",))
    assert plain != salted


def test_group_id_is_part_of_the_key():
    """R6.7."""
    hash_fn = get_hash_fn_by_name("sha256")
    none_hash = compute_none_hash(hash_fn, 0)
    block_hash = hash_block_tokens(hash_fn, None, [1, 2], none_hash)
    assert make_block_hash_with_group_id(
        block_hash, 0
    ) != make_block_hash_with_group_id(block_hash, 1)


def test_only_full_blocks_are_hashed():
    """A partial tail must not be published: a later token changes its contents,
    so any request that matched it would be reading a block about to change."""
    manager = make_manager()
    request = make_request(manager, "r0", list(range(10)))  # 2 full blocks + 2 tokens
    assert len(request.block_hashes) == 2

    request.append_output_token_ids([99, 98])  # completes the third block
    assert len(request.block_hashes) == 3


def test_an_unknown_hash_algorithm_is_rejected():
    with pytest.raises(ValueError, match="unknown prefix_caching_hash_algo"):
        get_hash_fn_by_name("md5")


def test_the_builtin_algorithm_is_available():
    hash_fn = get_hash_fn_by_name("builtin")
    assert isinstance(hash_fn(("a", 1)), bytes)


# --- lookup (R6.4) ---------------------------------------------------------


def test_a_cold_cache_returns_nothing():
    manager = make_manager()
    request = make_request(manager, "r0", list(range(16)))
    blocks, num_cached = manager.get_computed_blocks(request)
    assert num_cached == 0 and blocks.num_blocks == 0


def test_a_shared_prefix_hits():
    manager = make_manager()
    prefix = list(range(16))

    first = make_request(manager, "a", [*prefix, 100, 101, 102, 103])
    assert admit(manager, first) == 0

    second = make_request(manager, "b", [*prefix, 200, 201, 202, 203])
    assert admit(manager, second) == 16  # four blocks of the shared prefix


def test_the_lookup_stops_at_the_first_miss():
    """A prefix cache is only usable contiguously: the KV for a gap does not exist,
    so a hit beyond it cannot be read."""
    manager = make_manager()
    admit(manager, make_request(manager, "a", list(range(16))))

    # Diverges in block 2, then re-converges on block 3's content.
    diverged = [0, 1, 2, 3, 4, 5, 6, 7, 99, 99, 99, 99, 12, 13, 14, 15]
    second = make_request(manager, "b", diverged)
    _, num_cached = manager.get_computed_blocks(second)
    assert num_cached == 8  # the first two blocks only


def test_an_exact_full_hit_still_recomputes_one_block():
    """R6.4. Without this the request is scheduled with zero new tokens: nothing to
    run, no logits, no sampled token, and it never progresses. Only bites on an
    exact hit, which is what makes it easy to miss."""
    manager = make_manager()
    tokens = list(range(16))
    admit(manager, make_request(manager, "a", tokens))

    identical = make_request(manager, "b", tokens)
    _, num_cached = manager.get_computed_blocks(identical)
    assert num_cached == 12  # one block held back
    assert num_cached < len(tokens)


def test_a_different_salt_does_not_hit():
    """Tenant partitioning."""
    manager = make_manager()
    tokens = list(range(16))
    admit(manager, make_request(manager, "a", tokens, cache_salt="tenant-1"))

    other_tenant = make_request(manager, "b", tokens, cache_salt="tenant-2")
    _, num_cached = manager.get_computed_blocks(other_tenant)
    assert num_cached == 0


def test_caching_disabled_never_hits():
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=32,
        dtype="bfloat16",
        dtype_bytes=2,
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=32,
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
        ),
        max_model_len=256,
        enable_caching=False,
    )
    assert manager.block_hasher is None
    request = Request(
        "r0", list(range(16)), SamplingParams(max_tokens=4), arrival_time=0.0
    )
    _, num_cached = manager.get_computed_blocks(request)
    assert num_cached == 0


# --- sharing and accounting (R6.5) -----------------------------------------


def test_a_hit_shares_blocks_rather_than_allocating_new_ones():
    """The whole point: the second request's prefix costs no new blocks."""
    manager = make_manager(num_blocks=32)
    prefix = list(range(16))

    admit(manager, make_request(manager, "a", [*prefix, 100, 101, 102, 103]))
    used_after_first = 32 - manager.block_pool.get_num_free_blocks()

    admit(manager, make_request(manager, "b", [*prefix, 200, 201, 202, 203]))
    used_after_second = 32 - manager.block_pool.get_num_free_blocks()

    # Five blocks for the first request; the second adds only its own tail.
    assert used_after_first == 5
    assert used_after_second - used_after_first == 1


def test_shared_blocks_carry_two_references():
    manager = make_manager()
    prefix = list(range(16))
    first = make_request(manager, "a", [*prefix, 100, 101, 102, 103])
    admit(manager, first)
    second = make_request(manager, "b", [*prefix, 200, 201, 202, 203])
    admit(manager, second)

    shared = manager.get_blocks("a").blocks[0][0]
    assert shared.ref_cnt == 2

    manager.free(first)
    assert shared.ref_cnt == 1
    assert manager.block_pool.get_cached_block(first.block_hashes[0], 0) is shared


def test_a_freed_prefix_stays_cached_until_evicted():
    """Blocks with a hash go to the *back* of the free queue, so a later request
    with the same prefix can still hit them."""
    manager = make_manager()
    tokens = list(range(16))
    first = make_request(manager, "a", tokens)
    admit(manager, first)
    manager.free(first)

    second = make_request(manager, "b", tokens)
    _, num_cached = manager.get_computed_blocks(second)
    assert num_cached == 12


# --- metrics (R6.9, R6.10) -------------------------------------------------


def test_queries_and_hits_are_counted():
    manager = make_manager()
    prefix = list(range(16))
    admit(manager, make_request(manager, "a", [*prefix, 100, 101, 102, 103]))
    assert manager.prefix_cache_queries == 20
    assert manager.prefix_cache_hits == 0

    admit(manager, make_request(manager, "b", [*prefix, 200, 201, 202, 203]))
    assert manager.prefix_cache_queries == 40
    assert manager.prefix_cache_hits == 16


def test_reset_clears_the_cache_and_the_counters():
    """R6.10."""
    manager = make_manager()
    tokens = list(range(16))
    request = make_request(manager, "a", tokens)
    admit(manager, request)
    manager.free(request)

    assert manager.reset_prefix_cache() is True
    assert manager.make_prefix_cache_stats() == (0, 0)

    after = make_request(manager, "b", tokens)
    _, num_cached = manager.get_computed_blocks(after)
    assert num_cached == 0


def test_reset_refuses_while_blocks_are_in_use():
    manager = make_manager()
    request = make_request(manager, "a", list(range(16)))
    admit(manager, request)
    assert manager.reset_prefix_cache() is False
