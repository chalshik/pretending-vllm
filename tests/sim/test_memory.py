"""Analytic memory sizing. R10.1--R10.6."""

from __future__ import annotations

import pytest

from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.memory import (
    MemoryLedger,
    SimOutOfMemoryError,
    compute_memory_profile,
    compute_weight_bytes,
)
from pvllm.sim.model_db import load_model_card


def profile_for(model="dense-8b", device="datacenter-80gb", **overrides):
    kwargs = {
        "dtype": "bfloat16",
        "kv_cache_dtype": None,
        "block_size": 16,
        "gpu_memory_utilization": 0.92,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
    }
    kwargs.update(overrides)
    return compute_memory_profile(
        load_model_card(model), load_device_card(device), **kwargs
    )


# --- the exact terms -------------------------------------------------------


def test_weight_bytes_is_parameters_times_dtype():
    model = load_model_card("dense-8b")
    assert compute_weight_bytes(model, "bfloat16") == model.num_parameters * 2
    assert compute_weight_bytes(model, "float32") == model.num_parameters * 4


def test_tensor_parallelism_does_not_shard_embeddings():
    """Dividing everything by TP is the common shortcut, and it understates
    per-device memory on large-vocabulary models -- a 128k-vocab model puts over a
    gigabyte in embeddings alone."""
    model = load_model_card("dense-8b")
    whole = compute_weight_bytes(model, "bfloat16", tp_size=1)
    sharded = compute_weight_bytes(model, "bfloat16", tp_size=8)
    assert sharded > whole // 8, "embedding tables are replicated, not sharded"
    assert sharded < whole


def test_kv_bytes_per_block_scales_with_block_size():
    small = profile_for(block_size=16)
    large = profile_for(block_size=32)
    assert large.kv_bytes_per_block == 2 * small.kv_bytes_per_block
    assert small.kv_bytes_per_token == large.kv_bytes_per_token


def test_num_gpu_blocks_is_the_pool_divided_by_block_size():
    profile = profile_for()
    assert profile.num_gpu_blocks == profile.kv_pool_bytes // profile.kv_bytes_per_block


def test_max_concurrency_is_blocks_of_context_over_context():
    profile = profile_for(max_model_len=8192)
    expected = profile.num_gpu_blocks * 16 / 8192
    assert profile.max_concurrency == pytest.approx(expected)


def test_utilization_bounds_the_budget():
    tight = profile_for(gpu_memory_utilization=0.5)
    loose = profile_for(gpu_memory_utilization=0.95)
    assert tight.usable_bytes < loose.usable_bytes
    assert tight.num_gpu_blocks < loose.num_gpu_blocks


def test_a_bigger_device_fits_more_blocks():
    assert (
        profile_for(device="workstation-24gb").num_gpu_blocks
        < profile_for(device="datacenter-80gb").num_gpu_blocks
    )


def test_an_override_bypasses_the_derivation():
    profile = profile_for(num_gpu_blocks_override=1234)
    assert profile.num_gpu_blocks == 1234


# --- the modeled term ------------------------------------------------------


def test_activation_peak_is_flagged_as_modeled():
    """Upstream measures this with a real profiling run; there is nothing here to
    measure. The flag exists so anything reporting these numbers can say so."""
    profile = profile_for()
    assert profile.activation_is_modeled is True
    assert "modeled" in profile.summary()


def test_activation_peak_grows_with_batch_and_vocabulary():
    """The logits buffer scales with vocabulary rather than hidden size, and is
    frequently the largest single activation."""
    small_batch = profile_for(max_num_seqs=8)
    large_batch = profile_for(max_num_seqs=512)
    assert large_batch.activation_peak_bytes > small_batch.activation_peak_bytes


# --- startup failures ------------------------------------------------------


def test_a_model_that_does_not_fit_fails_at_startup():
    """R10.5, with an actionable message rather than a bare OOM."""
    with pytest.raises(SimOutOfMemoryError) as excinfo:
        profile_for(model="dense-70b", device="workstation-24gb")
    message = str(excinfo.value)
    assert "No memory left for the KV cache" in message
    assert "gpu_memory_utilization" in message


def test_a_max_model_len_that_can_never_be_served_fails_at_startup():
    """R10.6. Left to request time this looks like a request queueing forever for
    capacity that will never exist."""
    with pytest.raises(SimOutOfMemoryError, match="No request could ever be served"):
        profile_for(
            model="dense-8b",
            device="workstation-24gb",
            max_model_len=131072,
            num_gpu_blocks_override=16,
        )


def test_a_tiny_device_still_serves_a_tiny_model():
    """The pairing tests use to force preemption without a large workload."""
    profile = profile_for(
        model="tiny-test", device="tiny-2gb", max_model_len=512, max_num_seqs=8
    )
    assert profile.num_gpu_blocks > 0


# --- the ledger ------------------------------------------------------------


def test_ledger_tracks_named_pools():
    ledger = MemoryLedger(1000)
    ledger.allocate("weights", 400)
    ledger.allocate("kv_cache", 300)
    assert ledger.allocated_bytes == 700
    assert ledger.free_bytes == 300
    assert ledger.pools() == {"kv_cache": 300, "weights": 400}


def test_ledger_refuses_to_overcommit():
    """The ledger is what makes a capacity answer trustworthy; without the refusal
    the simulator would quietly pretend the device is bigger than its card says."""
    ledger = MemoryLedger(1000)
    ledger.allocate("weights", 900)
    with pytest.raises(SimOutOfMemoryError, match="out of memory"):
        ledger.allocate("kv_cache", 200)


def test_ledger_oom_message_names_the_pools():
    ledger = MemoryLedger(1 << 30)
    ledger.allocate("weights", 1 << 29)
    with pytest.raises(SimOutOfMemoryError) as excinfo:
        ledger.allocate("kv_cache", 1 << 30)
    assert "weights" in str(excinfo.value)


def test_freeing_a_pool_returns_its_bytes():
    ledger = MemoryLedger(1000)
    ledger.allocate("scratch", 500)
    ledger.free("scratch")
    assert ledger.free_bytes == 1000
