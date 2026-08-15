"""Configuration resolution and validation. R1.1--R1.5."""

from __future__ import annotations

import argparse

import pytest

from pvllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SimConfig,
    VllmConfig,
)
from pvllm.config.scheduler import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_BATCHED_TOKENS_NO_CHUNKING,
)
from pvllm.engine.arg_utils import EngineArgs


def test_vllm_config_composes_and_platform_fills_worker_cls():
    """R1.1 plus B2: the platform gets the last word on the resolved config."""
    config = VllmConfig()
    assert config.parallel_config.worker_cls == "pvllm.v1.worker.sim_worker.Worker"
    assert config.scheduler_config is not None
    assert config.sim_config is config.device_config.sim_config


def test_upstream_defaults_are_preserved():
    """R1.4: defaults match upstream wherever the field exists upstream."""
    config = VllmConfig()
    assert config.cache_config.gpu_memory_utilization == 0.92
    assert config.cache_config.enable_prefix_caching is True
    assert config.cache_config.prefix_caching_hash_algo == "sha256"
    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.policy == "fcfs"
    assert config.model_config.max_logprobs == 20


def test_model_card_resolves_through_an_alias():
    config = ModelConfig(model="meta-llama/Llama-3.1-8B-Instruct")
    assert config.hf_config.name == "dense-8b"
    assert config.get_num_layers() == 32
    assert config.get_num_kv_heads() == 8
    assert config.get_head_size() == 128


def test_unknown_model_is_an_error_not_a_guess():
    """Inventing an architecture would make memory and latency numbers fiction."""
    with pytest.raises(FileNotFoundError, match="no model card"):
        ModelConfig(model="acme/imaginary-13b")


def test_max_model_len_defaults_to_the_architecture_limit():
    config = ModelConfig(model="dense-8b")
    assert config.max_model_len == 131072


def test_max_model_len_beyond_the_architecture_is_rejected():
    """R1.5: reproduces upstream's error intent."""
    with pytest.raises(ValueError, match="larger than the maximum"):
        ModelConfig(model="dense-8b", max_model_len=999_999)


def test_batched_token_budget_default_depends_on_chunked_prefill():
    """R1.5. Without chunking a step must hold a whole prompt, so the budget differs."""
    chunked = SchedulerConfig(max_model_len=1024)
    assert chunked.max_num_batched_tokens == DEFAULT_MAX_NUM_BATCHED_TOKENS

    unchunked = SchedulerConfig(enable_chunked_prefill=False, max_model_len=1024)
    assert (
        unchunked.max_num_batched_tokens == DEFAULT_MAX_NUM_BATCHED_TOKENS_NO_CHUNKING
    )


def test_unchunked_budget_is_raised_to_fit_the_longest_prompt():
    """Otherwise a prompt longer than the budget can never be scheduled, and the
    request wedges the queue forever instead of failing."""
    config = SchedulerConfig(enable_chunked_prefill=False, max_model_len=16384)
    assert config.max_num_batched_tokens == 16384


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: CacheConfig(gpu_memory_utilization=1.5), "gpu_memory_utilization"),
        (lambda: CacheConfig(gpu_memory_utilization=0.0), "gpu_memory_utilization"),
        (lambda: CacheConfig(block_size=0), "block_size"),
        (
            lambda: CacheConfig(prefix_caching_hash_algo="md5"),
            "prefix_caching_hash_algo",
        ),
        (lambda: SchedulerConfig(policy="lifo"), "scheduling policy"),
        (lambda: SchedulerConfig(max_num_seqs=0), "max_num_seqs"),
        (lambda: SchedulerConfig(watermark=1.0), "watermark"),
        (lambda: ModelConfig(model="dense-8b", dtype="int4"), "unsupported dtype"),
        (lambda: SimConfig(time_scale=0.0), "time_scale"),
        (lambda: SimConfig(jitter_sigma=-1.0), "jitter_sigma"),
        (lambda: SimConfig(clock_mode="wallclock"), "clock_mode"),
    ],
)
def test_invalid_values_are_rejected(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ParallelConfig(data_parallel_size=2), "data parallel"),
        (lambda: ParallelConfig(enable_expert_parallel=True), "expert parallelism"),
        (lambda: CacheConfig(sliding_window=4096), "sliding-window"),
        (lambda: SchedulerConfig(async_scheduling=True), "async scheduling"),
    ],
)
def test_deferred_subsystems_refuse_rather_than_silently_degrade(factory, match):
    """The unsupported-path discipline.

    Accepting `tensor_parallel_size=2` and then reporting single-device memory would
    answer a capacity question wrongly while looking like it worked -- the precise
    failure this project exists to avoid.
    """
    with pytest.raises(NotImplementedError, match=match):
        factory()


def test_tensor_and_pipeline_parallelism_are_accepted():
    """R13.1/R13.2. Both shard per-device memory and both are modeled, so they are
    accepted rather than refused -- the refusal existed only while accepting them
    would have reported single-device capacity."""
    config = ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=2)
    assert config.world_size == 8


def test_engine_args_round_trip_through_the_cli():
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--max-model-len",
            "16384",
            "--block-size",
            "32",
            "--device-card",
            "tiny-2gb",
            "--clock-mode",
            "scaled",
            "--time-scale",
            "50",
            "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill",
            "--seed",
            "7",
        ]
    )
    config = EngineArgs.from_cli_args(args).create_engine_config()

    assert config.model_config.hf_config.name == "dense-8b"
    assert config.model_config.max_model_len == 16384
    assert config.cache_config.block_size == 32
    assert config.cache_config.enable_prefix_caching is False
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.sim_config.device_card == "tiny-2gb"
    assert config.sim_config.clock_mode == "scaled"
    assert config.sim_config.time_scale == 50.0


def test_one_seed_reaches_both_the_model_and_the_simulator():
    """R19.2: a single seed reproduces the entire run."""
    config = EngineArgs(seed=1234).create_engine_config()
    assert config.model_config.seed == 1234
    assert config.sim_config.seed == 1234


def test_scheduler_max_model_len_follows_the_model():
    """A scheduler budget computed against a different length than the model enforces
    would let requests be admitted that can never fit."""
    config = EngineArgs(model="dense-8b", max_model_len=4096).create_engine_config()
    assert config.scheduler_config.max_model_len == 4096


def test_configuring_the_device_card_updates_what_the_platform_reports():
    from pvllm.platforms.sim import SimPlatform

    EngineArgs(device_card="tiny-2gb").create_engine_config()
    assert SimPlatform.get_device_name() == "tiny-2gb"


def test_use_v2_model_runner_is_true_by_default():
    """F1/D6: V2 is upstream's default at the pin, and the only shape mirrored here."""
    assert VllmConfig().use_v2_model_runner is True


def test_config_repr_names_what_is_pretending():
    text = str(VllmConfig())
    for field in ("card=", "device=", "clock=", "cost_model=", "seed="):
        assert field in text
