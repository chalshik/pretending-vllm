"""EngineArgs: the flat, user-facing surface that resolves into a VllmConfig.

Upstream: vllm/engine/arg_utils.py
Tier: C

R1.2 fixes the minimum field set. R1.4 fixes the defaults: they match upstream
wherever the field exists upstream, which is why `gpu_memory_utilization` is 0.92 and
both prefix caching and chunked prefill are on.

Simulator-only arguments are grouped under `--device-card`, `--clock-mode`, and
friends, and resolve into `SimConfig` (R1.3). They are the only arguments here with no
upstream counterpart, and `add_cli_args` puts them in their own argument group so
`--help` shows plainly which knobs are pretending.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Any

from pvllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    SchedulerConfig,
    SimConfig,
    StructuredOutputsConfig,
    VllmConfig,
)
from pvllm.config.kv_transfer import KVTransferConfig
from pvllm.config.lora import LoRAConfig
from pvllm.config.model import DEFAULT_MODEL
from pvllm.config.speculative import SpeculativeConfig
from pvllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class EngineArgs:
    """Flat arguments for the engine. R1.2."""

    # --- model ---------------------------------------------------------------
    model: str = DEFAULT_MODEL
    tokenizer: str | None = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    dtype: str = "auto"
    max_model_len: int | None = None
    revision: str | None = None
    seed: int = 0
    max_logprobs: int = 20
    served_model_name: str | None = None
    skip_tokenizer_init: bool = False
    enforce_eager: bool = False

    # --- cache ---------------------------------------------------------------
    block_size: int = 16
    gpu_memory_utilization: float = 0.92
    num_gpu_blocks_override: int | None = None
    kv_cache_dtype: str = "auto"
    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "sha256"
    #: R6.7. Attend to only the last N tokens. Bounds KV per request, so capacity
    #: stops depending on conversation length.
    sliding_window: int | None = None

    # --- scheduler -----------------------------------------------------------
    max_num_batched_tokens: int | None = None
    max_num_seqs: int = 1024
    max_num_partial_prefills: int = 1
    long_prefill_token_threshold: int = 0
    enable_chunked_prefill: bool = True
    #: R6.7. Promote every sliding-window layer to full attention, collapsing a
    #: hybrid model into one KV cache group. Upstream's flag, and the honest A/B for
    #: what hybrid attention buys -- same model, both ways.
    disable_hybrid_kv_cache_manager: bool = False
    scheduling_policy: str = "fcfs"

    # --- parallelism ---------------------------------------------------------
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1

    # --- observability -------------------------------------------------------
    disable_log_stats: bool = False

    # --- deferred subsystems -------------------------------------------------
    #: R14. Upstream accepts a dict here and resolves it; this takes the resolved
    #: object, because there is no draft model to resolve *from* and a dict would
    #: only postpone the same validation to a worse place.
    speculative_config: SpeculativeConfig | None = None
    kv_transfer_config: dict[str, Any] | None = None
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    max_cpu_loras: int | None = None
    #: R16.1. `name=path` per adapter, served under its own model name.
    lora_modules: list[str] | None = None

    # --- simulator (no upstream counterpart) ---------------------------------
    device_card: str = "datacenter-80gb"
    num_devices: int = 1
    model_card: str | None = None
    clock_mode: str = "virtual"
    time_scale: float = 1.0
    cost_model_profile: str = "constant"
    jitter_sigma: float = 0.0
    #: R14. Simulator knob; see SimConfig.spec_acceptance_rate.
    spec_acceptance_rate: float = 0.7
    output_length_policy: str = "from_request"
    output_length_fixed: int = 128
    content_policy: str = "pseudoword"
    trace_path: str | None = None

    def create_engine_config(self) -> VllmConfig:
        """Resolve into the composite config the engine actually consumes."""
        sim_config = SimConfig(
            device_card=self.device_card,
            num_devices=self.num_devices,
            clock_mode=self.clock_mode,  # type: ignore[arg-type]
            time_scale=self.time_scale,
            cost_model_profile=self.cost_model_profile,  # type: ignore[arg-type]
            jitter_sigma=self.jitter_sigma,
            spec_acceptance_rate=self.spec_acceptance_rate,
            model_card=self.model_card,
            output_length_policy=self.output_length_policy,  # type: ignore[arg-type]
            output_length_fixed=self.output_length_fixed,
            content_policy=self.content_policy,  # type: ignore[arg-type]
            # R19.2: one seed drives arrivals, output lengths, tokens, and jitter.
            seed=self.seed,
            trace_path=self.trace_path,
        )

        model_config = ModelConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            max_model_len=self.max_model_len,
            enforce_eager=self.enforce_eager,
            max_logprobs=self.max_logprobs,
            skip_tokenizer_init=self.skip_tokenizer_init,
            served_model_name=self.served_model_name,
            model_card=self.model_card,
        )

        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            cache_dtype=self.kv_cache_dtype,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            enable_prefix_caching=self.enable_prefix_caching,
            prefix_caching_hash_algo=self.prefix_caching_hash_algo,
            sliding_window=self.sliding_window,
        )

        parallel_config = ParallelConfig(
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            data_parallel_size=self.data_parallel_size,
        )

        assert model_config.max_model_len is not None
        scheduler_config = SchedulerConfig(
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
            max_num_partial_prefills=self.max_num_partial_prefills,
            policy=self.scheduling_policy,  # type: ignore[arg-type]
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
        )

        # R16.1. Built only when asked for: `None` is what tells the scheduler and
        # the memory model to skip the adapter paths entirely.
        lora_config = (
            LoRAConfig(
                max_loras=self.max_loras,
                max_lora_rank=self.max_lora_rank,
                max_cpu_loras=self.max_cpu_loras,
            )
            if self.enable_lora
            else None
        )
        if self.lora_modules and not self.enable_lora:
            raise ValueError(
                "--lora-modules was given without --enable-lora. Serving an adapter "
                "changes both the memory budget and the admission constraint, so it "
                "is not inferred from the presence of a module."
            )

        # R17. Built only when asked for; `None` is what keeps the connector path
        # off the scheduler's hot loop entirely.
        kv_transfer = (
            KVTransferConfig(**self.kv_transfer_config)
            if self.kv_transfer_config is not None
            else None
        )

        return VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=DeviceConfig(sim_config=sim_config),
            observability_config=ObservabilityConfig(
                disable_log_stats=self.disable_log_stats
            ),
            structured_outputs_config=StructuredOutputsConfig(),
            lora_config=lora_config,
            lora_modules=self.lora_modules,
            speculative_config=self.speculative_config,
            kv_transfer_config=kv_transfer,
        )

    # --- CLI -----------------------------------------------------------------

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        model = parser.add_argument_group("model")
        model.add_argument("--model", default=DEFAULT_MODEL)
        model.add_argument("--tokenizer", default=None)
        model.add_argument(
            "--tokenizer-mode", default="auto", choices=["auto", "mock", "slow"]
        )
        model.add_argument("--dtype", default="auto")
        model.add_argument("--max-model-len", type=int, default=None)
        model.add_argument("--max-logprobs", type=int, default=20)
        model.add_argument("--served-model-name", default=None)
        model.add_argument(
            "--seed", type=int, default=0, help="reproduces an entire run"
        )
        model.add_argument("--trust-remote-code", action="store_true")
        model.add_argument("--enforce-eager", action="store_true")

        cache = parser.add_argument_group("cache")
        cache.add_argument("--block-size", type=int, default=16)
        cache.add_argument("--gpu-memory-utilization", type=float, default=0.92)
        cache.add_argument("--num-gpu-blocks-override", type=int, default=None)
        cache.add_argument("--kv-cache-dtype", default="auto")
        cache.add_argument("--sliding-window", type=int, default=None)
        cache.add_argument(
            "--no-enable-prefix-caching",
            dest="enable_prefix_caching",
            action="store_false",
        )
        cache.add_argument(
            "--prefix-caching-hash-algo",
            default="sha256",
            choices=["sha256", "builtin"],
        )

        sched = parser.add_argument_group("scheduler")
        sched.add_argument("--max-num-batched-tokens", type=int, default=None)
        sched.add_argument("--max-num-seqs", type=int, default=1024)
        sched.add_argument("--max-num-partial-prefills", type=int, default=1)
        sched.add_argument("--long-prefill-token-threshold", type=int, default=0)
        sched.add_argument(
            "--no-enable-chunked-prefill",
            dest="enable_chunked_prefill",
            action="store_false",
        )
        sched.add_argument(
            "--scheduling-policy", default="fcfs", choices=["fcfs", "priority"]
        )

        par = parser.add_argument_group("parallelism")
        par.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
        par.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1)
        par.add_argument("--data-parallel-size", "-dp", type=int, default=1)

        obs = parser.add_argument_group("observability")
        obs.add_argument("--disable-log-stats", action="store_true")

        # Registered so that asking for a deferred subsystem gets the actionable
        # refusal from create_engine_config rather than an argparse "unrecognized
        # arguments" error that says nothing about why.
        lora = parser.add_argument_group("lora")
        lora.add_argument("--enable-lora", action="store_true")
        lora.add_argument("--max-loras", type=int, default=1)
        lora.add_argument("--max-lora-rank", type=int, default=16)
        lora.add_argument("--max-cpu-loras", type=int, default=None)
        lora.add_argument(
            "--lora-modules",
            nargs="+",
            default=None,
            metavar="NAME=PATH",
            help=(
                "adapters to serve, each under its own model name. A request naming "
                "one is routed to it, exactly as upstream routes them."
            ),
        )

        sim = parser.add_argument_group(
            "simulator",
            "Knobs with no upstream counterpart. These configure the parts that "
            "are pretending: the device, the clock, and the token generator.",
        )
        sim.add_argument("--device-card", default="datacenter-80gb")
        sim.add_argument("--num-devices", type=int, default=1)
        sim.add_argument(
            "--model-card", default=None, help="override the architecture lookup"
        )
        sim.add_argument(
            "--clock-mode", default="virtual", choices=["virtual", "real", "scaled"]
        )
        sim.add_argument("--time-scale", type=float, default=1.0)
        sim.add_argument(
            "--cost-model-profile", default="constant", choices=["constant", "roofline"]
        )
        sim.add_argument("--jitter-sigma", type=float, default=0.0)
        sim.add_argument("--spec-acceptance-rate", type=float, default=0.7)
        sim.add_argument(
            "--output-length-policy",
            default="from_request",
            choices=["fixed", "uniform", "lognormal", "from_request", "from_fixture"],
        )
        sim.add_argument("--output-length-fixed", type=int, default=128)
        sim.add_argument(
            "--content-policy",
            default="pseudoword",
            choices=["pseudoword", "echo", "fixture"],
        )
        sim.add_argument(
            "--trace-path", default=None, help="JSONL event trace destination"
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> EngineArgs:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in vars(args).items() if k in names})


@dataclass
class AsyncEngineArgs(EngineArgs):
    """EngineArgs for the async engine.

    Upstream adds `disable_log_requests` here. The multiprocess engine core is
    selected by `PVLLM_ENABLE_V1_MULTIPROCESSING` rather than by an argument,
    mirroring upstream's `VLLM_ENABLE_V1_MULTIPROCESSING` -- but defaulting off,
    because it trades away B4's determinism (see pvllm/envs.py).
    """

    disable_log_requests: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        EngineArgs.add_cli_args(parser)
        parser.add_argument("--disable-log-requests", action="store_true")
        return parser
