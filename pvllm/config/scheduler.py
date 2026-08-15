"""Scheduler configuration.

Upstream: vllm/config/scheduler.py
Tier: C

R1.5: the derived defaults here are load-bearing. `max_num_batched_tokens` decides how
much prefill fits in a step, which decides the whole shape of a trace -- so the
derivation reproduces upstream's intent rather than picking a round number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pvllm.logger import init_logger

logger = init_logger(__name__)

SchedulerPolicy = Literal["fcfs", "priority"]

DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_MAX_NUM_SEQS = 1024
#: Upstream uses a smaller default budget when chunked prefill is off, because a step
#: must then hold a whole prompt.
DEFAULT_MAX_NUM_BATCHED_TOKENS_NO_CHUNKING = 2048


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    max_num_batched_tokens: int | None = None
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    max_model_len: int = 8192
    # R1.4: enabled by default upstream.
    enable_chunked_prefill: bool = True
    long_prefill_token_threshold: int = 0
    max_num_partial_prefills: int = 1
    policy: SchedulerPolicy = "fcfs"
    #: Fraction of the block pool held back from allocation.
    watermark: float = 0.0
    #: Emit a stats log line and metrics update every N steps.
    stream_interval: int = 1
    scheduler_cls: str | None = None
    async_scheduling: bool = False

    #: Encoder budget (R5.2). Zero until multimodal lands in M4.
    #: R18.1. Encoder tokens one step may process, and how many embeddings stay
    #: resident. Both default to the token budget, as upstream does: an image that
    #: cannot fit the step budget could never be scheduled at all.
    max_num_encoder_input_tokens: int = field(init=False, default=0)
    encoder_cache_size: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = (
                DEFAULT_MAX_NUM_BATCHED_TOKENS
                if self.enable_chunked_prefill
                else DEFAULT_MAX_NUM_BATCHED_TOKENS_NO_CHUNKING
            )

        # R18.1. Sized from the token budget rather than configured separately:
        # upstream does the same, and an encoder budget larger than the step budget
        # would schedule an image whose placeholders cannot fit beside it.
        if not self.max_num_encoder_input_tokens:
            self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        if not self.encoder_cache_size:
            self.encoder_cache_size = self.max_num_batched_tokens

        if self.max_num_batched_tokens < 1:
            raise ValueError(
                f"max_num_batched_tokens must be at least 1, got "
                f"{self.max_num_batched_tokens}"
            )
        if self.max_num_seqs < 1:
            raise ValueError(
                f"max_num_seqs must be at least 1, got {self.max_num_seqs}"
            )
        if not 0.0 <= self.watermark < 1.0:
            raise ValueError(f"watermark must be in [0, 1), got {self.watermark}")
        if self.policy not in ("fcfs", "priority"):
            raise ValueError(
                f"unsupported scheduling policy {self.policy!r}; expected 'fcfs' or "
                f"'priority'"
            )

        # Without chunked prefill a prompt must fit in a single step's budget, so a
        # budget below max_model_len makes long prompts unschedulable forever. Upstream
        # raises the budget rather than letting a request wedge the queue.
        if (
            not self.enable_chunked_prefill
            and self.max_num_batched_tokens < self.max_model_len
        ):
            logger.info(
                "Chunked prefill is disabled, so a prompt must fit in one step. "
                "Raising max_num_batched_tokens from %d to max_model_len (%d).",
                self.max_num_batched_tokens,
                self.max_model_len,
            )
            self.max_num_batched_tokens = self.max_model_len

        if self.max_num_partial_prefills < 1:
            raise ValueError(
                f"max_num_partial_prefills must be at least 1, got "
                f"{self.max_num_partial_prefills}"
            )
        if self.long_prefill_token_threshold < 0:
            raise ValueError(
                f"long_prefill_token_threshold must be non-negative, got "
                f"{self.long_prefill_token_threshold}"
            )
        if self.async_scheduling:
            raise NotImplementedError(
                "async scheduling (vllm/v1/core/sched/async_scheduler.py) is not "
                "ported; it needs more than one batch in flight, which lands in M4"
            )
        if self.scheduler_cls is not None:
            raise NotImplementedError(
                "a custom scheduler_cls is not supported; the scheduler is the "
                "component this project exists to reproduce faithfully"
            )
