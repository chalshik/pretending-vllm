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
    #: R6.7. Upstream's escape hatch: promote every sliding-window layer to full
    #: attention, giving up the memory saving and keeping one KV cache group. Worth
    #: more here than compatibility -- the two runs side by side *are* the capacity
    #: argument for hybrid attention, on one model rather than two.
    disable_hybrid_kv_cache_manager: bool = False

    #: R18.1. Encoder tokens one step may process, and how many embeddings stay
    #: resident. Both default to the token budget, floored at the largest single
    #: multimodal item -- see `__post_init__` for why the floor is not optional.
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
        #
        # Floored at the largest item, also as upstream does
        # (`compute_mm_encoder_budget`). Without the floor an ordinary
        # `--max-num-batched-tokens 128` sizes the encoder cache below a single
        # 256-embedding image, and `can_allocate` then answers "no" on every step
        # for the rest of time: the request is admitted, trimmed to zero tokens,
        # and retried forever with no error and no log line.
        from pvllm.multimodal.inputs import MAX_TOKENS_PER_MM_ITEM

        if not self.max_num_encoder_input_tokens:
            self.max_num_encoder_input_tokens = max(
                self.max_num_batched_tokens, MAX_TOKENS_PER_MM_ITEM
            )
        if not self.encoder_cache_size:
            self.encoder_cache_size = max(
                self.max_num_batched_tokens, MAX_TOKENS_PER_MM_ITEM
            )

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
            # Refused on purpose, and the reason is not "unfinished". Upstream's
            # `AsyncScheduler` overlaps the scheduler's *Python* time with the GPU's
            # compute by committing to the next batch before the current one's output
            # lands, carrying `num_output_placeholders` until it does. The thing it
            # hides is CPU time -- and this engine charges none: the clock advances
            # only for modeled device work (`SimDevice.execute`), never for
            # scheduling.
            #
            # So porting it would change the C1 decision sequence and change latency
            # by exactly zero. A capacity study comparing `--async-scheduling` on
            # against off would see the step counts move and the throughput sit
            # still, and conclude the flag buys nothing -- when on real hardware it
            # is a throughput optimization. Refusing is a better answer than a
            # confidently wrong one, and this joins `spec_acceptance_rate` and
            # grammar conformance on the short list of things a simulator without a
            # model cannot derive.
            raise NotImplementedError(
                "async scheduling (vllm/v1/core/sched/async_scheduler.py) is "
                "deliberately not modeled. It exists to hide the scheduler's CPU "
                "time behind the forward pass, and this engine charges no CPU time "
                "at all -- the clock advances only for modeled device work. Porting "
                "it would move the scheduling decisions without moving any latency, "
                "so a comparison of the flag on and off would report that it buys "
                "nothing. Measure it on real vLLM; everything downstream of the "
                "decisions it makes is faithful here."
            )
        if self.scheduler_cls is not None:
            raise NotImplementedError(
                "a custom scheduler_cls is not supported; the scheduler is the "
                "component this project exists to reproduce faithfully"
            )
