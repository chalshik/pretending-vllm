"""Statistics collected per step and per request.

Upstream: vllm/v1/metrics/stats.py
Tier: B

Every duration here comes from the engine core's clock, so under a virtual clock they
are modeled rather than measured -- a fact R12.4 requires be discoverable, which the
Prometheus help strings carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchedulerStats:
    """A snapshot of the scheduler after one step."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    kv_cache_usage: float = 0.0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    num_preemptions: int = 0
    step_index: int = 0
    #: R14. Cumulative speculative decoding counters. Zero without speculation.
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0


@dataclass
class RequestStateStats:
    """Per-request timing, accumulated as it moves through the engine."""

    arrival_time: float = 0.0
    #: When the first output token was produced. Zero until it is.
    first_token_time: float = 0.0
    last_token_time: float = 0.0
    #: When the request was first scheduled -- the end of its queue wait.
    first_scheduled_time: float = 0.0
    num_generation_tokens: int = 0


@dataclass
class FinishedRequestStats:
    """What a completed request contributes to the histograms."""

    finish_reason: str
    e2e_latency: float = 0.0
    queue_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0
    time_to_first_token: float = 0.0
    #: Mean seconds per output token after the first.
    time_per_output_token: float = 0.0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    num_cached_tokens: int = 0
    max_tokens_param: int | None = None
    n_param: int = 1


@dataclass
class IterationStats:
    """What one engine step contributes."""

    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    num_preempted_reqs: int = 0
    #: Seconds between an output token and the previous one, per request.
    inter_token_latencies: list[float] = field(default_factory=list)
    time_to_first_tokens: list[float] = field(default_factory=list)
    finished_requests: list[FinishedRequestStats] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_generation_tokens
