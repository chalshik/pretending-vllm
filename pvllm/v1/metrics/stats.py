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
    #: R18.1. Encoder cache lookups, in items rather than tokens -- upstream's
    #: `vllm:mm_cache_*` counts the same way. Zero without multimodal traffic.
    mm_cache_queries: int = 0
    mm_cache_hits: int = 0
    #: R17.2. KV connector lookups, in *tokens*, as upstream's
    #: `vllm:external_prefix_cache_*` counts. Zero without a connector.
    external_prefix_cache_queries: int = 0
    external_prefix_cache_hits: int = 0


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
    #: R11.7. Whether *this* record should contribute the `n` observation. Every child
    #: of an `n > 1` request produces a record (upstream counts them all), but only one
    #: of them reports the `n`, or `vllm:request_params_n` would see `n` observations
    #: of the same value instead of one.
    observe_n_param: bool = True


@dataclass
class IterationStats:
    """What one engine step contributes."""

    #: Prompt tokens *prefilled* this step, cached ones included. Upstream counts
    #: the same way -- `PromptTokenStats.total` accumulates the whole prompt length,
    #: not just the part that missed the prefix cache -- and `vllm:prompt_tokens_total`
    #: is fed from it.
    num_prompt_tokens: int = 0
    #: Of `num_prompt_tokens`, the ones the prefix cache supplied, so no compute
    #: happened for them. Split out for `total_tokens`; upstream splits the same way
    #: (`PromptTokenStats.cached_tokens` against `.computed`).
    num_cached_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    num_preempted_reqs: int = 0
    #: Seconds between an output token and the previous one, per request.
    inter_token_latencies: list[float] = field(default_factory=list)
    time_to_first_tokens: list[float] = field(default_factory=list)
    finished_requests: list[FinishedRequestStats] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """What `vllm:iteration_tokens_total` observes: the step's real token work.

        Cached prompt tokens are excluded, because no compute happened for them --
        upstream observes `prompt_token_stats.computed + num_generation_tokens`.
        Counting a prefix cache hit here would report a batch the step never ran,
        and the histogram exists to show how full the batches actually are.
        """
        computed_prompt_tokens = self.num_prompt_tokens - self.num_cached_prompt_tokens
        return computed_prompt_tokens + self.num_generation_tokens
