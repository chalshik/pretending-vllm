"""Prometheus metrics. R12.1, R12.2, C6.

Upstream: vllm/v1/metrics/loggers.py
Tier: B

C6 binds metric names, types, labels, and histogram bucket edges exactly, so that a
dashboard built against real vLLM renders against this without modification.

**F5, and worth stating plainly because the draft spec had it wrong:** upstream
declares counters *without* a `_total` suffix. `prometheus_client` appends it on
export. Declaring `vllm:prompt_tokens_total` therefore exports
`vllm:prompt_tokens_total_total`, and every counter panel on the dashboard goes empty.
Histograms carry no auto-suffix, which is why `vllm:iteration_tokens_total` genuinely
does end in `_total` -- an inconsistency that looks like a typo and is not.

R12.4: every duration is a *modeled* duration under a virtual clock, and that has to
be discoverable rather than implied. Each latency metric's help text says so.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Gauge, Histogram

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.v1.engine import FINISH_REASON_STRINGS
from pvllm.v1.metrics.stats import IterationStats, SchedulerStats

logger = init_logger(__name__)

#: Appended to every latency metric's help text (R12.4). A consumer reading only
#: `/metrics` must still be able to tell these were not measured.
_MODELED = (
    " NOTE: this duration is MODELED by pretending-vllm's cost model, not measured. "
    "See the fidelity contract in the README."
)

#: R12.2. Upstream's edges, verbatim.
REQUEST_LATENCY_BUCKETS = [
    0.3,
    0.5,
    0.8,
    1.0,
    1.5,
    2.0,
    2.5,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    120.0,
    240.0,
    480.0,
    960.0,
    1920.0,
    7680.0,
]
TIME_TO_FIRST_TOKEN_BUCKETS = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    640.0,
    2560.0,
]
INTER_TOKEN_LATENCY_BUCKETS = [
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    40.0,
    80.0,
]
ITERATION_TOKENS_BUCKETS = [
    1,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
]
REQUEST_N_BUCKETS = [1, 2, 5, 10, 20]


def build_buckets(mantissa: list[int], max_value: int) -> list[int]:
    """Exponential buckets over the given mantissa, up to `max_value`."""
    exponent = 0
    buckets: list[int] = []
    while True:
        for m in mantissa:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> list[int]:
    """`build_1_2_5_buckets(100) == [1, 2, 5, 10, 20, 50, 100]`."""
    return build_buckets([1, 2, 5], max_value)


class PrometheusStatLogger:
    """Publishes engine statistics in upstream's metric surface.

    Args:
        vllm_config: For the label values and the token-count bucket ranges.
        registry: Where to register. Tests pass a fresh registry so metric names do
            not collide across engine instances in one process.
    """

    def __init__(self, vllm_config: VllmConfig, registry: Any = None) -> None:
        self.vllm_config = vllm_config
        model_name = (
            vllm_config.model_config.served_model_name or vllm_config.model_config.model
        )
        labelnames = ["model_name", "engine"]
        labelvalues = [model_name, "0"]
        assert vllm_config.model_config.max_model_len is not None
        max_model_len = vllm_config.model_config.max_model_len

        kwargs: dict[str, Any] = {"registry": registry} if registry is not None else {}

        def gauge(name: str, doc: str) -> Any:
            return Gauge(name, doc, labelnames=labelnames, **kwargs).labels(
                *labelvalues
            )

        def counter(name: str, doc: str, extra: list[str] | None = None) -> Any:
            metric = Counter(name, doc, labelnames=labelnames + (extra or []), **kwargs)
            return metric if extra else metric.labels(*labelvalues)

        def histogram(name: str, doc: str, buckets: list[Any]) -> Any:
            return Histogram(
                name, doc, buckets=buckets, labelnames=labelnames, **kwargs
            ).labels(*labelvalues)

        # --- gauges ---------------------------------------------------------
        self.gauge_scheduler_running = gauge(
            "vllm:num_requests_running",
            "Number of requests in model execution batches.",
        )
        self.gauge_scheduler_waiting = gauge(
            "vllm:num_requests_waiting", "Number of requests waiting to be processed."
        )
        self.gauge_kv_cache_usage = gauge(
            "vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage."
        )

        # --- counters (exported with `_total` appended by the client) --------
        self.counter_num_preempted_reqs = counter(
            "vllm:num_preemptions", "Cumulative number of preemptions from the engine."
        )
        self.counter_prompt_tokens = counter(
            "vllm:prompt_tokens", "Number of prefill tokens processed."
        )
        self.counter_generation_tokens = counter(
            "vllm:generation_tokens", "Number of generation tokens processed."
        )
        self.counter_prefix_cache_queries = counter(
            "vllm:prefix_cache_queries",
            "Prefix cache queries, in terms of number of queried tokens.",
        )
        self.counter_prefix_cache_hits = counter(
            "vllm:prefix_cache_hits",
            "Prefix cache hits, in terms of number of cached tokens.",
        )
        # R17.2 + R18.1, C6. Upstream's names for the two caches that sit either side
        # of the local one: the KV connector's cross-instance cache, counted in
        # tokens, and the multimodal encoder cache, counted in items. Both counters
        # existed and were reported by nothing, so the two features whose whole point
        # is a hit rate had no hit rate on the surface a dashboard reads.
        self.counter_external_prefix_cache_queries = counter(
            "vllm:external_prefix_cache_queries",
            "External prefix cache queries from KV connector cross-instance cache "
            "sharing, in terms of number of queried tokens.",
        )
        self.counter_external_prefix_cache_hits = counter(
            "vllm:external_prefix_cache_hits",
            "External prefix cache hits from KV connector cross-instance cache "
            "sharing, in terms of number of cached tokens.",
        )
        self.counter_mm_cache_queries = counter(
            "vllm:mm_cache_queries",
            "Multi-modal cache queries, in terms of number of queried items.",
        )
        self.counter_mm_cache_hits = counter(
            "vllm:mm_cache_hits",
            "Multi-modal cache hits, in terms of number of cached items.",
        )
        # R14. Upstream's names, without the `_total` the client library appends
        # (F5). Acceptance *rate* is deliberately not a gauge: it is a ratio of two
        # counters, and Prometheus computes ratios from counters so a dashboard can
        # window it. A gauge would report the lifetime average forever.
        self._counter_spec_draft_tokens = counter(
            "vllm:spec_decode_num_draft_tokens",
            "Number of draft tokens proposed [MODELED acceptance]",
        )
        self._counter_spec_accepted_tokens = counter(
            "vllm:spec_decode_num_accepted_tokens",
            "Number of draft tokens accepted by verification [MODELED acceptance]",
        )

        self._counter_request_success = counter(
            "vllm:request_success",
            "Count of successfully processed requests.",
            extra=["finished_reason"],
        )
        self._request_success_labelvalues = labelvalues
        # Instantiated for every finish reason at construction, as upstream does.
        # Without this the series does not exist until the first request finishes
        # with that reason, so a dashboard panel is empty at t=0 and -- worse -- the
        # C6 conformance golden records the family with no labels at all, which made
        # renaming `finished_reason` a change the whole suite passed.
        for reason in FINISH_REASON_STRINGS:
            self._counter_request_success.labels(*labelvalues, reason)

        # --- histograms ------------------------------------------------------
        self.histogram_iteration_tokens = histogram(
            "vllm:iteration_tokens_total",
            "Histogram of number of tokens per engine_step.",
            ITERATION_TOKENS_BUCKETS,
        )
        self.histogram_time_to_first_token = histogram(
            "vllm:time_to_first_token_seconds",
            "Histogram of time to first token in seconds." + _MODELED,
            TIME_TO_FIRST_TOKEN_BUCKETS,
        )
        self.histogram_inter_token_latency = histogram(
            "vllm:inter_token_latency_seconds",
            "Histogram of inter token latency in seconds." + _MODELED,
            INTER_TOKEN_LATENCY_BUCKETS,
        )
        self.histogram_time_per_output_token = histogram(
            "vllm:request_time_per_output_token_seconds",
            "Histogram of time per output token in seconds." + _MODELED,
            INTER_TOKEN_LATENCY_BUCKETS,
        )
        self.histogram_e2e_time_request = histogram(
            "vllm:e2e_request_latency_seconds",
            "Histogram of e2e request latency in seconds." + _MODELED,
            REQUEST_LATENCY_BUCKETS,
        )
        self.histogram_queue_time_request = histogram(
            "vllm:request_queue_time_seconds",
            "Histogram of time spent in WAITING phase for request." + _MODELED,
            REQUEST_LATENCY_BUCKETS,
        )
        self.histogram_inference_time_request = histogram(
            "vllm:request_inference_time_seconds",
            "Histogram of time spent in RUNNING phase for request." + _MODELED,
            REQUEST_LATENCY_BUCKETS,
        )
        self.histogram_prefill_time_request = histogram(
            "vllm:request_prefill_time_seconds",
            "Histogram of time spent in PREFILL phase for request." + _MODELED,
            REQUEST_LATENCY_BUCKETS,
        )
        self.histogram_decode_time_request = histogram(
            "vllm:request_decode_time_seconds",
            "Histogram of time spent in DECODE phase for request." + _MODELED,
            REQUEST_LATENCY_BUCKETS,
        )
        self.histogram_num_prompt_tokens_request = histogram(
            "vllm:request_prompt_tokens",
            "Number of prefill tokens processed.",
            build_1_2_5_buckets(max_model_len),
        )
        self.histogram_num_generation_tokens_request = histogram(
            "vllm:request_generation_tokens",
            "Number of generation tokens processed.",
            build_1_2_5_buckets(max_model_len),
        )
        self.histogram_max_tokens_request = histogram(
            "vllm:request_params_max_tokens",
            "Histogram of the max_tokens request parameter.",
            build_1_2_5_buckets(max_model_len),
        )
        self.histogram_n_request = histogram(
            "vllm:request_params_n",
            "Histogram of the n request parameter.",
            REQUEST_N_BUCKETS,
        )

    # --- recording -----------------------------------------------------------

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
    ) -> None:
        if scheduler_stats is not None:
            self.gauge_scheduler_running.set(scheduler_stats.num_running_reqs)
            self.gauge_scheduler_waiting.set(scheduler_stats.num_waiting_reqs)
            self.gauge_kv_cache_usage.set(scheduler_stats.kv_cache_usage)
            # Counters are set from cumulative totals rather than incremented, so a
            # dropped step cannot make the series drift from the engine's own count.
            _set_counter(
                self.counter_num_preempted_reqs, scheduler_stats.num_preemptions
            )
            _set_counter(
                self._counter_spec_draft_tokens, scheduler_stats.num_draft_tokens
            )
            _set_counter(
                self._counter_spec_accepted_tokens,
                scheduler_stats.num_accepted_tokens,
            )
            _set_counter(
                self.counter_prefix_cache_queries, scheduler_stats.prefix_cache_queries
            )
            _set_counter(
                self.counter_prefix_cache_hits, scheduler_stats.prefix_cache_hits
            )
            _set_counter(
                self.counter_external_prefix_cache_queries,
                scheduler_stats.external_prefix_cache_queries,
            )
            _set_counter(
                self.counter_external_prefix_cache_hits,
                scheduler_stats.external_prefix_cache_hits,
            )
            _set_counter(
                self.counter_mm_cache_queries, scheduler_stats.mm_cache_queries
            )
            _set_counter(self.counter_mm_cache_hits, scheduler_stats.mm_cache_hits)

        if iteration_stats is None:
            return

        if iteration_stats.num_prompt_tokens:
            self.counter_prompt_tokens.inc(iteration_stats.num_prompt_tokens)
        if iteration_stats.num_generation_tokens:
            self.counter_generation_tokens.inc(iteration_stats.num_generation_tokens)
        if iteration_stats.total_tokens:
            self.histogram_iteration_tokens.observe(iteration_stats.total_tokens)

        for ttft in iteration_stats.time_to_first_tokens:
            self.histogram_time_to_first_token.observe(ttft)
        for itl in iteration_stats.inter_token_latencies:
            self.histogram_inter_token_latency.observe(itl)

        for finished in iteration_stats.finished_requests:
            self._counter_request_success.labels(
                *self._request_success_labelvalues, finished.finish_reason
            ).inc()
            self.histogram_e2e_time_request.observe(finished.e2e_latency)
            self.histogram_queue_time_request.observe(finished.queue_time)
            self.histogram_prefill_time_request.observe(finished.prefill_time)
            self.histogram_inference_time_request.observe(finished.inference_time)
            self.histogram_decode_time_request.observe(finished.decode_time)
            self.histogram_time_per_output_token.observe(finished.time_per_output_token)
            self.histogram_num_prompt_tokens_request.observe(finished.num_prompt_tokens)
            self.histogram_num_generation_tokens_request.observe(
                finished.num_generation_tokens
            )
            self.histogram_n_request.observe(finished.n_param)
            if finished.max_tokens_param is not None:
                self.histogram_max_tokens_request.observe(finished.max_tokens_param)


def _set_counter(counter: Any, total: float) -> None:
    """Move a counter to a cumulative total.

    `prometheus_client` counters only increment, so the delta is computed here. The
    engine's own totals are authoritative; incrementing per step would let a missed
    step desync the series permanently.
    """
    current = counter._value.get()
    if total > current:
        counter.inc(total - current)
