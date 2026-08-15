"""Driving the engine for a benchmark. R20.

Upstream: vllm/benchmarks/lib/utils.py (counterpart role)
Tier: B

Upstream's benchmarks call `LLM.generate` and wrap it in `time.perf_counter()`. That
cannot work here for two reasons, and both are informative.

**Wall clock measures the wrong thing.** Under a virtual clock the engine models
durations without spending them, so `perf_counter` around a run measures how fast
Python simulates a scheduler. The purity lint makes this a compile-time fact rather
than a discipline: `time.perf_counter` is unreachable outside `pvllm/sim/`, so the
only clock available here is the engine's.

**Arrivals have to happen in modeled time.** A benchmark at 5 requests/second is
measuring queueing, and queueing only exists if request seven actually arrives after
request six has been running a while. Sleeping for real would make a benchmark take as
long as the workload it models -- an hour of traffic would take an hour. So this loop
submits a request when the *engine clock* reaches its arrival time. An hour of modeled
traffic runs in a second, and the queueing is real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.v1.metrics.stats import FinishedRequestStats


@dataclass
class BenchRequest:
    """One request in a benchmark workload."""

    prompt_token_ids: list[int]
    max_tokens: int
    #: Seconds after the run starts, in modeled time. `0.0` means the request is
    #: submitted before the first step, which is the offline batch case.
    arrival: float = 0.0


@dataclass
class RunResult:
    finished: list[FinishedRequestStats]
    #: Modeled seconds from the first submission to the last completion.
    duration: float
    num_steps: int
    num_preemptions: int
    prefix_cache_hit_rate: float


def run_workload(engine: LLM, requests: list[BenchRequest]) -> RunResult:
    """Submit a workload at its arrival times and drain it.

    Requests are submitted in arrival order and the engine is stepped until nothing
    is outstanding. Timing comes from `FinishedRequestStats`, which the engine stamps
    from its own clock -- so these numbers and a `/metrics` scrape of the same run
    agree by construction rather than by coincidence.
    """
    llm_engine = engine.llm_engine
    core = llm_engine.engine_core
    start = core.clock_time  # type: ignore[attr-defined]

    pending = sorted(enumerate(requests), key=lambda item: item[1].arrival)
    next_index = 0
    finished: list[FinishedRequestStats] = []

    def submit_due() -> None:
        nonlocal next_index
        elapsed = core.clock_time - start  # type: ignore[attr-defined]
        while next_index < len(pending):
            index, request = pending[next_index]
            if request.arrival > elapsed:
                break
            llm_engine.add_request(
                str(index),
                list(request.prompt_token_ids),
                SamplingParams(max_tokens=request.max_tokens),
            )
            next_index += 1

    submit_due()
    while next_index < len(pending) or llm_engine.has_unfinished_requests():
        if not llm_engine.has_unfinished_requests():
            # Every submitted request has drained but the workload is not over: the
            # next arrival is still in the future. Jump the clock to it rather than
            # spinning -- there is nothing for the engine to do in between, and
            # stepping an empty engine would advance modeled time by a step's
            # duration and invent idle work that never happened.
            _, request = pending[next_index]
            core.engine_core.clock.advance(  # type: ignore[attr-defined]
                max(0.0, request.arrival - (core.clock_time - start))  # type: ignore[attr-defined]
            )
            submit_due()
            continue

        llm_engine.step()
        finished.extend(llm_engine.last_iteration_stats.finished_requests)
        submit_due()

    duration = core.clock_time - start  # type: ignore[attr-defined]
    stats: dict[str, Any] = llm_engine.make_stats()
    queries = stats.get("prefix_cache_queries", 0)
    return RunResult(
        finished=finished,
        duration=duration,
        num_steps=stats.get("step_index", 0),
        num_preemptions=stats.get("num_preemptions", 0),
        prefix_cache_hit_rate=(
            stats.get("prefix_cache_hits", 0) / queries if queries else 0.0
        ),
    )


def synthetic_prompt(index: int, num_tokens: int, vocab_size: int) -> list[int]:
    """A prompt of an exact token length. R20.

    Token *ids* rather than text, because a benchmark asking for a 512-token prompt
    must get exactly 512 tokens -- going through the tokenizer would make the real
    length depend on the tokenizer, and the whole point of `--input-len` is that it is
    the independent variable.

    Deterministic and distinct per request by construction rather than by drawing
    randomly: an unseeded draw would break B4, and a seeded one would need an RNG here
    where the purity lint (correctly) does not allow one. Distinctness matters because
    identical prompts would all hit the prefix cache and quietly turn a throughput
    benchmark into a cache benchmark.
    """
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    # Avoid the low ids, which the mock tokenizer reserves for special tokens.
    base = 100 + (index * 7919) % max(1, vocab_size - 200)
    return [100 + (base + i * 31) % max(1, vocab_size - 200) for i in range(num_tokens)]
