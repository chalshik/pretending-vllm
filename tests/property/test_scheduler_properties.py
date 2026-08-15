"""Property-based tests over random workloads and configs. R21.1, R21.2.

Example-based tests check the cases someone thought of. These check the invariants
that must hold for *every* workload, which is where a scheduler bug actually hides:
in the interaction between a budget, a block count, and an arrival order nobody
would think to write down.

Every invariant here is one the spec names. They run against randomly generated
configs and workloads, and hypothesis shrinks any counterexample to a minimal case.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pvllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from pvllm.sampling_params import SamplingParams
from pvllm.v1.core.sched.scheduler import Scheduler
from pvllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from pvllm.v1.outputs import ModelRunnerOutput
from pvllm.v1.request import Request

# Kept small and fast: the whole suite has a 30-second budget (R21.5), and these
# ranges already cover the boundaries that matter -- a budget below one prompt, a
# pool below one request, a single sequence slot.
BLOCK_SIZES = st.sampled_from([4, 8, 16])
PROMPT_LENS = st.integers(min_value=1, max_value=64)
MAX_TOKENS = st.integers(min_value=1, max_value=8)

WORKLOADS = st.lists(st.tuples(PROMPT_LENS, MAX_TOKENS), min_size=1, max_size=8)

SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def build_scheduler(
    block_size: int,
    num_blocks: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    enable_caching: bool,
    max_model_len: int = 128,
) -> Scheduler:
    config = VllmConfig(
        model_config=ModelConfig(model="tiny-test", max_model_len=max_model_len),
        cache_config=CacheConfig(
            block_size=block_size, enable_prefix_caching=enable_caching
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
        ),
    )
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=32,
        dtype="bfloat16",
        dtype_bytes=2,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
    )
    return Scheduler(config, kv_cache_config, log_stats=False)


def add_workload(
    scheduler: Scheduler, workload: list[tuple[int, int]], shared_prefix: int = 0
) -> list[Request]:
    prefix = list(range(shared_prefix))
    requests = []
    for i, (prompt_len, max_tokens) in enumerate(workload):
        tokens = prefix + [1000 + i] * prompt_len
        request = Request(
            request_id=f"r{i}",
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=max_tokens),
            arrival_time=float(i),
        )
        scheduler.add_request(request)
        requests.append(request)
    return requests


def step(scheduler: Scheduler) -> None:
    """One engine step, with a runner stub that honours the real contract."""
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[
            [] if scheduler.requests[r].is_prefill_chunk else [7] for r in req_ids
        ],
    )
    scheduler.update_from_output(output, runner_output)


def drain(scheduler: Scheduler, limit: int = 2000) -> int:
    steps = 0
    while scheduler.has_requests() and steps < limit:
        step(scheduler)
        steps += 1
    return steps


# --- the invariants (R21.1) ------------------------------------------------


@SETTINGS
@given(
    workload=WORKLOADS,
    block_size=BLOCK_SIZES,
    budget=st.integers(min_value=1, max_value=128),
    max_num_seqs=st.integers(min_value=1, max_value=6),
    enable_caching=st.booleans(),
)
def test_every_admitted_request_terminates(
    workload, block_size, budget, max_num_seqs, enable_caching
):
    """The strongest invariant, and the one that catches deadlock.

    A budget below the shortest prompt, a pool below one request, a single sequence
    slot -- any of these could wedge the queue, and the failure mode is a hang
    rather than a wrong answer.
    """
    scheduler = build_scheduler(block_size, 128, budget, max_num_seqs, enable_caching)
    add_workload(scheduler, workload)

    steps = drain(scheduler)
    assert steps < 2000, "workload failed to drain"
    assert scheduler.get_num_unfinished_requests() == 0
    assert scheduler.running == []


@SETTINGS
@given(
    workload=WORKLOADS,
    block_size=BLOCK_SIZES,
    budget=st.integers(min_value=1, max_value=64),
    max_num_seqs=st.integers(min_value=1, max_value=6),
)
def test_the_token_budget_is_never_exceeded(workload, block_size, budget, max_num_seqs):
    """R5.3."""
    scheduler = build_scheduler(block_size, 128, budget, max_num_seqs, False)
    add_workload(scheduler, workload)

    for _ in range(200):
        if not scheduler.has_requests():
            break
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens <= budget
        assert sum(output.num_scheduled_tokens.values()) == (
            output.total_num_scheduled_tokens
        )
        req_ids = list(output.num_scheduled_tokens)
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={r: i for i, r in enumerate(req_ids)},
                sampled_token_ids=[
                    [] if scheduler.requests[r].is_prefill_chunk else [7]
                    for r in req_ids
                ],
            ),
        )


@SETTINGS
@given(
    workload=WORKLOADS,
    block_size=BLOCK_SIZES,
    max_num_seqs=st.integers(min_value=1, max_value=4),
)
def test_running_never_exceeds_max_num_seqs(workload, block_size, max_num_seqs):
    """R5.3."""
    scheduler = build_scheduler(block_size, 128, 64, max_num_seqs, False)
    add_workload(scheduler, workload)

    for _ in range(200):
        if not scheduler.has_requests():
            break
        step(scheduler)
        assert len(scheduler.running) <= max_num_seqs


@SETTINGS
@given(
    workload=WORKLOADS,
    block_size=BLOCK_SIZES,
    num_blocks=st.integers(min_value=8, max_value=64),
    enable_caching=st.booleans(),
)
def test_block_accounting_balances_and_usage_stays_in_range(
    workload, block_size, num_blocks, enable_caching
):
    """R21.1: total == free + allocated, and KV usage never leaves [0, 1].

    The pool asserts this internally on every mutation under
    PVLLM_DEBUG_INVARIANTS, which conftest sets; this drives it over random
    workloads so the assertion is actually reached from many states.
    """
    scheduler = build_scheduler(block_size, num_blocks, 64, 4, enable_caching)
    add_workload(scheduler, workload)
    pool = scheduler.kv_cache_manager.block_pool

    for _ in range(300):
        if not scheduler.has_requests():
            break
        step(scheduler)
        allocated = sum(1 for b in pool.blocks if b.ref_cnt > 0)
        assert pool.get_num_free_blocks() + allocated == num_blocks
        assert 0.0 <= scheduler.get_kv_cache_usage() <= 1.0
        assert all(b.ref_cnt >= 0 for b in pool.blocks)


@SETTINGS
@given(
    workload=WORKLOADS,
    block_size=BLOCK_SIZES,
    num_blocks=st.integers(min_value=8, max_value=64),
    enable_caching=st.booleans(),
)
def test_the_pool_returns_to_empty_after_draining(
    workload, block_size, num_blocks, enable_caching
):
    """A leak shows up here and nowhere else: every count balances step to step
    while blocks are never returned."""
    scheduler = build_scheduler(block_size, num_blocks, 64, 4, enable_caching)
    add_workload(scheduler, workload)
    drain(scheduler)
    assert scheduler.get_kv_cache_usage() == 0.0


# --- determinism (B4, C1) --------------------------------------------------


@SETTINGS
@given(
    workload=WORKLOADS,
    block_size=BLOCK_SIZES,
    budget=st.integers(min_value=4, max_value=64),
    enable_caching=st.booleans(),
)
def test_the_same_workload_yields_the_same_decision_sequence(
    workload, block_size, budget, enable_caching
):
    """C1: the scheduler decision sequence per step, and the total steps to drain."""

    def run() -> list[dict[str, int]]:
        scheduler = build_scheduler(block_size, 64, budget, 4, enable_caching)
        add_workload(scheduler, workload)
        decisions = []
        for _ in range(400):
            if not scheduler.has_requests():
                break
            output = scheduler.schedule()
            decisions.append(dict(sorted(output.num_scheduled_tokens.items())))
            req_ids = list(output.num_scheduled_tokens)
            scheduler.update_from_output(
                output,
                ModelRunnerOutput(
                    req_ids=req_ids,
                    req_id_to_index={r: i for i, r in enumerate(req_ids)},
                    sampled_token_ids=[
                        [] if scheduler.requests[r].is_prefill_chunk else [7]
                        for r in req_ids
                    ],
                ),
            )
        return decisions

    assert run() == run()


# --- prefix caching (C3) ---------------------------------------------------


@SETTINGS
@given(
    workload=st.lists(st.tuples(PROMPT_LENS, MAX_TOKENS), min_size=2, max_size=5),
    block_size=BLOCK_SIZES,
    shared_prefix=st.integers(min_value=0, max_value=48),
)
def test_caching_never_produces_more_hits_than_queries(
    workload, block_size, shared_prefix
):
    """A hit rate above 100% means the counters are double-counting somewhere."""
    scheduler = build_scheduler(block_size, 128, 64, 4, True)
    add_workload(scheduler, workload, shared_prefix=shared_prefix)
    drain(scheduler)

    stats = scheduler.kv_cache_manager.make_prefix_cache_stats()
    assert 0 <= stats.hits <= stats.queries
    assert 0.0 <= stats.hit_rate <= 1.0


@SETTINGS
@given(
    workload=st.lists(st.tuples(PROMPT_LENS, MAX_TOKENS), min_size=2, max_size=5),
    block_size=BLOCK_SIZES,
)
def test_caching_does_not_change_what_is_generated(workload, block_size):
    """The output must not depend on whether a prefix was recomputed or reused --
    that is the whole premise of a prefix cache, and the one thing a cache bug
    would silently break."""

    def run(enable_caching: bool) -> dict[str, list[int]]:
        scheduler = build_scheduler(block_size, 128, 64, 4, enable_caching)
        requests = add_workload(scheduler, workload, shared_prefix=16)
        drain(scheduler)
        return {r.request_id: list(r.output_token_ids) for r in requests}

    assert run(True) == run(False)
