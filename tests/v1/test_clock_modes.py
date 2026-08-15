"""The clock modes, end to end through the engine. R19.1, R4.4.

`tests/sim/test_clock.py` tests the clocks in isolation. This tests the thing that
actually matters: that choosing a mode changes how long a *run* takes without changing
anything the run reports.

The reason real mode exists at all is to let a product observe true latency -- its own
timeouts, its own retry logic, its own streaming behavior under load. That only works
if two things hold. The modeled numbers must be identical to a virtual-clock run, or a
CI run and a demo would disagree about the same workload. And the async engine must
release its event loop while the modeled time passes, or the server stops answering
for exactly as long as the step it is meant to be streaming through.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from pvllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from pvllm.sampling_params import SamplingParams
from pvllm.v1.engine.async_llm import AsyncLLM
from pvllm.v1.engine.llm_engine import LLMEngine

BASE = {
    "model": "tiny-test",
    "max_model_len": 256,
    "block_size": 8,
    "max_num_batched_tokens": 64,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
}


def run_sync(max_tokens: int = 6, **overrides) -> tuple[list[int], float, float]:
    """Generate once. Returns `(tokens, modeled elapsed, wall elapsed)`.

    The wall clock is read here, in a test, which is the one place it belongs: the
    whole question is whether modeled time and real time diverge, and that cannot be
    asked from inside a system forbidden to read a real clock.
    """
    engine = LLMEngine.from_engine_args(EngineArgs(**{**BASE, **overrides}))
    start_modeled = engine.engine_core.clock_time
    start_wall = time.perf_counter()

    engine.add_request("r0", "hello there", SamplingParams(max_tokens=max_tokens))
    tokens: list[int] = []
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                tokens = list(output.outputs[0].token_ids)

    modeled = engine.engine_core.clock_time - start_modeled
    wall = time.perf_counter() - start_wall
    engine.shutdown()
    return tokens, modeled, wall


# --- one timeline, three modes ---------------------------------------------


def test_every_mode_reports_the_same_modeled_time():
    """B4 across clock modes. A virtual-clock CI run and a real-clock demo have to
    agree about the numbers, or neither can be used to check the other."""
    virtual_tokens, virtual_modeled, _ = run_sync(clock_mode="virtual")
    scaled_tokens, scaled_modeled, _ = run_sync(clock_mode="scaled", time_scale=1000.0)

    assert virtual_tokens == scaled_tokens
    assert virtual_modeled == pytest.approx(scaled_modeled)


def test_a_virtual_clock_runs_a_workload_faster_than_it_models():
    """The property that makes G5's cheap experiments possible.

    Measured on a realistic card, because that is where the claim is true and worth
    making. **On a tiny card it is false**, and knowingly so: `tiny-test` on
    `tiny-2gb` models sub-millisecond steps, and the interpreter takes longer than
    that to run one -- so a 60-token run there models ~60 ms and spends ~90 ms. The
    speedup comes from modeling *expensive* hardware; simulating a toy is not faster
    than the toy. Anyone timing a virtual-clock run on a tiny card and finding it
    slow has not found a bug.
    """
    _, modeled, wall = run_sync(
        max_tokens=60,
        clock_mode="virtual",
        model="dense-8b",
        device_card="datacenter-80gb",
        cost_model_profile="roofline",
        max_model_len=512,
        block_size=16,
    )
    assert modeled > 0.0
    assert wall < modeled


#: Enough steps that the sleeping is measurable, few enough that the test is cheap.
#: The comparison below is always *against a virtual run of the same workload*, which
#: subtracts the interpreter's own cost -- an absolute threshold would be measuring
#: Python on a tiny card, where a modeled step is shorter than the code that models
#: it. `dense-8b` would give a cleaner separation and cost eight real seconds of
#: modeled weight loading to get it.
TIMED_TOKENS = 40


def test_a_real_clock_actually_spends_the_modeled_duration():
    """What makes real mode worth having: a product under test observes the latency
    the cost model predicted, so its own timeouts and retries are exercised."""
    _, modeled, virtual_wall = run_sync(max_tokens=TIMED_TOKENS, clock_mode="virtual")
    _, _, real_wall = run_sync(max_tokens=TIMED_TOKENS, clock_mode="real")

    # The difference between the two runs is the sleeping, and nothing else: same
    # workload, same interpreter, same modeled durations.
    assert real_wall - virtual_wall >= modeled * 0.5


def test_a_scaled_clock_compresses_the_same_run():
    """`time_scale` divides the sleep, not the modeled duration -- a long soak can be
    replayed in a fraction of the time without changing what it reports."""
    _, modeled, virtual_wall = run_sync(max_tokens=TIMED_TOKENS, clock_mode="virtual")
    _, _, real_wall = run_sync(max_tokens=TIMED_TOKENS, clock_mode="real")
    _, scaled_modeled, scaled_wall = run_sync(
        max_tokens=TIMED_TOKENS, clock_mode="scaled", time_scale=10.0
    )

    assert scaled_wall - virtual_wall < real_wall - virtual_wall
    # And the timeline it reports is untouched, which is the whole point.
    assert scaled_modeled == pytest.approx(modeled)


# --- the async engine under a spending clock -------------------------------


async def test_the_event_loop_keeps_running_while_a_step_costs_time():
    """The reason `execute_model_async` exists.

    Under a real or scaled clock the engine spends the step's modeled duration. If it
    spent it synchronously the whole event loop would stop -- so a server would stop
    streaming, stop answering `/health`, and stop accepting connections for exactly
    as long as it was busy. Here a second coroutine has to keep ticking throughout.

    Asserted on the *count* of ticks rather than on timing, because a loaded CI
    machine can make any wall-clock threshold flaky while the property being tested
    is simply "the loop was not blocked". A blocked loop ticks once or twice however
    slow the machine is; an unblocked one ticks steadily.

    Real mode rather than a compressed scaled one: the point is that the engine
    spends a stretch of time long enough for another coroutine to be scheduled in it,
    and `time_scale=200` would compress each step below the scheduler's granularity,
    passing for the wrong reason.
    """
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(**{**BASE, "clock_mode": "real"})
    )
    ticks = 0
    generating = True

    async def ticker() -> None:
        nonlocal ticks
        while generating:
            await asyncio.sleep(0.0005)
            ticks += 1

    async def generate() -> None:
        nonlocal generating
        async for _ in engine.generate(
            "hello there", SamplingParams(max_tokens=8), "r0"
        ):
            pass
        generating = False

    await asyncio.gather(generate(), ticker())
    engine.shutdown()

    assert ticks > 5, (
        f"the event loop only ticked {ticks} times while the engine generated 8 "
        f"tokens under a spending clock; the step is blocking it"
    )


async def test_the_async_engine_reports_the_same_tokens_under_any_clock():
    """A clock mode may change how long a run takes and nothing else."""

    async def run(**overrides) -> list[int]:
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(**{**BASE, **overrides, "seed": 3})
        )
        tokens: list[int] = []
        async for output in engine.generate(
            "hello there", SamplingParams(max_tokens=6), "r0"
        ):
            tokens = list(output.outputs[0].token_ids)
        engine.shutdown()
        return tokens

    assert await run(clock_mode="virtual") == await run(
        clock_mode="scaled", time_scale=500.0
    )


def test_stats_say_which_clock_produced_them():
    """R12.4. A dashboard built on modeled latency should be able to tell that it
    is, and the mode is the honest way to say so."""
    engine = LLMEngine.from_engine_args(
        EngineArgs(**{**BASE, "clock_mode": "scaled", "time_scale": 100.0})
    )
    stats = engine.make_stats()
    assert stats["clock_mode"] == "scaled"
    # Still modeled: a scaled clock sleeps, but the duration it sleeps came from the
    # cost model, not from measuring anything.
    assert stats["durations_are_modeled"] is True
    engine.shutdown()
