"""The multiprocess engine core. R4.2, D2, R19.1.

These spawn real OS processes, so they are the slowest tests in the suite by a wide
margin -- kept few and small on purpose (R21.5 budgets 30 seconds for everything).

What they are for is the handful of properties that *only* break across a process
boundary, and that the in-process client cannot detect no matter how thoroughly it is
exercised: every wire type actually serializes, the frontend can still learn the
engine's time when it cannot reach the clock, and a dead engine reaches the frontend
rather than hanging it.
"""

from __future__ import annotations

import pytest

from pvllm.engine.arg_utils import EngineArgs
from pvllm.sampling_params import SamplingParams
from pvllm.v1.engine import EngineCoreEventType, EngineCoreRequest
from pvllm.v1.engine.core_client import EngineCoreClient

BASE = {
    "model": "tiny-test",
    "max_model_len": 256,
    "block_size": 8,
    "max_num_batched_tokens": 64,
    "max_num_seqs": 4,
    "device_card": "tiny-2gb",
    "disable_log_stats": True,
}


@pytest.fixture
def client():
    config = EngineArgs(**BASE).create_engine_config()
    engine_client = EngineCoreClient.make_client(config, multiprocess_mode=True)
    try:
        yield engine_client
    finally:
        engine_client.shutdown()


def drain(client, request_ids: set[str], limit: int = 200) -> dict[str, list]:
    """Step until every named request finishes.

    An empty poll is *not* a stopping condition while requests are still outstanding.
    `get_output` returns `{}` when the engine reports no pending work, and there is a
    window right after `add_request` where that is true simply because the message is
    still in flight -- the frontend has sent it and the child has not yet enqueued it.
    Breaking there raced the child on a loaded runner and made the test report that
    nothing finished, which is how it failed on macOS/3.13 while passing everywhere
    else.

    The caller knows what it submitted, so that is the loop condition; `limit` is the
    backstop for a genuinely stuck engine.
    """
    tokens: dict[str, list[int]] = {req_id: [] for req_id in request_ids}
    events: dict[str, list] = {req_id: [] for req_id in request_ids}
    finished: set[str] = set()
    for _ in range(limit):
        if finished == request_ids:
            break
        outputs = client.get_output()
        if not outputs:
            continue
        for client_outputs in outputs.values():
            for output in client_outputs.outputs:
                tokens[output.request_id].extend(output.new_token_ids)
                events[output.request_id].extend(output.events or ())
                if output.finish_reason is not None:
                    finished.add(output.request_id)
    return {"tokens": tokens, "events": events, "finished": finished}


def request(request_id: str, prompt_len: int = 8, max_tokens: int = 5):
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(range(100, 100 + prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


# --- the transport ---------------------------------------------------------


def test_a_request_survives_the_wire_and_comes_back_generated(client):
    """The whole point of not stubbing this: every field of every wire type is
    actually encoded and decoded. `SamplingParams` reaching the core intact is what
    the in-process client could never prove, because it passes the object by
    reference."""
    client.add_request(request("r0", max_tokens=5))
    result = drain(client, {"r0"})

    assert result["finished"] == {"r0"}
    assert len(result["tokens"]["r0"]) == 5


def test_several_requests_run_concurrently_across_the_boundary(client):
    ids = {f"r{i}" for i in range(4)}
    for req_id in ids:
        client.add_request(request(req_id, max_tokens=4))
    result = drain(client, ids)

    assert result["finished"] == ids
    assert all(len(result["tokens"][req_id]) == 4 for req_id in ids)


def test_the_frontend_learns_the_engine_clock_without_reaching_for_one(client):
    """R19.1's whole point, and the reason `clock_time` has been on the interface
    since the first commit. The clock lives in the child; the only way the frontend
    knows the time is because the engine told it."""
    start = client.clock_time
    assert start > 0.0

    client.add_request(request("r0", max_tokens=6))
    drain(client, {"r0"})

    assert client.clock_time > start


def test_lifecycle_events_cross_the_boundary(client):
    """Per-request timing is computed from these, not from a frontend clock. If they
    did not survive serialization the queue/prefill split would silently read zero
    in multiprocess mode and nowhere else."""
    client.add_request(request("r0", max_tokens=5))
    events = drain(client, {"r0"})["events"]["r0"]

    kinds = {event.type for event in events}
    assert EngineCoreEventType.QUEUED in kinds
    assert EngineCoreEventType.SCHEDULED in kinds
    # Stamped by the core, so both sit on the modeled timeline the frontend sees.
    assert all(event.timestamp > 0.0 for event in events)


def test_utility_calls_round_trip(client):
    """Stats and cache resets are request/response over the same socket as the
    output stream, correlated by call id. If the demultiplexing were wrong this
    would return an output frame decoded as a reply."""
    stats = client.make_stats()
    assert stats["step_index"] == 0
    assert stats["durations_are_modeled"] is True

    client.add_request(request("r0", max_tokens=4))
    drain(client, {"r0"})

    assert client.make_stats()["step_index"] > 0
    assert client.reset_prefix_cache() in (True, False)


def test_aborting_reaches_the_engine(client):
    """R2.4 across a process boundary: the abort has to arrive and free blocks, not
    just be dropped on a socket."""
    client.add_request(request("r0", max_tokens=100))
    client.get_output()
    client.abort_requests(["r0"])

    # Drains within a step or two; the abort is processed at the top of the loop.
    for _ in range(20):
        if client.make_stats()["num_running_reqs"] == 0:
            break
        client.get_output()
    assert client.make_stats()["num_running_reqs"] == 0


def test_output_frames_are_not_lost_while_a_utility_call_is_in_flight(client):
    """A step's tokens arriving between a utility call and its reply must be kept.
    Dropping them would lose a request's output entirely, and only under the timing
    where a scrape happens to land mid-generation."""
    client.add_request(request("r0", max_tokens=8))
    # Interleave scrapes with stepping, which is what a `/metrics` scrape during
    # generation does.
    tokens: list[int] = []
    for _ in range(60):
        client.make_stats()
        outputs = client.get_output()
        if not outputs:
            break
        for client_outputs in outputs.values():
            for output in client_outputs.outputs:
                tokens.extend(output.new_token_ids)
        if len(tokens) >= 8:
            break
    assert len(tokens) == 8


# --- failure modes ---------------------------------------------------------


def test_shutdown_is_idempotent(client):
    client.shutdown()
    client.shutdown()
    assert not client.proc.is_alive()


def test_using_a_shut_down_client_does_not_hang():
    """R4.5. A frontend that keeps calling after shutdown should get an error, not
    block forever on a socket nobody is reading."""
    from pvllm.v1.engine.core_client_mp import EngineDeadError

    config = EngineArgs(**BASE).create_engine_config()
    engine_client = EngineCoreClient.make_client(config, multiprocess_mode=True)
    engine_client.shutdown()

    with pytest.raises(EngineDeadError):
        engine_client.make_stats()


def test_the_async_client_refuses_the_blocking_interface():
    """Unsupported-path discipline. A synchronous `make_stats` on the async client
    would have to block the event loop on a round trip, so it names the async form
    rather than quietly doing it."""
    from pvllm.v1.engine.core_client_mp import AsyncMPClient

    config = EngineArgs(**BASE).create_engine_config()
    engine_client = AsyncMPClient(config, log_stats=False)
    try:
        with pytest.raises(NotImplementedError, match="make_stats_async"):
            engine_client.make_stats()
        with pytest.raises(NotImplementedError, match="get_output_async"):
            engine_client.get_output()
    finally:
        engine_client.shutdown()


def test_multiprocess_is_off_unless_asked_for(monkeypatch):
    """A deliberate divergence from upstream, which defaults it on.

    Determinism is load-bearing here -- it is what the conformance suite compares and
    what makes two runs of a sweep differ only in what was configured. Multiprocess
    mode gives that up (arrival order becomes a function of OS scheduling), so it is
    opt-in rather than the default anyone gets by accident.
    """
    from pvllm import envs
    from pvllm.v1.engine.core_client import InprocClient
    from pvllm.v1.engine.llm_engine import LLMEngine

    assert envs.PVLLM_ENABLE_V1_MULTIPROCESSING is False
    engine = LLMEngine.from_engine_args(EngineArgs(**BASE))
    assert isinstance(engine.engine_core, InprocClient)
    engine.shutdown()


def test_the_offline_llm_works_over_a_process_boundary(monkeypatch):
    """The whole stack, opted in through the environment: `LLM.generate` with the
    engine in another process. Everything between the two is the wire protocol."""
    monkeypatch.setenv("PVLLM_ENABLE_V1_MULTIPROCESSING", "1")

    from pvllm.entrypoints.llm import LLM
    from pvllm.v1.engine.core_client_mp import SyncMPClient

    engine = LLM(**{**BASE, "model": "tiny-test"})
    try:
        assert isinstance(engine.llm_engine.engine_core, SyncMPClient)
        outputs = engine.generate(
            ["first prompt", "second prompt"], SamplingParams(max_tokens=5)
        )
        assert len(outputs) == 2
        assert all(len(o.outputs[0].token_ids) == 5 for o in outputs)
        # Detokenized text made it through, so the frontend's own half of the
        # pipeline works on the far side of the boundary too.
        assert all(o.outputs[0].text for o in outputs)
        assert all(o.outputs[0].finish_reason == "length" for o in outputs)
    finally:
        engine.shutdown()


def test_timing_still_decomposes_over_a_process_boundary(monkeypatch):
    """The queue/prefill split survives the transport.

    It is computed from the core's own QUEUED and SCHEDULED stamps rather than from
    a frontend clock, which is exactly so that it means the same thing here as in
    process -- a frontend estimate would be stale by up to a step and would inflate
    every queue time.
    """
    monkeypatch.setenv("PVLLM_ENABLE_V1_MULTIPROCESSING", "1")

    from pvllm.v1.engine.llm_engine import LLMEngine

    engine = LLMEngine.from_engine_args(EngineArgs(**{**BASE, "max_num_seqs": 1}))
    try:
        for index in range(3):
            engine.add_request(
                f"r{index}", f"prompt {index}", SamplingParams(max_tokens=4)
            )
        finished = []
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            engine.step()
            finished.extend(engine.last_iteration_stats.finished_requests)

        assert len(finished) == 3
        for stats in finished:
            assert stats.queue_time + stats.prefill_time == pytest.approx(
                stats.time_to_first_token
            )
        # max_num_seqs=1, so everything after the first request waited.
        assert max(f.queue_time for f in finished) > 0.0
    finally:
        engine.shutdown()


async def test_the_async_client_generates():
    """The transport `AsyncLLM` uses in multiprocess mode."""
    from pvllm.v1.engine.core_client_mp import AsyncMPClient

    config = EngineArgs(**BASE).create_engine_config()
    engine_client = AsyncMPClient(config, log_stats=False)
    try:
        engine_client.add_request(request("r0", max_tokens=5))
        tokens: list[int] = []
        for _ in range(200):
            outputs = await engine_client.get_output_async()
            for client_outputs in outputs.values():
                for output in client_outputs.outputs:
                    tokens.extend(output.new_token_ids)
            if len(tokens) >= 5:
                break
        assert len(tokens) == 5
        assert engine_client.clock_time > 0.0
    finally:
        engine_client.shutdown()
