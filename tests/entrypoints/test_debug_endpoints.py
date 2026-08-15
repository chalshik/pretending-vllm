"""The `/debug/*` introspection surface. D9.

The trace answers questions after a run; these answer them during one. The tests that
matter here are the ones that would catch a debug endpoint *lying*: reporting a state
the engine is not actually in, or perturbing the state it reports on.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from prometheus_client import CollectorRegistry

from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.entrypoints.openai.api_server import build_app

MODEL = "test-model"


def make_config(**overrides):
    return AsyncEngineArgs(
        model="dense-0.6b",
        served_model_name=MODEL,
        max_model_len=512,
        block_size=16,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        device_card="workstation-24gb",
        disable_log_stats=True,
        **overrides,
    ).create_engine_config()


@pytest.fixture
async def client():
    app = build_app(
        make_config(), registry=CollectorRegistry(), enable_debug_endpoints=True
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client


async def complete(client: httpx.AsyncClient, prompt: str, max_tokens: int = 8):
    response = await client.post(
        "/v1/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": max_tokens},
    )
    assert response.status_code == 200, response.text
    return response.json()


# --- the gate --------------------------------------------------------------


async def test_debug_routes_are_absent_by_default():
    """The flag is the whole point: these expose prompt token ids, so a server
    started without asking for them must not serve them."""
    app = build_app(make_config(), registry=CollectorRegistry())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for path in (
            "/debug/scheduler",
            "/debug/requests",
            "/debug/blocks",
            "/debug/prefix_cache",
            "/debug/cost_model",
            "/debug/config",
        ):
            assert (await client.get(path)).status_code == 404, path


async def test_the_ordinary_surface_is_unchanged_when_debug_is_on(client):
    """Turning introspection on must not change what the engine does -- otherwise
    every debugging session is observing a different system than the broken one."""
    assert (await client.get("/health")).status_code == 200
    payload = await complete(client, "hello")
    assert payload["choices"][0]["text"]


# --- scheduler -------------------------------------------------------------


async def test_scheduler_state_reports_an_idle_engine(client):
    state = (await client.get("/debug/scheduler")).json()
    assert state["running"] == []
    assert state["waiting"] == []
    assert state["policy"] == "fcfs"
    assert state["budget"]["max_num_batched_tokens"] == 256
    assert state["budget"]["max_num_seqs"] == 4
    # Durations here came from the cost model, not a wall clock. Anything reading
    # this endpoint to build a latency dashboard needs to know that.
    assert state["clock_mode"] == "virtual"
    assert state["durations_are_modeled"] is True


async def test_scheduler_state_shows_work_in_flight(client):
    """The endpoint is worthless if it can only describe an idle engine."""
    seen: list[dict] = []

    async def watch() -> None:
        for _ in range(200):
            state = (await client.get("/debug/scheduler")).json()
            if state["running"] or state["waiting"]:
                seen.append(state)
                return
            await asyncio.sleep(0)

    await asyncio.gather(
        complete(client, "a prompt long enough to take several steps", 32), watch()
    )

    assert seen, "never observed a request in the scheduler while one was generating"
    state = seen[0]
    tracked = state["running"] + state["waiting"]
    assert tracked
    request = tracked[0]
    assert request["num_prompt_tokens"] > 0
    assert request["max_tokens"] == 32
    assert "is_prefill_chunk" in request


async def test_queued_requests_appear_in_admission_order(client):
    """A count of waiting requests cannot answer "why is mine not running", which is
    the question this endpoint exists for. The order can."""
    observed: list[list[str]] = []

    async def watch() -> None:
        for _ in range(400):
            state = (await client.get("/debug/scheduler")).json()
            waiting = [r["request_id"] for r in state["waiting"]]
            if len(waiting) >= 2:
                observed.append(waiting)
                return
            await asyncio.sleep(0)

    # More requests than max_num_seqs, so some must queue.
    await asyncio.gather(
        *(complete(client, f"prompt number {i}", 24) for i in range(10)), watch()
    )

    assert observed, "never observed two requests queued at once"
    waiting = observed[0]
    # FCFS: request ids are minted in arrival order, so the queue must be sorted by
    # the numeric suffix. Any other order means the queue is not FCFS.
    suffixes = [int(rid.rsplit("-", 1)[-1]) for rid in waiting]
    assert suffixes == sorted(suffixes), waiting


# --- per-request -----------------------------------------------------------


async def test_an_unknown_request_is_a_404(client):
    response = await client.get("/debug/requests/does-not-exist")
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["error"]


async def test_request_state_carries_the_block_table(client):
    captured: list[dict] = []

    async def watch() -> None:
        for _ in range(400):
            listing = (await client.get("/debug/requests")).json()
            for request_id in listing["request_ids"]:
                response = await client.get(f"/debug/requests/{request_id}")
                if response.status_code == 200 and response.json()["block_ids"][0]:
                    captured.append(response.json())
                    return
            await asyncio.sleep(0)

    await asyncio.gather(complete(client, "a prompt with several blocks", 32), watch())

    assert captured, "never caught a request holding blocks"
    state = captured[0]
    assert state["prompt_token_ids"]
    assert state["num_prompt_tokens"] == len(state["prompt_token_ids"])
    # The block table must cover every token the request has computed. A shorter
    # table means the scheduler scheduled tokens the KV manager never allocated
    # for -- the class of bug R8.3's oracle exists to catch.
    num_blocks = len(state["block_ids"][0])
    assert num_blocks * 16 >= state["num_computed_tokens"]


async def test_status_counts_list_every_state(client):
    counts = (await client.get("/debug/requests")).json()["counts"]
    assert counts["WAITING"] == 0
    assert counts["RUNNING"] == 0
    # Present-with-zero rather than omitted, so a chart of all states does not have
    # series appear and disappear.
    assert "PREEMPTED" in counts
    assert "FINISHED_LENGTH_CAPPED" in counts


# --- KV cache --------------------------------------------------------------


async def test_block_pool_totals_are_exact_even_when_the_listing_truncates(client):
    state = (await client.get("/debug/blocks?limit=0")).json()
    assert state["num_gpu_blocks"] > 0
    assert state["blocks"] == []
    assert state["blocks_listed"] == 0
    # The totals must survive truncation, or a truncated response reads as an empty
    # pool rather than a long one.
    assert state["num_free_blocks"] == state["num_gpu_blocks"]
    assert state["num_allocated_blocks"] == 0


async def test_blocks_report_who_holds_them(client):
    captured: list[dict] = []

    async def watch() -> None:
        for _ in range(400):
            state = (await client.get("/debug/blocks")).json()
            if state["num_allocated_blocks"]:
                captured.append(state)
                return
            await asyncio.sleep(0)

    await asyncio.gather(complete(client, "hold some blocks please", 32), watch())

    assert captured, "never observed an allocated block"
    state = captured[0]
    held = [block for block in state["blocks"] if block["held_by"]]
    assert held, "blocks were allocated but no owner was reported"
    assert all(block["ref_cnt"] >= 1 for block in held)
    assert (
        state["num_free_blocks"] + state["num_allocated_blocks"]
        <= (state["num_gpu_blocks"])
    )


async def test_prefix_cache_state_tracks_a_shared_prefix(client):
    shared = "the same long shared prefix that both requests begin with"
    await complete(client, shared + " first")
    await complete(client, shared + " second")

    state = (await client.get("/debug/prefix_cache")).json()
    assert state["enabled"] is True
    assert state["hash_algorithm"] == "sha256"
    assert state["prefix_cache_queries"] > 0
    assert state["prefix_cache_hits"] > 0
    assert 0.0 < state["prefix_cache_hit_rate"] <= 1.0


async def test_prefix_cache_reports_which_requests_hit(client):
    """An aggregate rate can look healthy while one prompt template never hits. The
    per-request map is what distinguishes those."""
    shared = "a prefix long enough to span more than one sixteen-token block, twice"
    captured: list[list[dict]] = []

    async def watch() -> None:
        for _ in range(400):
            state = (await client.get("/debug/prefix_cache")).json()
            hits = [r for r in state["by_request"] if r["num_cached_tokens"] > 0]
            if hits:
                captured.append(hits)
                return
            await asyncio.sleep(0)

    await complete(client, shared + " warm the cache")
    await asyncio.gather(complete(client, shared + " reuse it", 24), watch())

    assert captured, "no live request was reported as hitting the prefix cache"
    hit = captured[0][0]
    assert hit["num_cached_tokens"] <= hit["num_prompt_tokens"]
    assert 0.0 < hit["hit_rate"] <= 1.0


async def test_reading_the_prefix_cache_does_not_drain_it(client):
    """A debug endpoint that consumed the counters would silently empty `/metrics`
    for anyone who happened to look."""
    shared = "another shared prefix, long enough to fill a block or two"
    await complete(client, shared + " one")
    await complete(client, shared + " two")

    first = (await client.get("/debug/prefix_cache")).json()
    second = (await client.get("/debug/prefix_cache")).json()
    assert first["prefix_cache_hits"] == second["prefix_cache_hits"]
    assert first["prefix_cache_queries"] == second["prefix_cache_queries"]

    scrape = (await client.get("/metrics")).text
    assert "vllm:prefix_cache_hits_total" in scrape


# --- cost model ------------------------------------------------------------


async def test_cost_model_reports_a_window_of_steps(client):
    await complete(client, "run several steps", 16)
    state = (await client.get("/debug/cost_model")).json()

    assert state["cost_model"] == "constant"
    assert state["provenance"] == "modeled"
    assert len(state["steps"]) > 1, "a single step cannot show whether a run is bound"
    # Every row is labeled, not just the envelope: rows get copied out of context.
    assert all(step["provenance"] == "modeled" for step in state["steps"])
    assert [step["step"] for step in state["steps"]] == sorted(
        step["step"] for step in state["steps"]
    )
    assert all(step["duration"] > 0 for step in state["steps"])


async def test_cost_model_window_is_bounded(client):
    """An unbounded history would make a long-running server leak one record per
    step, which is a worse bug than the one it was added to diagnose."""
    await complete(client, "a long generation", 64)
    state = (await client.get("/debug/cost_model?limit=256")).json()
    assert state["num_steps"] >= len(state["steps"])
    assert len(state["steps"]) <= state["history_size"]


async def test_cost_model_shows_the_batch_that_produced_the_last_step(client):
    await complete(client, "a prompt", 8)
    batch = (await client.get("/debug/cost_model")).json()["last_batch"]
    assert batch["num_reqs"] >= 1
    assert batch["num_tokens"] >= 1
    assert batch["num_prefills"] + batch["num_decodes"] == batch["num_reqs"]


# --- config ----------------------------------------------------------------


async def test_config_dump_separates_configured_from_derived(client):
    config = (await client.get("/debug/config")).json()

    assert config["model"]["model"] == "dense-0.6b"
    assert config["cache"]["block_size"] == 16
    assert config["scheduler"]["max_num_batched_tokens"] == 256

    # num_gpu_blocks is derived by the memory model, not configured -- and is the
    # usual surprise when a product's concurrency does not match its expectation.
    assert config["cache"]["num_gpu_blocks"] > 0
    assert config["memory"]["num_gpu_blocks"] == config["cache"]["num_gpu_blocks"]
    assert config["memory"]["capacity_bytes"] > config["memory"]["weight_bytes"]


async def test_config_dump_labels_what_is_simulated(client):
    """The whole reason this project exists is that these numbers are not real. A
    config dump that did not say so would be actively misleading."""
    config = (await client.get("/debug/config")).json()

    assert config["simulated"]["device_card"] == "workstation-24gb"
    assert "approximation" in config["simulated"]["device_provenance"]
    assert "approximation" in config["model"]["provenance"]
    assert config["memory"]["activation_is_modeled"] is True
    assert config["startup"]["durations_are_modeled"] is True


async def test_config_dump_survives_json_encoding(client):
    """Everything reported must be JSON-serializable. A dataclass or an enum that
    slipped in would 500 the endpoint only once that config path was exercised."""
    response = await client.get("/debug/config")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


# --- the CLI flag ----------------------------------------------------------


def test_serve_defaults_to_debug_endpoints_off():
    import argparse

    from pvllm.entrypoints.cli.serve import add_serve_args

    parser = add_serve_args(argparse.ArgumentParser())
    assert parser.parse_args(["--model", "dense-0.6b"]).enable_debug_endpoints is False
    assert (
        parser.parse_args(
            ["--model", "dense-0.6b", "--enable-debug-endpoints"]
        ).enable_debug_endpoints
        is True
    )
