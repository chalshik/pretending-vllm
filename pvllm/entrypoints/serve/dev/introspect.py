"""Live engine introspection. D9.

Upstream: (none -- pvllm addition)
Tier: B

The third leg of "super transparent" (D9), alongside the JSONL trace and the timeline
viewer. A trace answers questions after the fact; these answer them *while a product
is driving the engine* -- which is when you actually want to know why the ninth
request is queued behind the eighth.

There is no upstream counterpart to the reader itself. Upstream's `serve/dev/` routers
report configuration and environment; none of them opens up the scheduler or the block
pool, because on real hardware nobody can. `api_router.py` beside this file is the
port; this is the part upstream has no equivalent of.

**Read-only, and off unless asked for.** Every method here returns a dict and mutates
nothing. They are gated behind `--enable-debug-endpoints` because they expose prompt
token ids and per-request state, which is fine for a test double being driven by your
own test suite and not fine to leave on by habit.

Everything reported is *real*: the block map is the actual block pool, the request
states are the actual scheduler queues. The one exception is labelled -- the
cost-model breakdown is modeled (R9.5), and says so in its payload.

**This module is the single deliberate exception to B1.** It traverses through the
worker into the simulator -- the device's cost history, the memory profile, the
device card -- which no other module above the boundary may do. It has to: a
cost-model breakdown *is* simulator state, and showing it is what D9 asked for. The
exception is safe because the introspector decides nothing, so no engine behavior can
depend on what it reports, and because `tests/unit/test_purity.py` forbids anything
outside `entrypoints/serve/dev/` from importing it. Reaching for engine internals
from elsewhere is still a boundary violation; this is not a precedent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pvllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from pvllm.v1.engine.async_llm import AsyncLLM


class EngineIntrospector:
    """Reads engine state for the debug endpoints.

    Reaches into the engine core directly, which only works in process. When the
    multiprocess core lands this becomes an RPC; the shapes here are what it will
    return, so the endpoints do not change.
    """

    def __init__(self, engine: AsyncLLM) -> None:
        self.engine = engine
        self._check_in_process()

    def _check_in_process(self) -> None:
        """Refuse at construction rather than at request time. R4.2.

        These endpoints read engine state directly, which only works in process. A
        server started with both `--enable-debug-endpoints` and the multiprocess core
        used to accept the flag and then return 500 from every debug route -- a
        failure that shows up only when someone reaches for the debugging tools,
        which is the worst moment to discover a configuration is unsupported.
        """
        from pvllm.v1.engine.core_client import InprocClient
        from pvllm.v1.engine.dp_client import DPInprocClient

        client = self.engine.engine_core
        if isinstance(client, DPInprocClient):
            # R13.3. Replica 0's state, and said rather than implied: every number
            # below describes one replica, not the deployment. `/metrics` is the
            # aggregate surface. Rejecting DP outright -- which is what naming only
            # `InprocClient` did -- made `--data-parallel-size N
            # --enable-debug-endpoints` fail at startup with an error about an
            # environment variable the operator never set.
            return
        if not isinstance(client, InprocClient):
            raise NotImplementedError(
                "the /debug/* endpoints read engine state directly and need the "
                "in-process engine core, but this server was started with "
                "PVLLM_ENABLE_V1_MULTIPROCESSING=1. Over a process boundary the "
                "introspector would need an RPC for every field, which does not "
                "exist yet. Run without multiprocess mode to use --enable-debug-"
                "endpoints, or drop the flag."
            )

    @property
    def _core(self) -> Any:
        # Narrowed here rather than trusted: `_check_in_process` ran at construction,
        # but the type checker cannot carry that across, and an untyped reach into
        # the client would hide a real mistake behind `Any`.
        from pvllm.v1.engine.core_client import InprocClient
        from pvllm.v1.engine.dp_client import DPInprocClient

        client = self.engine.engine_core
        # R13.3. Replica 0 for a data-parallel deployment. `_check_in_process` allows
        # it, so this has to as well -- and both say the same thing: every number
        # below describes one replica. `/metrics` is the aggregate surface.
        assert isinstance(client, InprocClient | DPInprocClient)
        return client.engine_core

    def _replica_marker(self) -> dict[str, Any]:
        """Which replica the numbers below describe, in the payload.

        R13.3. A comment saying "replica 0, said rather than implied" said it only to
        whoever read the source. On a data-parallel deployment every `/debug/*`
        response describes one replica out of N, and a request routed elsewhere is
        simply absent -- so an operator chasing a stuck request got "not found" for a
        request the deployment was actively running, with nothing in the response to
        suggest looking further. Now the response says so.
        """
        from pvllm.v1.engine.dp_client import DPInprocClient

        client = self.engine.engine_core
        if not isinstance(client, DPInprocClient):
            return {}
        return {
            "data_parallel_rank": 0,
            "data_parallel_size": client.data_parallel_size,
            "scope": (
                f"replica 0 of {client.data_parallel_size}. Requests routed to the "
                f"other replicas are not visible here; /metrics is the aggregate."
            ),
        }

    # --- scheduler -----------------------------------------------------------

    def scheduler_state(self) -> dict[str, Any]:
        """What the scheduler is doing right now."""
        scheduler = self._core.scheduler
        return {
            **self._replica_marker(),
            "step": scheduler.step_index,
            "time": self._core.clock.time(),
            "elapsed": self._core.clock.elapsed,
            "clock_mode": self._core.clock.mode,
            "durations_are_modeled": True,
            "policy": scheduler.policy.value,
            "budget": {
                "max_num_batched_tokens": scheduler.max_num_scheduled_tokens,
                "max_num_seqs": scheduler.max_num_running_reqs,
                "max_num_partial_prefills": scheduler.max_num_partial_prefills,
            },
            "running": [
                self._request_summary(request) for request in scheduler.running
            ],
            # In queue order, so the head of this list is the next admission
            # candidate -- which is the question anyone reading this is asking.
            "waiting": [
                self._request_summary(request) for request in scheduler.waiting
            ],
            "num_preemptions_total": scheduler.num_preemptions_total,
            "kv_cache_usage": scheduler.get_kv_cache_usage(),
        }

    def _request_summary(self, request: Any) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "status": str(request.status),
            "arrival_time": request.arrival_time,
            "num_prompt_tokens": request.num_prompt_tokens,
            "num_computed_tokens": request.num_computed_tokens,
            "num_output_tokens": request.num_output_tokens,
            "max_tokens": request.max_tokens,
            "is_prefill_chunk": request.is_prefill_chunk,
            "num_preemptions": request.num_preemptions,
            "num_cached_tokens": request.num_cached_tokens,
            "priority": request.priority,
        }

    def request_state(self, request_id: str) -> dict[str, Any] | None:
        """One request's full state machine, or `None` if unknown."""
        scheduler = self._core.scheduler
        request = scheduler.requests.get(request_id)
        if request is None:
            return None

        blocks = scheduler.kv_cache_manager.get_blocks(request_id)
        summary = self._request_summary(request)
        summary.update(
            {
                "prompt_token_ids": list(request.prompt_token_ids),
                "output_token_ids": list(request.output_token_ids),
                "block_ids": blocks.get_block_ids(),
                "num_block_hashes": len(request.block_hashes),
                "is_finished": request.is_finished(),
                "finish_reason": (
                    str(request.get_finished_reason())
                    if request.is_finished()
                    else None
                ),
                "in_running_queue": request in scheduler.running,
            }
        )
        return summary

    # --- KV cache ------------------------------------------------------------

    def block_pool_state(self, limit: int = 64) -> dict[str, Any]:
        """The block pool, including who holds what.

        `limit` bounds the per-block listing: a real pool has tens of thousands of
        blocks, and a response that large is unreadable and slow to build. The
        *totals* are always exact; only the enumeration is truncated, and the
        payload says by how much rather than silently cutting off.
        """
        manager = self._core.scheduler.kv_cache_manager
        pool = manager.block_pool

        owners: dict[int, list[str]] = {}
        for request_id in self._core.scheduler.requests:
            for group in manager.get_blocks(request_id).blocks:
                for block in group:
                    owners.setdefault(block.block_id, []).append(request_id)

        held = [b for b in pool.blocks if b.ref_cnt > 0]
        return {
            "num_gpu_blocks": pool.num_gpu_blocks,
            "num_free_blocks": pool.get_num_free_blocks(),
            "usage": pool.get_usage(),
            "block_size": manager.block_size,
            "num_cached_block_hashes": len(pool.cached_block_hash_to_block),
            "num_evicted_blocks": pool.num_evicted_blocks,
            "num_allocated_blocks": len(held),
            "blocks_listed": min(len(held), limit),
            "blocks_omitted": max(0, len(held) - limit),
            "blocks": [
                {
                    "block_id": block.block_id,
                    "ref_cnt": block.ref_cnt,
                    "is_cached": block.block_hash is not None,
                    "held_by": owners.get(block.block_id, []),
                }
                for block in held[:limit]
            ],
            # A block with ref_cnt > 1 is shared by a prefix cache hit. Surfaced
            # directly because "is the cache actually sharing anything" is the
            # question a cache-tuning session keeps asking.
            "num_shared_blocks": sum(1 for b in held if b.ref_cnt > 1),
        }

    def prefix_cache_state(self) -> dict[str, Any]:
        """Cache effectiveness, in aggregate and per live request. R6.9.

        The aggregate rate says whether the cache is working; the per-request
        breakdown says *for whom*. A product shaping its prompts needs the second --
        a healthy overall rate can hide the one request template that never hits.
        """
        manager = self._core.scheduler.kv_cache_manager
        cache_config = self._core.vllm_config.cache_config
        # Refreshes two derived fields (evictions, cached blocks) from the pool and
        # returns the live counters. It does not reset them -- that is
        # `reset_prefix_cache`'s job -- so scraping this cannot perturb `/metrics`.
        stats = manager.make_prefix_cache_stats()
        return {
            "enabled": manager.enable_caching,
            "hash_algorithm": (
                cache_config.prefix_caching_hash_algo
                if manager.enable_caching
                else None
            ),
            **stats.as_dict(),
            "by_request": [
                {
                    "request_id": request.request_id,
                    "num_prompt_tokens": request.num_prompt_tokens,
                    "num_cached_tokens": request.num_cached_tokens,
                    "hit_rate": (
                        request.num_cached_tokens / request.num_prompt_tokens
                        if request.num_prompt_tokens
                        else 0.0
                    ),
                }
                for request in self._core.scheduler.requests.values()
            ],
        }

    # --- cost model ----------------------------------------------------------

    def step_costs(self, limit: int = 16) -> dict[str, Any]:
        """Why recent steps took what they took.

        **Modeled, not measured (R9.5).** Every row carries that label, because a
        breakdown this specific reads like a measurement and is not one.

        The window is what makes this useful: one step's numbers say nothing about
        whether a run is compute- or memory-bound, and a product tuning a batch size
        is asking exactly that.
        """
        worker = self._core.executor.driver_worker
        device = worker.device
        payload: dict[str, Any] = {
            "cost_model": device.cost_model.name,
            "provenance": "modeled",
            "history_size": device.history_size,
            "num_steps": device.num_steps,
            "steps": device.recent_steps(limit),
        }

        metadata = worker.model_runner.last_attn_metadata
        if metadata is not None:
            # The attention metadata is the cost model's input (R8.2), so showing it
            # beside the output is what makes a surprising duration diagnosable.
            payload["last_batch"] = {
                "num_reqs": metadata.num_reqs,
                "num_tokens": metadata.num_actual_tokens,
                "num_prefills": metadata.num_prefills,
                "num_decodes": metadata.num_decodes,
                "num_prefill_tokens": metadata.num_prefill_tokens,
                "num_decode_tokens": metadata.num_decode_tokens,
                "max_query_len": metadata.max_query_len,
                "max_seq_len": metadata.max_seq_len,
                "is_mixed_batch": metadata.is_mixed_batch,
                "num_common_prefix_blocks": metadata.num_common_prefix_blocks,
            }
        return payload

    # --- config --------------------------------------------------------------

    def config_dump(self) -> dict[str, Any]:
        """The resolved config, including what is being simulated."""
        config = self._core.vllm_config
        model = config.model_config
        cache = config.cache_config
        scheduler = config.scheduler_config
        sim = config.sim_config
        worker = self._core.executor.driver_worker
        profile = worker.memory_profile

        return {
            "model": {
                "model": model.model,
                "model_card": model.hf_config.name,
                "dtype": model.resolved_dtype,
                "max_model_len": model.max_model_len,
                "num_layers": model.get_num_layers(),
                "num_kv_heads": model.get_num_kv_heads(),
                "head_size": model.get_head_size(),
                "vocab_size": model.get_vocab_size(),
                "num_parameters": model.hf_config.num_parameters,
                "provenance": model.hf_config.provenance,
            },
            "cache": {
                "block_size": cache.block_size,
                "gpu_memory_utilization": cache.gpu_memory_utilization,
                "enable_prefix_caching": cache.enable_prefix_caching,
                "num_gpu_blocks": cache.num_gpu_blocks,
                "kv_cache_size_tokens": cache.kv_cache_size_tokens,
                "max_concurrency": cache.kv_cache_max_concurrency,
            },
            "scheduler": {
                "max_num_batched_tokens": scheduler.max_num_batched_tokens,
                "max_num_seqs": scheduler.max_num_seqs,
                "enable_chunked_prefill": scheduler.enable_chunked_prefill,
                "max_num_partial_prefills": scheduler.max_num_partial_prefills,
                "long_prefill_token_threshold": scheduler.long_prefill_token_threshold,
                "policy": scheduler.policy,
            },
            "simulated": {
                "device_card": sim.device_card,
                "device_provenance": worker.device_card.provenance,
                "clock_mode": sim.clock_mode,
                "cost_model_profile": sim.cost_model_profile,
                "jitter_sigma": sim.jitter_sigma,
                "output_length_policy": sim.output_length_policy,
                "seed": sim.seed,
            },
            "memory": (
                {
                    "capacity_bytes": profile.capacity_bytes,
                    "weight_bytes": profile.weight_bytes,
                    "activation_peak_bytes": profile.activation_peak_bytes,
                    "activation_is_modeled": profile.activation_is_modeled,
                    "kv_pool_bytes": profile.kv_pool_bytes,
                    "num_gpu_blocks": profile.num_gpu_blocks,
                    "max_concurrency": profile.max_concurrency,
                }
                if profile
                else None
            ),
            "startup": {
                "load_weights_seconds": worker.startup.load_weights_seconds,
                "profile_run_seconds": worker.startup.profile_run_seconds,
                "graph_capture_seconds": worker.startup.graph_capture_seconds,
                "total_seconds": worker.startup.total_seconds,
                "durations_are_modeled": True,
            },
        }

    def status_counts(self) -> dict[str, int]:
        """How many requests sit in each state. R5.1's state machine, counted.

        Every state is present with a zero rather than only the occupied ones, so a
        consumer can chart all of them without the series appearing and vanishing.
        """
        counts = dict.fromkeys((str(s) for s in RequestStatus), 0)
        for request in self._core.scheduler.requests.values():
            counts[str(request.status)] += 1
        return counts

    def request_ids(self) -> list[str]:
        """Every request the engine still tracks."""
        return list(self._core.scheduler.requests)
