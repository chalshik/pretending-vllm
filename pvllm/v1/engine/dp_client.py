"""Data parallelism: several engine replicas behind one router. R13.3.

Upstream: vllm/v1/engine/core_client.py (the `DPLBAsyncMPClient` half)
Tier: B

Data parallelism is not sharding. Each replica is a *whole* engine -- its own copy of
the weights, its own device, its own KV pool, its own scheduler -- and a router picks
one per request. Nothing is split; a request lives entirely inside one replica for its
whole life.

That structure has three consequences a capacity plan turns on, and all three are
reproduced here rather than assumed:

* **Capacity multiplies, a single request does not get faster.** Four replicas serve
  roughly four times the throughput at the same per-request latency. A request too
  large for one replica's KV pool is too large for the deployment.
* **The prefix cache is partitioned.** Two requests sharing a long system prompt hit
  the cache only if the router sent them to the same replica. A workload whose hit
  rate looks excellent on one engine can lose most of it at `--data-parallel-size 8`,
  and that is the single most surprising thing about turning DP on.
* **The router's policy is load, not round-robin.** Upstream scores each replica by
  its in-flight count against the coordinator's `waiting + running` snapshot, and
  penalises a queue in proportion to KV pressure. That policy is ported verbatim,
  because "which replica gets this request" is exactly what a DP experiment asks.

**Expert parallelism changes what these replicas are.** With
`--enable-expert-parallel` they stop being independent copies and become shards of one
expert set (R13.4), which is why the memory picture improves so sharply. They then have
to step in *lockstep*: the MoE collective is taken across every EP rank, so a replica
with no work of its own cannot skip a step -- it runs a dummy single-token forward pass
to keep the collective whole, and that is real device time spent producing nothing.

`_lockstep_round` models it, and the stats report it per replica, because the arithmetic
is invisible otherwise: one request on a four-replica EP deployment costs three dummy
steps for every real one, and nothing in upstream's metrics says so. A deployment can
sit at full device utilisation with a quarter of the goodput it looks like it has.

**Clock.** Each replica owns its own clock, because each models its own device, and
the deployment's elapsed time is the *slowest* replica's rather than the sum -- they
run concurrently. Under a real or scaled clock they would have to run concurrently in
wall-clock terms too, and stepping them from one process serialises them, so those
modes refuse rather than reporting a deployment four times slower than the one being
modeled.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.v1.engine import EngineCoreOutputs
from pvllm.v1.engine.core import EngineCore
from pvllm.v1.engine.core_client import EngineCoreClient
from pvllm.v1.executor.abstract import Executor

logger = init_logger(__name__)


class DPInprocClient(EngineCoreClient):
    """`data_parallel_size` engine replicas in this process, with upstream's router."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor] | None = None,
        log_stats: bool = True,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        self.data_parallel_size = parallel_config.data_parallel_size

        sim_config = vllm_config.sim_config
        if sim_config is not None and sim_config.clock_mode != "virtual":
            raise NotImplementedError(
                f"--data-parallel-size {self.data_parallel_size} with "
                f"--clock-mode {sim_config.clock_mode} is not supported. The replicas "
                f"run concurrently on separate devices, and stepping them from one "
                f"process spends their durations in sequence -- a real-clock run "
                f"would take {self.data_parallel_size}x as long as the deployment it "
                f"is modeling and exercise a client's timeouts against a number that "
                f"is not the answer. Use the virtual clock, which models the "
                f"concurrency correctly."
            )

        # `EngineCore` builds its own clock (R19.1), so the replicas already have one
        # each -- which is what makes their modeled times independent.
        #
        # One trace file per replica (R19.4). A timeline is a property of one device,
        # and N replicas writing one file would interleave steps from engines whose
        # clocks are independent -- a trace that reads as a single engine behaving
        # impossibly. `pvllm trace view` reads a replica's file unchanged.
        self.engine_cores: list[EngineCore] = [
            EngineCore(
                self._config_for_rank(vllm_config, rank),
                executor_class=executor_class,
                log_stats=log_stats,
            )
            for rank in range(self.data_parallel_size)
        ]

        #: Which replica holds each in-flight request, so an abort reaches it.
        self.request_to_engine: dict[str, int] = {}
        #: This client's exact in-flight count per replica. Upstream keeps the same
        #: counter beside the coordinator's snapshot, because the count is exact
        #: while a snapshot can be stale.
        self.engine_inflight: list[int] = [0] * self.data_parallel_size
        # R13.4. Lockstep, and only under expert parallelism. Plain data parallelism
        # is independent whole engines behind a router and stays that way -- a replica
        # with nothing to do idles, as it should. Under EP the replicas are shards of
        # one expert set, the MoE collective is taken across all of them, and a
        # replica that skipped a step would leave the others waiting on a message
        # that never arrives. So it runs a forward pass over one token instead.
        parallel = vllm_config.parallel_config
        self.lockstep = parallel.enable_expert_parallel and self.data_parallel_size > 1
        #: Rounds in which at least one replica had work, so the others paid a dummy
        #: step. Reported as `lockstep_rounds`, and worth reporting beside
        #: `num_dummy_steps` because the ratio is the imbalance: R rounds against
        #: D dummy steps says D/R replica-steps went to nothing on an average round.
        self.lockstep_rounds = 0

        #: Where the scan starts. Upstream rotates this per request to break ties;
        #: here the in-flight counter below is incremented on every routed request,
        #: so the previously-chosen replica already scores strictly higher and the
        #: tie the rotation exists to break never arises. Kept at zero, and the scan
        #: order is therefore fixed -- stated because the rotation *was* here, was
        #: inert, and the comment claiming otherwise outlived two readings.
        self.scan_start = 0

    @staticmethod
    def _config_for_rank(vllm_config: VllmConfig, rank: int) -> VllmConfig:
        """This replica's config: its own trace path, everything else shared."""
        sim_config = vllm_config.sim_config
        if sim_config is None or not sim_config.trace_path:
            return vllm_config
        path = Path(sim_config.trace_path)
        # `sim_config` lives under `device_config`, so the replacement goes through
        # it -- the nesting is upstream's `DeviceConfig`, and R1.3 puts the simulator
        # knobs there rather than beside them.
        return replace(
            vllm_config,
            device_config=replace(
                vllm_config.device_config,
                sim_config=replace(
                    sim_config,
                    trace_path=str(
                        path.with_name(f"{path.stem}.dp{rank}{path.suffix}")
                    ),
                ),
            ),
        )

    # --- the router ----------------------------------------------------------

    def get_core_engine_for_request(self, request: Any) -> int:
        """Which replica should serve this request. R13.3.

        Upstream's score, ported: the greater of this client's exact in-flight count
        and the replica's own `waiting + running`, plus a penalty for queueing on a
        replica whose KV cache is already under pressure. The penalty ramps from zero
        at 50% usage to three times the queue length at 100%, because a queue on a
        KV-bound replica drains slowly while a queue on an empty one is transient.
        """
        rank = getattr(request, "data_parallel_rank", None)
        if rank is not None:
            if not 0 <= rank < self.data_parallel_size:
                raise ValueError(
                    f"data_parallel_rank {rank} is out of range for "
                    f"data_parallel_size {self.data_parallel_size}"
                )
            chosen = int(rank)
        else:
            min_score: float = sys.maxsize
            chosen = 0
            for offset in range(self.data_parallel_size):
                index = (self.scan_start + offset) % self.data_parallel_size
                stats = self.engine_cores[index].make_stats()
                waiting = int(stats["num_waiting_reqs"])
                running = int(stats["num_running_reqs"])
                usage = float(stats["kv_cache_usage"])
                score: float = max(self.engine_inflight[index], waiting + running)
                if waiting:
                    score += waiting * 6.0 * max(0.0, usage - 0.5)
                if score < min_score:
                    min_score = score
                    chosen = index

        self.request_to_engine[request.request_id] = chosen
        self.engine_inflight[chosen] += 1
        return chosen

    # --- the client interface ------------------------------------------------

    def add_request(self, request: Any) -> None:
        self.engine_cores[self.get_core_engine_for_request(request)].add_request(
            request
        )

    def abort_requests(self, request_ids: list[str]) -> None:
        """Route each abort to the replica actually holding it."""
        by_engine: dict[int, list[str]] = {}
        for request_id in request_ids:
            index = self.request_to_engine.get(request_id)
            if index is None:
                continue
            by_engine.setdefault(index, []).append(request_id)
        for index, ids in by_engine.items():
            self.engine_cores[index].abort_requests(ids)
            for request_id in ids:
                self._release(request_id)

    def get_output(self) -> dict[int, EngineCoreOutputs]:
        """Step every replica that has work, and merge what they produced.

        One call is one *round*: the replicas run concurrently on separate devices,
        so a round costs the slowest replica's step rather than the sum. Their clocks
        already reflect that, since each spends only its own duration.
        """
        merged: dict[int, EngineCoreOutputs] = {}
        if self.lockstep:
            return self._lockstep_round(merged)
        for engine_core in self.engine_cores:
            if not engine_core.has_requests():
                continue
            outputs, _ = engine_core.step()
            self._merge(merged, outputs)
        return merged

    def _lockstep_round(
        self, merged: dict[int, EngineCoreOutputs]
    ) -> dict[int, EngineCoreOutputs]:
        """One round in which every replica steps, or none does. R13.4.

        A replica with work takes its step; a replica without one runs a dummy
        forward pass, because the collective needs it present.

        **The drain tail is deliberately not modeled.** Upstream only learns that
        every replica is finished at a periodic all-reduce -- every 32 steps in
        `_has_global_unfinished_reqs` -- so on real hardware the group keeps running
        dummy steps for up to 31 rounds after the last request completes. That is real
        device time, but it delays no request: charging it here would inflate the
        latency of whichever request happened to finish last, which is the wrong
        number to move. It costs utilisation, not latency, and utilisation is not a
        thing this simulator reports.
        """
        local_work = [engine.has_requests() for engine in self.engine_cores]
        if not any(local_work):
            return merged

        self.lockstep_rounds += 1
        for engine_core, has_work in zip(self.engine_cores, local_work, strict=True):
            if has_work:
                outputs, _ = engine_core.step()
                self._merge(merged, outputs)
            else:
                engine_core.execute_dummy_batch()
        return merged

    async def get_output_async(self) -> dict[int, EngineCoreOutputs]:
        merged: dict[int, EngineCoreOutputs] = {}
        if self.lockstep:
            return await self._lockstep_round_async(merged)
        for engine_core in self.engine_cores:
            if not engine_core.has_requests():
                continue
            outputs, _ = await engine_core.step_async()
            self._merge(merged, outputs)
        return merged

    async def _lockstep_round_async(
        self, merged: dict[int, EngineCoreOutputs]
    ) -> dict[int, EngineCoreOutputs]:
        """`_lockstep_round`, awaiting each step. R13.4.

        The async twin exists because `pvllm serve` never takes the synchronous path,
        and lockstep landing on only one of the two meant the HTTP surface -- the one
        an operator actually benchmarks -- reported an idle EP replica as free while
        the offline path charged it. Every stats field said `lockstep: True` either
        way, so nothing gave the discrepancy away.
        """
        local_work = [engine.has_requests() for engine in self.engine_cores]
        if not any(local_work):
            return merged

        self.lockstep_rounds += 1
        for engine_core, has_work in zip(self.engine_cores, local_work, strict=True):
            if has_work:
                outputs, _ = await engine_core.step_async()
                self._merge(merged, outputs)
            else:
                await engine_core.execute_dummy_batch_async()
        return merged

    def _merge(
        self,
        merged: dict[int, EngineCoreOutputs],
        outputs: dict[int, EngineCoreOutputs],
    ) -> None:
        """Fold one replica's outputs into the round's, keyed by client index."""
        for client_index, engine_outputs in outputs.items():
            for output in engine_outputs.outputs:
                if output.finished:
                    self._release(output.request_id)
            existing = merged.get(client_index)
            if existing is None:
                merged[client_index] = engine_outputs
            else:
                existing.outputs.extend(engine_outputs.outputs)
                # The later timestamp, so a frontend dating a batch from it never
                # sees time run backwards across replicas.
                existing.timestamp = max(existing.timestamp, engine_outputs.timestamp)

    def _release(self, request_id: str) -> None:
        index = self.request_to_engine.pop(request_id, None)
        if index is not None:
            self.engine_inflight[index] = max(0, self.engine_inflight[index] - 1)

    def has_requests(self) -> bool:
        return any(engine.has_requests() for engine in self.engine_cores)

    def get_num_unfinished_requests(self) -> int:
        return sum(engine.get_num_unfinished_requests() for engine in self.engine_cores)

    def reset_prefix_cache(self) -> bool:
        """Reset every replica's cache. R6.10.

        Every replica is reset before the answer is computed. `all()` over a
        generator short-circuits, which would wipe the replicas before the first
        refusal, skip the ones after it, and report failure for the whole deployment
        -- leaving an operator told the cache was untouched with half of it gone.
        """
        results = [engine.reset_prefix_cache() for engine in self.engine_cores]
        return all(results)

    def make_stats(self) -> dict[str, Any]:
        """The deployment's numbers, aggregated the way each one means something.

        Counts and totals sum. KV usage is averaged, because it is already a
        fraction, and the deployment being "half full" is what a plan reads. Elapsed
        time is the *maximum*: the replicas ran concurrently, so the deployment's
        clock is the slowest of them and never their sum.
        """
        per_engine = [engine.make_stats() for engine in self.engine_cores]
        summed = (
            "num_running_reqs",
            "num_waiting_reqs",
            "num_preemptions",
            "prefix_cache_queries",
            "prefix_cache_hits",
            "num_draft_tokens",
            "num_accepted_tokens",
            "mm_cache_queries",
            "mm_cache_hits",
        )
        stats: dict[str, Any] = dict(per_engine[0])
        for key in summed:
            if key in stats:
                stats[key] = sum(engine.get(key, 0) for engine in per_engine)
        # R17.2. *Not* summed. The KV connector's store is a process-global keyed by
        # name, so every replica's connector resolves to the same object and reads
        # the same counter -- summing N identical readings of one shared counter
        # reported N times the transfers that happened, straight into
        # `vllm:external_prefix_cache_*` (C6). One reading is the deployment's.
        for key in ("external_prefix_cache_queries", "external_prefix_cache_hits"):
            if key in stats:
                stats[key] = per_engine[0].get(key, 0)
        stats["kv_cache_usage"] = sum(
            float(engine.get("kv_cache_usage", 0.0)) for engine in per_engine
        ) / len(per_engine)
        stats["step_index"] = max(
            int(engine.get("step_index", 0)) for engine in per_engine
        )
        stats["engine_step"] = stats["step_index"]
        stats["elapsed"] = max(
            float(engine.get("elapsed", 0.0)) for engine in per_engine
        )
        stats["data_parallel_size"] = self.data_parallel_size
        # R13.4. What the replicas spent keeping the collective whole rather than
        # serving anyone. Surfaced per replica because the imbalance is the point: a
        # deployment can be at full device utilisation and near-zero goodput, and
        # upstream's metrics do not say so anywhere.
        stats["lockstep"] = self.lockstep
        stats["num_dummy_steps"] = sum(
            engine.num_dummy_steps for engine in self.engine_cores
        )
        stats["dummy_step_seconds"] = sum(
            engine.dummy_step_seconds for engine in self.engine_cores
        )
        stats["lockstep_rounds"] = self.lockstep_rounds
        stats["per_engine_dummy_steps"] = [
            engine.num_dummy_steps for engine in self.engine_cores
        ]
        #: Per replica, so an imbalance is visible rather than averaged away -- which
        #: is the failure mode a DP experiment is looking for.
        stats["per_engine_running"] = [
            int(engine["num_running_reqs"]) for engine in per_engine
        ]
        stats["per_engine_waiting"] = [
            int(engine["num_waiting_reqs"]) for engine in per_engine
        ]
        return stats

    @property
    def clock_time(self) -> float:
        """The deployment's time: the furthest any replica has got.

        The replicas run concurrently, so this is a maximum and never a sum. Taking
        the maximum also keeps the frontend's stamps monotonic -- a request routed to
        a lagging replica must not be dated before one that finished earlier.
        """
        return max(engine.clock.time() for engine in self.engine_cores)

    @property
    def engine_core(self) -> EngineCore:
        """Replica 0, for the introspection paths that assume a single engine.

        Named rather than silently aliased: anything reaching through this on a DP
        deployment is seeing one replica's state, not the deployment's.
        """
        return self.engine_cores[0]

    def shutdown(self) -> None:
        for engine in self.engine_cores:
            engine.shutdown()


__all__ = ["DPInprocClient"]
