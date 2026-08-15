"""The engine core: scheduler, executor, clock, trace.

Upstream: vllm/v1/engine/core.py
Tier: B

R4.1. `step()` schedules, executes, and folds the result back.

**This object owns the clock (R19.1, R4.4), and that ownership is the reason the rest
of the design holds together.** It stamps arrival times, it stamps every output, and
nothing upstream of it reads a clock. The rule matters most in a configuration that
does not exist yet: once the engine core runs in its own process (M3), a frontend that
read its own clock would produce timestamps from a different timeline than the engine's,
and every latency metric would silently become the sum of two unrelated clocks. Enforcing
it now costs nothing; retrofitting it would mean auditing every call site.

The trace is opened here too, for the same reason -- every record needs a timestamp,
and only this object has one to give.
"""

from __future__ import annotations

from typing import Any

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.timebase import Clock
from pvllm.tracing import TraceSink
from pvllm.v1.core.sched.scheduler import Scheduler
from pvllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutputs,
    EngineCoreRequest,
)
from pvllm.v1.executor.abstract import Executor
from pvllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)
from pvllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class EngineCore:
    """Owns the scheduler, the executor, the clock, and the trace."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor] | None = None,
        log_stats: bool = True,
        trace: TraceSink | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.log_stats = log_stats
        sim_config = vllm_config.sim_config

        # R19.1: the clock is created here and nowhere else. It comes from the
        # platform (B2) rather than being constructed directly, so the engine core
        # never learns that its timebase is simulated -- the same reason it does not
        # import a worker class.
        from pvllm.platforms import current_platform

        self.clock: Clock = current_platform.build_clock(
            sim_config.clock_mode, time_scale=sim_config.time_scale
        )
        self.trace: TraceSink = trace or self._open_trace()

        executor_class = executor_class or Executor.get_class(vllm_config)
        self.executor = executor_class(vllm_config, self.clock)

        kv_cache_config = self._initialize_kv_caches()
        self.scheduler = Scheduler(
            vllm_config,
            kv_cache_config,
            log_stats=log_stats,
            trace=self.trace,
        )
        self.kv_cache_config = kv_cache_config

        self.step_index = 0
        self._request_counter = 0

    def _open_trace(self) -> TraceSink:
        from pvllm import UPSTREAM_VERSION
        from pvllm.platforms import current_platform

        return current_platform.build_trace_sink(
            self.vllm_config.sim_config.trace_path,
            seed=self.vllm_config.sim_config.seed,
            clock_mode=self.vllm_config.sim_config.clock_mode,
            upstream_version=UPSTREAM_VERSION,
            config={
                "model": self.vllm_config.model_config.model,
                "model_card": self.vllm_config.model_config.hf_config.name,
                "device_card": self.vllm_config.sim_config.device_card,
                "block_size": self.vllm_config.cache_config.block_size,
                "max_model_len": self.vllm_config.model_config.max_model_len,
                "cost_model": self.vllm_config.sim_config.cost_model_profile,
            },
        )

    def _initialize_kv_caches(self) -> KVCacheConfig:
        """Run the memory model and hand the layout to the workers. R10.3."""
        available = self.executor.determine_available_memory()[0]
        specs: dict[str, KVCacheSpec] = self.executor.get_kv_cache_specs()[0]
        if not specs:
            raise ValueError("the model reported no attention layers")

        # Every layer of a dense model shares a spec, so they collapse into one
        # group. The grouping is computed rather than assumed so hybrid models
        # (R6.7) slot in without changing this.
        by_type: dict[str, list[str]] = {}
        for layer_name, spec in specs.items():
            by_type.setdefault(spec.type_id, []).append(layer_name)

        if len(by_type) > 1:
            raise NotImplementedError(
                f"this model needs {len(by_type)} KV cache groups (hybrid attention); "
                f"multiple groups (requirement R6.7) land in M4"
            )

        layer_names = next(iter(by_type.values()))
        spec = specs[layer_names[0]]
        page_size = spec.page_size_bytes * len(layer_names)
        num_blocks = available // page_size

        kv_cache_config = KVCacheConfig(
            num_blocks=int(num_blocks),
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            ],
        )
        self.executor.initialize_from_config([kv_cache_config])
        self.executor.compile_or_warm_up_model()
        return kv_cache_config

    # --- request lifecycle ---------------------------------------------------

    def add_request(self, request: EngineCoreRequest) -> None:
        """Admit a request, stamping its arrival time. R19.1."""
        arrival_time = self.clock.time()
        req = Request.from_engine_core_request(request, arrival_time=arrival_time)
        if self.log_stats:
            req.record_event(EngineCoreEventType.QUEUED, arrival_time)

        self.scheduler.add_request(req)
        self.trace.emit(
            "request",
            t=arrival_time,
            request_id=req.request_id,
            event="arrived",
            num_prompt_tokens=req.num_prompt_tokens,
            max_tokens=req.max_tokens,
        )

    def abort_requests(self, request_ids: list[str]) -> None:
        """R2.4: blocks come back within one step."""
        for request_id in request_ids:
            self.trace.emit(
                "request",
                t=self.clock.time(),
                request_id=request_id,
                event="aborted",
            )
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    # --- the loop ------------------------------------------------------------

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """One engine step.

        Returns the outputs keyed by client index -- the shape data parallelism needs
        (F7) -- and whether the model actually ran.
        """
        if not self.scheduler.has_requests():
            return {}, False

        scheduler_output = self.scheduler.schedule()

        # R19.1: the scheduler decides *what* was admitted, this object decides
        # *when*. Stamped before the model runs so the queue wait ends where the
        # request's own work begins, not after the whole batch's forward pass.
        scheduled_at = self.clock.time()
        for request in self.scheduler.take_newly_scheduled():
            request.record_event(EngineCoreEventType.SCHEDULED, scheduled_at)

        model_output = self.executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # Stamped here, from the one clock, so a frontend never needs its own.
        timestamp = self.clock.time()
        for outputs in engine_core_outputs.values():
            outputs.timestamp = timestamp

        # R19.3: the scheduler builds the step record; only this object can date it.
        step_record = self.scheduler.take_step_record()
        if step_record is not None:
            self.trace.emit("step", t=timestamp, **step_record)

        # One record per request lifecycle transition, so a trace answers "when did
        # this request finish, and why" without replaying the step stream.
        for outputs in engine_core_outputs.values():
            for output in outputs.outputs:
                if output.finish_reason is not None:
                    self.trace.emit(
                        "request",
                        t=timestamp,
                        request_id=output.request_id,
                        event="finished",
                        finish_reason=str(output.finish_reason),
                        num_cached_tokens=output.num_cached_tokens,
                    )

        self.step_index += 1
        model_executed = scheduler_output.total_num_scheduled_tokens > 0
        return engine_core_outputs, model_executed

    # --- introspection -------------------------------------------------------

    def get_num_unfinished_requests(self) -> int:
        return self.scheduler.get_num_unfinished_requests()

    def has_requests(self) -> bool:
        return self.scheduler.has_requests()

    def reset_prefix_cache(self) -> bool:
        return self.scheduler.reset_prefix_cache()

    def make_stats(self) -> dict[str, Any]:
        stats = self.scheduler.make_stats()
        stats["engine_step"] = self.step_index
        stats["elapsed"] = self.clock.elapsed
        stats["clock_mode"] = self.clock.mode
        # R12.4: a consumer must be able to tell that these are modeled durations.
        stats["durations_are_modeled"] = True
        return stats

    def shutdown(self) -> None:
        """R4.5."""
        self.scheduler.shutdown()
        self.executor.shutdown()
        self.trace.close()

    def __repr__(self) -> str:
        return (
            f"EngineCore(step={self.step_index}, "
            f"num_blocks={self.kv_cache_config.num_blocks}, "
            f"clock={self.clock})"
        )
