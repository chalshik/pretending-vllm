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
from pvllm.tokenizers import get_tokenizer
from pvllm.tracing import TraceSink
from pvllm.v1.core.sched.output import SchedulerOutput
from pvllm.v1.core.sched.scheduler import Scheduler
from pvllm.v1.engine import (
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from pvllm.v1.executor.abstract import Executor
from pvllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
)
from pvllm.v1.outputs import ModelRunnerOutput
from pvllm.v1.request import Request, RequestStatus
from pvllm.v1.structured_output import StructuredOutputManager

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

        # R15. Owned here rather than by the scheduler: it compiles on a thread
        # pool and holds a tokenizer, neither of which belongs to a component whose
        # job is deciding what runs next.
        self.structured_output_manager = StructuredOutputManager(vllm_config)
        self.structured_output_manager.set_tokenizer(
            get_tokenizer(
                vllm_config.model_config.tokenizer or vllm_config.model_config.model,
                tokenizer_mode=vllm_config.model_config.tokenizer_mode,
                vocab_size=vllm_config.model_config.get_vocab_size(),
            )
        )

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

        # R15. Started before the request joins the queue, so compilation overlaps
        # the wait rather than beginning when the scheduler first looks at it.
        if req.use_structured_output:
            self.structured_output_manager.grammar_init(req)

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
        planned = self._plan_step()
        if planned is None:
            return {}, False
        return self._finish_step(planned, self.executor.execute_model(planned))

    async def step_async(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """One engine step, yielding to the event loop while modeled time passes.

        Identical to `step` except for the line that spends the step's duration --
        the two share `_plan_step` and `_finish_step` so a clock mode can never
        change what the engine decided, only how long the process spent deciding it.
        """
        planned = self._plan_step()
        if planned is None:
            return {}, False
        return self._finish_step(
            planned, await self.executor.execute_model_async(planned)
        )

    def _plan_step(self) -> SchedulerOutput | None:
        """Schedule and date the admissions. `None` when there is nothing to do."""
        if not self.scheduler.has_requests():
            return None

        scheduler_output = self.scheduler.schedule()

        # R19.1: the scheduler decides *what* was admitted, this object decides
        # *when*. Stamped before the model runs so the queue wait ends where the
        # request's own work begins, not after the whole batch's forward pass.
        scheduled_at = self.clock.time()
        for request in self.scheduler.take_newly_scheduled():
            request.record_event(EngineCoreEventType.SCHEDULED, scheduled_at)

        # R15. Computed between scheduling and execution, as upstream: the mask
        # depends on how far each grammar has advanced, which is only settled once
        # this step's batch is known. In `_plan_step` rather than in each of `step`
        # and `step_async`, for the same reason those two share this method at all.
        if scheduler_output.has_structured_output_requests:
            scheduler_output.grammar_bitmask = (
                self.structured_output_manager.grammar_bitmask(
                    self.scheduler.requests,
                    scheduler_output.structured_output_request_ids,
                )
            )
        return scheduler_output

    def _finish_grammar_failures(self) -> dict[int, list[EngineCoreOutput]]:
        """End requests whose grammar could not compile. R15.

        With FINISHED_ERROR rather than by raising: a malformed schema is a client
        error belonging to one request, and taking down the engine step that noticed
        it would let one bad request deny service to every other.

        Returns an output per failed request, keyed by client, because ending it in
        the scheduler is only half the job. The frontend is still holding the
        request's queue, and a caller waiting on it would wait forever -- the failure
        has to travel the same path a completion does.
        """
        failed = self.scheduler.take_grammar_compile_errors()
        outputs: dict[int, list[EngineCoreOutput]] = {}
        if not failed:
            return outputs

        for request_id in sorted(failed):
            request = self.scheduler.requests.get(request_id)
            reason = "the request's grammar failed to compile"
            client_index = 0
            if request is not None:
                client_index = request.client_index
                if request.structured_output_request is not None:
                    error = request.structured_output_request.grammar
                    if isinstance(error, Exception):
                        reason = f"{type(error).__name__}: {error}"
            logger.warning("request %s failed: %s", request_id, reason)
            self.trace.emit(
                "request",
                t=self.clock.time(),
                request_id=request_id,
                event="finished",
                finish_reason="error",
                error=reason,
            )
            outputs.setdefault(client_index, []).append(
                EngineCoreOutput(
                    request_id=request_id,
                    new_token_ids=[],
                    finish_reason=FinishReason.ERROR,
                    stop_reason=reason,
                )
            )
        self.scheduler.finish_requests(sorted(failed), RequestStatus.FINISHED_ERROR)
        return outputs

    def _finish_step(
        self, scheduler_output: SchedulerOutput, model_output: ModelRunnerOutput
    ) -> tuple[dict[int, EngineCoreOutputs], bool]:
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

        # After the step, so a grammar that failed while this step ran is dealt with
        # now rather than a step later. Merged into this step's outputs so the
        # frontend hears about it on the same pass -- a failure delivered nowhere is
        # a request that hangs.
        for client_index, failures in self._finish_grammar_failures().items():
            existing = engine_core_outputs.get(client_index)
            if existing is None:
                engine_core_outputs[client_index] = EngineCoreOutputs(
                    engine_index=client_index, outputs=failures, timestamp=timestamp
                )
            else:
                existing.outputs.extend(failures)

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
        self.structured_output_manager.shutdown()
        self.scheduler.shutdown()
        self.executor.shutdown()
        self.trace.close()

    def __repr__(self) -> str:
        return (
            f"EngineCore(step={self.step_index}, "
            f"num_blocks={self.kv_cache_config.num_blocks}, "
            f"clock={self.clock})"
        )
