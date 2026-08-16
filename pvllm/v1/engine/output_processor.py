"""Turning engine core output into RequestOutput.

Upstream: vllm/v1/engine/output_processor.py
Tier: B

Holds the per-request detokenizer state and assembles what the client sees. The
scheduler decides *whether* a request stopped on a token; this decides whether it
stopped on a *string*, because that needs text (R11.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pvllm.logger import init_logger
from pvllm.outputs import (
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    RequestOutput,
)
from pvllm.sampling_params import RequestOutputKind, SamplingParams
from pvllm.tokenizers.protocol import TokenizerLike
from pvllm.v1.engine import EngineCoreEventType, EngineCoreOutput, FinishReason
from pvllm.v1.engine.detokenizer import (
    IncrementalDetokenizer,
    SlowIncrementalDetokenizer,
)
from pvllm.v1.metrics.stats import FinishedRequestStats, IterationStats

logger = init_logger(__name__)


@dataclass
class RequestState:
    """Frontend state for one in-flight request."""

    request_id: str
    prompt: str | None
    prompt_token_ids: list[int]
    sampling_params: SamplingParams | None
    detokenizer: IncrementalDetokenizer
    #: Wall-clock-free: stamped from the engine core's clock (R19.1).
    arrival_time: float
    is_finished: bool = False
    num_cached_tokens: int = 0

    #: R11.7. The client request this is one of `n` children of, and which of them.
    #: `None` for the overwhelming majority of requests, where `n == 1` and the
    #: engine request *is* the client request.
    parent_request: Any = None
    index: int = 0

    #: R2.2. Set for an embedding request, which produces a vector and no text --
    #: so it has no detokenizer state worth updating and no stop conditions.
    pooling_params: Any = None

    # --- timing (R12.1) ---------------------------------------------------
    # Every stamp comes from the engine core's clock, so under a virtual clock
    # these are modeled durations -- which the metric help strings say.
    #: When the first output token arrived. Zero until it does.
    first_token_time: float = 0.0
    #: When the most recent output token arrived, for inter-token latency.
    last_token_time: float = 0.0
    num_generation_tokens: int = 0
    #: When the scheduler first admitted this request, from its SCHEDULED event.
    #: Zero until the core reports one, which is what separates the queue wait from
    #: the prefill -- without it the two are indistinguishable inside TTFT.
    scheduled_time: float = 0.0

    def record_tokens(self, now: float, count: int) -> list[float]:
        """Fold in `count` new tokens. Returns inter-token latencies to observe.

        A step can deliver more than one token (speculative decoding, or a prefill
        that completes and samples in the same step), and the *observed* gap covers
        all of them at once. Dividing by the count would report a per-token latency
        the system never actually exhibited, so the whole gap is attributed once --
        which is what upstream's histogram measures too.
        """
        if count <= 0:
            return []
        latencies: list[float] = []
        if self.first_token_time == 0.0:
            self.first_token_time = now
        elif self.last_token_time:
            latencies.append(now - self.last_token_time)
        self.last_token_time = now
        self.num_generation_tokens += count
        return latencies

    @property
    def time_to_first_token(self) -> float:
        return max(0.0, self.first_token_time - self.arrival_time)

    def finished_stats(self, now: float, finish_reason: str) -> FinishedRequestStats:
        """What this request contributes to the histograms once it ends."""
        e2e = max(0.0, now - self.arrival_time)
        ttft = self.time_to_first_token
        # Decode time is everything after the first token; a request that produced
        # only one token has none, and reporting its whole lifetime as decode would
        # skew the decode histogram toward prefill cost.
        decode = max(0.0, now - self.first_token_time) if self.first_token_time else 0.0
        num_after_first = max(0, self.num_generation_tokens - 1)
        # TTFT splits into the wait for admission and the prefill that followed it.
        # A request that queued for 200ms and prefilled in 5ms has the same TTFT as
        # one that prefilled for 205ms, and they call for opposite responses -- more
        # capacity versus a smaller batch. The split is only available because the
        # core dates the scheduler's admissions (EngineCoreEventType.SCHEDULED).
        queue = (
            max(0.0, self.scheduled_time - self.arrival_time)
            if (self.scheduled_time)
            else 0.0
        )
        return FinishedRequestStats(
            finish_reason=finish_reason,
            e2e_latency=e2e,
            queue_time=queue,
            prefill_time=max(0.0, ttft - queue),
            # From admission to the last token, not from arrival: the metric's help
            # text says "time spent in RUNNING phase", and upstream measures
            # `last_token_ts - scheduled_ts`. Including the wait would make
            # queue + inference exceed the end-to-end latency, so a dashboard
            # stacking the phases would report more time than the request took.
            inference_time=max(0.0, e2e - queue),
            decode_time=decode,
            time_to_first_token=ttft,
            time_per_output_token=(decode / num_after_first)
            if num_after_first
            else 0.0,
            num_prompt_tokens=len(self.prompt_token_ids),
            num_generation_tokens=self.num_generation_tokens,
            num_cached_tokens=self.num_cached_tokens,
            # R2.2. A pooling request generates nothing, so neither parameter
            # exists for it -- reported as the histogram's absent case rather than
            # as a zero, which would be a real request that asked for no tokens.
            max_tokens_param=(
                self.sampling_params.max_tokens
                if self.sampling_params is not None
                else None
            ),
            n_param=self.sampling_params.n if self.sampling_params is not None else 1,
        )


class OutputProcessor:
    """Assembles `RequestOutput`s from engine core output."""

    def __init__(self, tokenizer: TokenizerLike, log_stats: bool = False) -> None:
        self.tokenizer = tokenizer
        self.log_stats = log_stats
        self.request_states: dict[str, RequestState] = {}
        #: Requests the frontend ended on a stop string. The scheduler cannot see
        #: those -- it has no text -- so the core must be told to abort them.
        self.stopped_by_string: list[str] = []

    def add_request(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None,
        arrival_time: float,
        parent_request: Any = None,
        index: int = 0,
        pooling_params: Any = None,
    ) -> None:
        if request_id in self.request_states:
            raise ValueError(f"request {request_id!r} already exists")
        self.request_states[request_id] = RequestState(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            detokenizer=SlowIncrementalDetokenizer(
                self.tokenizer,
                list(prompt_token_ids),
                sampling_params if sampling_params is not None else SamplingParams(),
            ),
            arrival_time=arrival_time,
            parent_request=parent_request,
            index=index,
            pooling_params=pooling_params,
        )

    def abort_requests(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.request_states.pop(request_id, None)

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        now: float = 0.0,
        iteration_stats: IterationStats | None = None,
    ) -> list[RequestOutput | PoolingRequestOutput]:
        """Fold one step's output into per-request text, and time it.

        `now` is the engine core's clock (R19.1); the frontend has none of its own.
        """
        outputs: list[RequestOutput | PoolingRequestOutput] = []

        for engine_output in engine_core_outputs:
            state = self.request_states.get(engine_output.request_id)
            if state is None:
                # Aborted between the step being scheduled and its output arriving.
                continue

            # R2.2. An embedding request has no text: no detokenizer to advance, no
            # stop strings to look for, and one output rather than a stream of them.
            if state.pooling_params is not None:
                if engine_output.pooling_output is None:
                    continue
                outputs.append(
                    PoolingRequestOutput(
                        request_id=engine_output.request_id,
                        outputs=PoolingOutput(data=engine_output.pooling_output),
                        prompt_token_ids=state.prompt_token_ids,
                        finished=True,
                    )
                )
                self._retire(
                    state,
                    engine_output.request_id,
                    now,
                    engine_output.finish_reason,
                    iteration_stats,
                )
                continue

            finish_reason = engine_output.finish_reason
            stop_terminated = finish_reason is FinishReason.STOP

            stop_string = state.detokenizer.update(
                list(engine_output.new_token_ids), stop_terminated
            )
            if stop_string is not None and finish_reason is None:
                # R11.5: a stop *string* ends the request here, in the frontend --
                # the scheduler cannot see it, because it has no text. Recorded so
                # the caller aborts exactly these, and not every finished request.
                finish_reason = FinishReason.STOP
                self.stopped_by_string.append(engine_output.request_id)

            finished = finish_reason is not None
            state.num_cached_tokens = engine_output.num_cached_tokens

            # QUEUED is the core's own arrival stamp, and it supersedes the
            # frontend's. In process the two are equal; over a process boundary the
            # frontend only knows the *last step's* time when it submits, so its
            # estimate is stale by up to a step and every queue time computed from
            # it would be inflated by that much. Taking the core's stamp makes the
            # timing identical in both transports (R19.1).
            #
            # The first SCHEDULED event ends the queue wait. Only the first: a
            # preempted request is admitted again later, and taking the newer stamp
            # would report a queue time longer than the request's whole life.
            for event in engine_output.events or ():
                if event.type is EngineCoreEventType.QUEUED:
                    state.arrival_time = event.timestamp
                elif (
                    event.type is EngineCoreEventType.SCHEDULED
                    and not state.scheduled_time
                ):
                    state.scheduled_time = event.timestamp

            num_new = len(engine_output.new_token_ids)
            latencies = state.record_tokens(now, num_new)
            if iteration_stats is not None:
                iteration_stats.num_generation_tokens += num_new
                iteration_stats.inter_token_latencies.extend(latencies)
                if num_new and state.num_generation_tokens == num_new:
                    iteration_stats.time_to_first_tokens.append(
                        state.time_to_first_token
                    )

            assert state.sampling_params is not None
            delta = state.sampling_params.output_kind == RequestOutputKind.DELTA
            text = state.detokenizer.get_next_output_text(finished, delta)

            token_ids = (
                list(engine_output.new_token_ids)
                if delta
                else list(state.detokenizer.output_token_ids)
            )

            completion = CompletionOutput(
                index=state.index,
                text=text,
                token_ids=token_ids,
                finish_reason=str(finish_reason) if finished else None,
                stop_reason=stop_string
                if stop_string is not None
                else engine_output.stop_reason,
            )

            # R11.7. A child of an `n > 1` request is reported under its *parent's*
            # id, with the siblings gathered beside it. Streaming passes each chunk
            # through as it arrives; a non-streaming request waits for the last
            # child so the response carries all `n` in index order, whatever order
            # they actually finished in.
            parent = state.parent_request
            if parent is None:
                completions = [completion]
                report_id = engine_output.request_id
                report_finished = finished
            else:
                completions, report_finished = parent.collect(
                    engine_output.request_id, completion
                )
                report_id = parent.request_id
                if not completions and not report_finished:
                    if finished:
                        self._retire(
                            state,
                            engine_output.request_id,
                            now,
                            finish_reason,
                            iteration_stats,
                        )
                    continue

            outputs.append(
                RequestOutput(
                    request_id=report_id,
                    prompt=state.prompt,
                    prompt_token_ids=state.prompt_token_ids,
                    outputs=completions,
                    finished=report_finished,
                    num_cached_tokens=state.num_cached_tokens,
                )
            )

            if finished:
                self._retire(
                    state,
                    engine_output.request_id,
                    now,
                    finish_reason,
                    iteration_stats,
                )

        return outputs

    def _retire(
        self,
        state: RequestState,
        request_id: str,
        now: float,
        finish_reason: Any,
        iteration_stats: IterationStats | None,
    ) -> None:
        """Drop a finished request's frontend state and record what it contributed."""
        state.is_finished = True
        if iteration_stats is not None:
            iteration_stats.finished_requests.append(
                state.finished_stats(now, str(finish_reason))
            )
        self.request_states.pop(request_id, None)

    def take_stopped_by_string(self) -> list[str]:
        """Drain the requests the frontend ended on a stop string.

        The engine core still believes these are running, so it must be told to
        abort them -- otherwise they keep generating and holding blocks. Taken
        rather than read so a request is aborted once.
        """
        stopped, self.stopped_by_string = self.stopped_by_string, []
        return stopped

    @property
    def num_requests(self) -> int:
        return len(self.request_states)
