"""Turning engine core output into RequestOutput.

Upstream: vllm/v1/engine/output_processor.py
Tier: B

Holds the per-request detokenizer state and assembles what the client sees. The
scheduler decides *whether* a request stopped on a token; this decides whether it
stopped on a *string*, because that needs text (R11.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pvllm.logger import init_logger
from pvllm.outputs import CompletionOutput, RequestOutput
from pvllm.sampling_params import RequestOutputKind, SamplingParams
from pvllm.tokenizers.protocol import TokenizerLike
from pvllm.v1.engine import EngineCoreOutput, FinishReason
from pvllm.v1.engine.detokenizer import (
    IncrementalDetokenizer,
    SlowIncrementalDetokenizer,
)

logger = init_logger(__name__)


@dataclass
class RequestState:
    """Frontend state for one in-flight request."""

    request_id: str
    prompt: str | None
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    detokenizer: IncrementalDetokenizer
    #: Wall-clock-free: stamped from the engine core's clock (R19.1).
    arrival_time: float
    is_finished: bool = False
    num_cached_tokens: int = 0
    stats: dict[str, float] = field(default_factory=dict)


class OutputProcessor:
    """Assembles `RequestOutput`s from engine core output."""

    def __init__(self, tokenizer: TokenizerLike, log_stats: bool = False) -> None:
        self.tokenizer = tokenizer
        self.log_stats = log_stats
        self.request_states: dict[str, RequestState] = {}

    def add_request(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> None:
        if request_id in self.request_states:
            raise ValueError(f"request {request_id!r} already exists")
        self.request_states[request_id] = RequestState(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            detokenizer=SlowIncrementalDetokenizer(
                self.tokenizer, list(prompt_token_ids), sampling_params
            ),
            arrival_time=arrival_time,
        )

    def abort_requests(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self.request_states.pop(request_id, None)

    def process_outputs(
        self, engine_core_outputs: list[EngineCoreOutput]
    ) -> list[RequestOutput]:
        """Fold one step's output into per-request text."""
        outputs: list[RequestOutput] = []

        for engine_output in engine_core_outputs:
            state = self.request_states.get(engine_output.request_id)
            if state is None:
                # Aborted between the step being scheduled and its output arriving.
                continue

            finish_reason = engine_output.finish_reason
            stop_terminated = finish_reason is FinishReason.STOP

            stop_string = state.detokenizer.update(
                list(engine_output.new_token_ids), stop_terminated
            )
            if stop_string is not None and finish_reason is None:
                # R11.5: a stop *string* ends the request here, in the frontend --
                # the scheduler cannot see it, because it has no text.
                finish_reason = FinishReason.STOP

            finished = finish_reason is not None
            state.num_cached_tokens = engine_output.num_cached_tokens

            delta = state.sampling_params.output_kind == RequestOutputKind.DELTA
            text = state.detokenizer.get_next_output_text(finished, delta)

            token_ids = (
                list(engine_output.new_token_ids)
                if delta
                else list(state.detokenizer.output_token_ids)
            )

            outputs.append(
                RequestOutput(
                    request_id=engine_output.request_id,
                    prompt=state.prompt,
                    prompt_token_ids=state.prompt_token_ids,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=text,
                            token_ids=token_ids,
                            finish_reason=str(finish_reason) if finished else None,
                            stop_reason=stop_string
                            if stop_string is not None
                            else engine_output.stop_reason,
                        )
                    ],
                    finished=finished,
                    num_cached_tokens=state.num_cached_tokens,
                )
            )

            if finished:
                state.is_finished = True
                self.request_states.pop(engine_output.request_id, None)

        return outputs

    def get_stop_string_request_ids(self) -> list[str]:
        """Requests the frontend stopped that the engine core still thinks are live.

        A stop string is invisible to the scheduler, so the core has to be told to
        abort them -- otherwise they keep generating and holding blocks.
        """
        return [
            request_id
            for request_id, state in self.request_states.items()
            if state.is_finished
        ]

    @property
    def num_requests(self) -> int:
        return len(self.request_states)
