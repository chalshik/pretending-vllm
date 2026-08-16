"""The synchronous engine.

Upstream: vllm/v1/engine/llm_engine.py
Tier: B

What the offline `LLM` class drives (R2.1). One `step()` per call, outputs returned as
they finish.
"""

from __future__ import annotations

from typing import Any

from pvllm import envs
from pvllm.config import VllmConfig
from pvllm.engine.arg_utils import EngineArgs
from pvllm.logger import init_logger
from pvllm.outputs import PoolingRequestOutput, RequestOutput
from pvllm.pooling_params import PoolingParams
from pvllm.sampling_params import SamplingParams
from pvllm.tokenizers import get_tokenizer
from pvllm.tokenizers.protocol import TokenizerLike
from pvllm.v1.engine.core_client import EngineCoreClient
from pvllm.v1.engine.input_processor import InputProcessor
from pvllm.v1.engine.output_processor import OutputProcessor
from pvllm.v1.engine.parallel_sampling import ParentRequest
from pvllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)


class LLMEngine:
    """A synchronous engine: add requests, call `step` until they finish."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        log_stats: bool = True,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.log_stats = log_stats

        self.tokenizer: TokenizerLike = get_tokenizer(
            self.model_config.tokenizer or self.model_config.model,
            tokenizer_mode=self.model_config.tokenizer_mode,
            vocab_size=self.model_config.get_vocab_size(),
        )
        self.input_processor = InputProcessor(vllm_config, self.tokenizer)
        self.output_processor = OutputProcessor(self.tokenizer, log_stats=log_stats)
        # R4.2. Off unless asked for, unlike upstream -- see
        # PVLLM_ENABLE_V1_MULTIPROCESSING in pvllm/envs.py for why the default is
        # inverted here.
        self.engine_core: EngineCoreClient = EngineCoreClient.make_client(
            vllm_config,
            multiprocess_mode=envs.PVLLM_ENABLE_V1_MULTIPROCESSING,
            log_stats=log_stats,
        )
        #: The most recent step's stats, for whoever scrapes them.
        self.last_iteration_stats = IterationStats()

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> LLMEngine:
        return cls(engine_args.create_engine_config())

    # --- requests ------------------------------------------------------------

    def add_request(
        self,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        priority: int = 0,
        lora_request: Any = None,
        mm_features: list[Any] | None = None,
        pooling_params: PoolingParams | None = None,
    ) -> None:
        if pooling_params is not None:
            # R2.2. No fan-out and no sampling: an embedding request prefills once
            # and returns a vector.
            engine_request = self.input_processor.process_inputs(
                request_id, prompt, priority=priority, pooling_params=pooling_params
            )
            self.engine_core.add_request(engine_request)
            self.output_processor.add_request(
                request_id=request_id,
                prompt=prompt if isinstance(prompt, str) else None,
                prompt_token_ids=engine_request.prompt_token_ids or [],
                sampling_params=None,
                arrival_time=self.engine_core.clock_time,
                pooling_params=pooling_params,
            )
            return

        assert sampling_params is not None, (
            "add_request needs one of sampling_params or pooling_params"
        )

        # R11.7. `n > 1` fans out here, in the frontend, exactly as upstream does:
        # the engine core has no notion of `n`, and four completions of one prompt
        # are four requests that share a prompt. They queue, preempt, and share KV
        # through the ordinary prefix cache independently -- which is the behaviour a
        # capacity plan needs to see, not an implementation detail.
        parent = (
            ParentRequest(request_id, sampling_params)
            if sampling_params.n > 1
            else None
        )
        for index in range(sampling_params.n):
            child_id, child_params = (
                parent.child_info(index)
                if parent is not None
                else (request_id, sampling_params)
            )
            engine_request = self.input_processor.process_inputs(
                child_id,
                prompt,
                child_params,
                priority=priority,
                lora_request=lora_request,
                mm_features=mm_features,
            )
            # The arrival time comes back from the core, which stamped it (R19.1).
            self.engine_core.add_request(engine_request)
            self.output_processor.add_request(
                request_id=child_id,
                prompt=prompt if isinstance(prompt, str) else None,
                prompt_token_ids=engine_request.prompt_token_ids or [],
                sampling_params=child_params,
                arrival_time=self.engine_core.clock_time,
                parent_request=parent,
                index=index,
            )

    def abort_request(self, request_ids: list[str]) -> None:
        # R11.7: a client id may name `n` engine requests, or none.
        to_abort = self.output_processor.abort_requests(request_ids)
        if to_abort:
            self.engine_core.abort_requests(to_abort)

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        """Run one engine step and return whatever finished or advanced."""
        engine_core_outputs = self.engine_core.get_output()
        if not engine_core_outputs:
            # Cleared, not left alone. A caller accumulating `finished_requests`
            # across steps would otherwise re-count the last productive step's
            # finishers on every barren step -- which happens whenever a step is all
            # chunked prefill -- inflating every derived throughput and percentile.
            self.last_iteration_stats = IterationStats()
            return []

        now = self.engine_core.clock_time
        iteration_stats = IterationStats()

        outputs: list[RequestOutput | PoolingRequestOutput] = []
        for client_outputs in engine_core_outputs.values():
            outputs.extend(
                self.output_processor.process_outputs(
                    client_outputs.outputs, now, iteration_stats
                )
            )
        self.last_iteration_stats = iteration_stats

        # R11.5: only requests the frontend ended on a stop string need aborting.
        # A request the *scheduler* finished is already gone from its side, and
        # aborting it would emit a spurious lifecycle event and redo the cleanup.
        stopped = self.output_processor.take_stopped_by_string()
        if stopped:
            self.engine_core.abort_requests(stopped)
        return outputs

    def has_unfinished_requests(self) -> bool:
        """Whether any request is still outstanding.

        Deliberately *not* `engine_core.has_requests()`, which also stays true while
        a finished request's id waits to be delivered to the worker. Conflating the
        two makes an aborted request look outstanding forever to a caller looping on
        this.
        """
        return (
            self.engine_core.get_num_unfinished_requests() > 0
            or self.output_processor.num_requests > 0
        )

    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.num_requests

    def make_stats(self) -> dict[str, Any]:
        return self.engine_core.make_stats()

    def reset_prefix_cache(self) -> bool:
        return self.engine_core.reset_prefix_cache()

    def shutdown(self) -> None:
        self.engine_core.shutdown()

    def __enter__(self) -> LLMEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
