"""The synchronous engine.

Upstream: vllm/v1/engine/llm_engine.py
Tier: B

What the offline `LLM` class drives (R2.1). One `step()` per call, outputs returned as
they finish.
"""

from __future__ import annotations

from typing import Any

from pvllm.config import VllmConfig
from pvllm.engine.arg_utils import EngineArgs
from pvllm.logger import init_logger
from pvllm.outputs import RequestOutput
from pvllm.sampling_params import SamplingParams
from pvllm.tokenizers import get_tokenizer
from pvllm.tokenizers.protocol import TokenizerLike
from pvllm.v1.engine.core_client import EngineCoreClient, InprocClient
from pvllm.v1.engine.input_processor import InputProcessor
from pvllm.v1.engine.output_processor import OutputProcessor

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
        self.engine_core: EngineCoreClient = EngineCoreClient.make_client(
            vllm_config, log_stats=log_stats
        )

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> LLMEngine:
        return cls(engine_args.create_engine_config())

    # --- requests ------------------------------------------------------------

    def add_request(
        self,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        priority: int = 0,
    ) -> None:
        engine_request = self.input_processor.process_inputs(
            request_id, prompt, sampling_params, priority=priority
        )
        # The arrival time comes back from the core, which stamped it (R19.1).
        self.engine_core.add_request(engine_request)
        assert isinstance(self.engine_core, InprocClient)
        self.output_processor.add_request(
            request_id=request_id,
            prompt=prompt if isinstance(prompt, str) else None,
            prompt_token_ids=engine_request.prompt_token_ids or [],
            sampling_params=sampling_params,
            arrival_time=self.engine_core.clock_time,
        )

    def abort_request(self, request_ids: list[str]) -> None:
        self.engine_core.abort_requests(request_ids)
        self.output_processor.abort_requests(request_ids)

    def step(self) -> list[RequestOutput]:
        """Run one engine step and return whatever finished or advanced."""
        engine_core_outputs = self.engine_core.get_output()
        if not engine_core_outputs:
            return []

        outputs: list[RequestOutput] = []
        for client_outputs in engine_core_outputs.values():
            outputs.extend(
                self.output_processor.process_outputs(client_outputs.outputs)
            )

        # R11.5: a stop *string* is invisible to the scheduler, so requests the
        # frontend just ended have to be aborted in the core -- otherwise they keep
        # generating and holding blocks.
        stopped = [
            output.request_id
            for output in outputs
            if output.finished
            and output.request_id not in self.output_processor.request_states
        ]
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
        assert isinstance(self.engine_core, InprocClient)
        return self.engine_core.make_stats()

    def reset_prefix_cache(self) -> bool:
        assert isinstance(self.engine_core, InprocClient)
        return self.engine_core.reset_prefix_cache()

    def shutdown(self) -> None:
        self.engine_core.shutdown()

    def __enter__(self) -> LLMEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
