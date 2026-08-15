"""The offline entrypoint.

Upstream: vllm/entrypoints/llm.py
Tier: B

R2.1. One of the two compatibility surfaces this project promises (NG4): a product
doing `from pvllm import LLM; llm.generate(...)` gets what `vllm.LLM` would give it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pvllm.engine.arg_utils import EngineArgs
from pvllm.logger import init_logger
from pvllm.outputs import RequestOutput
from pvllm.sampling_params import SamplingParams
from pvllm.v1.engine.llm_engine import LLMEngine

logger = init_logger(__name__)


class LLM:
    """Generate completions offline, without a server.

    Args:
        model: Model name or Hugging Face id. Resolved against the bundled model
            cards; an unknown name is an error rather than a guess.
        **kwargs: Anything `EngineArgs` accepts, including the simulator knobs
            (`device_card`, `clock_mode`, `cost_model_profile`, `seed`).
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        engine_args = EngineArgs(model=model, **kwargs)
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self._request_counter = 0

    def generate(
        self,
        prompts: str | Sequence[str] | Sequence[list[int]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        *,
        use_tqdm: bool = False,
    ) -> list[RequestOutput]:
        """Generate completions.

        Every prompt is submitted before the first step, so they are scheduled as a
        batch -- which is what makes the batching behaviour observable offline
        rather than only under a server.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        # Each element is either text or token ids (R3.3), so the element type is
        # genuinely a union rather than an unresolved Any.
        prompt_list: list[str | list[int]] = [
            item if isinstance(item, str) else list(item) for item in prompts
        ]

        if sampling_params is None:
            params_list = [SamplingParams() for _ in prompt_list]
        elif isinstance(sampling_params, SamplingParams):
            # Cloned per request: the processor resolves `max_tokens` against the
            # prompt length in place, so sharing one object across prompts of
            # different lengths would let the first resolution win for all of them.
            params_list = [sampling_params.clone() for _ in prompt_list]
        else:
            params_list = list(sampling_params)
            if len(params_list) != len(prompt_list):
                raise ValueError(
                    f"got {len(params_list)} sampling params for "
                    f"{len(prompt_list)} prompts"
                )

        request_ids: list[str] = []
        for prompt, params in zip(prompt_list, params_list, strict=True):
            request_id = str(self._request_counter)
            self._request_counter += 1
            request_ids.append(request_id)
            self.llm_engine.add_request(request_id, prompt, params)

        finished: dict[str, RequestOutput] = {}
        while self.llm_engine.has_unfinished_requests():
            for output in self.llm_engine.step():
                if output.finished:
                    finished[output.request_id] = output

        # Returned in submission order, not completion order, so results line up
        # with the prompts the caller passed.
        return [finished[request_id] for request_id in request_ids]

    def chat(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Generate from chat messages. R2.1.

        The chat template is the tokenizer's; with `MockTokenizer` that is a minimal
        stable format rather than any real model's (R3.1's real templates arrive with
        the `realtok` extra).
        """
        conversations = (
            [messages] if messages and isinstance(messages[0], dict) else messages
        )
        prompts = [
            self.llm_engine.tokenizer.apply_chat_template(
                conversation,  # type: ignore[arg-type]
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        return self.generate([str(p) for p in prompts], sampling_params)

    def get_tokenizer(self) -> Any:
        return self.llm_engine.tokenizer

    def shutdown(self) -> None:
        self.llm_engine.shutdown()

    def __enter__(self) -> LLM:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
