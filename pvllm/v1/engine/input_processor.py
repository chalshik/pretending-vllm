"""Turning a user request into an EngineCoreRequest.

Upstream: vllm/v1/engine/input_processor.py
Tier: B

F3: upstream renamed this from `processor.py`. R3.1: validate, apply the chat template,
tokenize, build the wire request.

R19.1 shows here: the processor does **not** stamp `arrival_time`. It runs in the
frontend, which has no clock -- the engine core stamps on receipt. Upstream reads
`time.time()` at this point; that is the one place the clock-ownership rule forces a
divergence, and it is deliberate.
"""

from __future__ import annotations

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.sampling_params import SamplingParams
from pvllm.tokenizers.protocol import TokenizerLike
from pvllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


class InputProcessor:
    """Validates and tokenizes incoming requests."""

    def __init__(self, vllm_config: VllmConfig, tokenizer: TokenizerLike) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.tokenizer = tokenizer
        assert self.model_config.max_model_len is not None
        self.max_model_len: int = self.model_config.max_model_len

    def process_inputs(
        self,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        *,
        client_index: int = 0,
        priority: int = 0,
        cache_salt: str | None = None,
    ) -> EngineCoreRequest:
        """Build the wire request. R3.1.

        Args:
            prompt: Text, or token ids supplied directly. R3.3 allows the latter so a
                recorded trace can be replayed without depending on the tokenizer
                reproducing the original tokenization.
        """
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_token_ids = list(prompt)

        self._validate_params(sampling_params)
        self._validate_prompt_length(request_id, prompt_token_ids, sampling_params)

        # Bind the tokenizer's EOS, honouring ignore_eos by leaving it unset.
        sampling_params.update_from_tokenizer(self.tokenizer.eos_token_id)

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            # Left unset: the engine core owns the clock (R19.1).
            arrival_time=None,
            client_index=client_index,
            cache_salt=cache_salt,
            priority=priority,
        )

    def _validate_params(self, sampling_params: SamplingParams) -> None:
        """R2.5: error parity for unsupported sampling parameters."""
        max_logprobs = self.model_config.max_logprobs
        for name in ("logprobs", "prompt_logprobs"):
            value = getattr(sampling_params, name)
            if value is not None and value > max_logprobs:
                raise ValueError(
                    f"Requested {name} of {value} exceeds the max allowed value of "
                    f"{max_logprobs}. Raise --max-logprobs to allow it."
                )
        if sampling_params.allowed_token_ids is not None:
            raise NotImplementedError(
                "allowed_token_ids is not modeled; it constrains sampling, and the "
                "sampler's effects reach only as far as the PRNG draw (NG3)"
            )

    def _validate_prompt_length(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> None:
        """R2.5: context length exceeded, in upstream's shape.

        Checked here rather than in the scheduler so the client gets a 400 rather
        than a request that is admitted and then never fits.
        """
        prompt_len = len(prompt_token_ids)
        if prompt_len == 0:
            raise ValueError("Prompt cannot be empty.")

        if prompt_len >= self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is {self.max_model_len} tokens. "
                f"However, you requested {prompt_len} tokens in the messages. "
                f"Please reduce the length of the messages."
            )

        # max_tokens is resolved against the remaining budget here, so that a Request
        # is never built with an unresolved cap (see Request.__init__).
        remaining = self.max_model_len - prompt_len
        if sampling_params.max_tokens is None:
            sampling_params.max_tokens = remaining
        elif sampling_params.max_tokens > remaining:
            raise ValueError(
                f"This model's maximum context length is {self.max_model_len} tokens. "
                f"However, you requested {prompt_len + sampling_params.max_tokens} "
                f"tokens ({prompt_len} in the messages, "
                f"{sampling_params.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion."
            )
