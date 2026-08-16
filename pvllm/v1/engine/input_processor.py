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

from typing import Any

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.pooling_params import PoolingParams
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
        sampling_params: SamplingParams | None = None,
        *,
        client_index: int = 0,
        priority: int = 0,
        cache_salt: str | None = None,
        lora_request: Any = None,
        mm_features: list[Any] | None = None,
        pooling_params: PoolingParams | None = None,
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

        if (sampling_params is None) == (pooling_params is None):
            raise ValueError(
                "exactly one of sampling_params and pooling_params must be given"
            )

        if sampling_params is not None:
            self._validate_params(sampling_params)
            self._validate_prompt_length(request_id, prompt_token_ids, sampling_params)
            # Bind the tokenizer's EOS, honouring ignore_eos by leaving it unset.
            sampling_params.update_from_tokenizer(self.tokenizer.eos_token_id)
        else:
            # R2.2. A pooling request generates nothing, so the length check is the
            # prompt against the context window rather than prompt plus output.
            self._validate_pooling_prompt_length(request_id, prompt_token_ids)

        if lora_request is not None and self.vllm_config.lora_config is None:
            # R16.1. Accepting it would give the adapter a prefix-cache partition and
            # no memory cost, and leave `max_loras` unenforced -- a capacity answer
            # for a deployment that could not serve the request at all.
            raise ValueError(
                "a lora_request was given but LoRA is not enabled; start the engine "
                "with enable_lora=True (adapters cost KV pool memory and max_loras "
                "bounds how many are resident, so neither is inferred)"
            )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            # Left unset: the engine core owns the clock (R19.1).
            arrival_time=None,
            client_index=client_index,
            cache_salt=cache_salt,
            priority=priority,
            lora_request=lora_request,
            mm_features=list(mm_features or ()),
            pooling_params=pooling_params,
        )

    def _validate_pooling_prompt_length(
        self, request_id: str, prompt_token_ids: list[int]
    ) -> None:
        """R2.5 for pooling: the prompt alone must fit the context window."""
        # The same guard the generation path opens with, and for the same reason: a
        # zero-token request is admitted, is scheduled with nothing to compute, never
        # advances, and `has_unfinished_requests()` never goes false. A pooling
        # request has no sampled token to end it either, so it hangs the engine
        # rather than merely stalling. Caught here so the client gets a 400.
        if not prompt_token_ids:
            raise ValueError("Prompt cannot be empty.")
        max_model_len = self.model_config.max_model_len
        assert max_model_len is not None
        if len(prompt_token_ids) > max_model_len:
            raise ValueError(
                f"This model's maximum context length is {max_model_len} tokens. "
                f"However, you requested {len(prompt_token_ids)} tokens in the input "
                f"for embedding generation. Please reduce the length of the input."
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
