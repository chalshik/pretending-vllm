"""The token generator. R11.1--R11.3.

Upstream: (none -- simulator)
Tier: D

No weights are ever read (NG1) and the output means nothing (NG1, NG3). What the
output *does* have is the right shape, the right length distribution, and stable text.

**Output length is the knob that makes workload experiments meaningful (R11.2).** A
simulator where every request emits exactly `max_tokens` answers a different question
than a real serving system: real requests stop early and at varying lengths, which is
what drives the batch composition the scheduler actually sees. `from_request` is the
default because it is what a test double should do -- honour what the client asked for
-- but `lognormal` is what a capacity experiment wants.

**Content must detokenize to stable text (R11.3)**, so HTTP responses can be
golden-tested. `pseudoword` draws token ids that `MockTokenizer` renders as stable
nonsense words; the same seed and request id always produce the same text.

R11.1: no vocab-sized array is allocated unless logprobs are requested. On a 128k
vocabulary at batch 256 that array would be 128 MiB per step, which would make the
simulator slower than the thing it simulates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pvllm.sim.model_db import ModelCard
from pvllm.sim.rng import RngFactory
from pvllm.tokenizers.mock import BYTE_TOKEN_OFFSET, EOS_TOKEN_ID

#: Sampled ids are drawn above the byte range so they render as pseudo-words rather
#: than as raw bytes, which keeps generated text readable in a trace.
FIRST_CONTENT_TOKEN_ID = BYTE_TOKEN_OFFSET + 256


@dataclass
class SimModel:
    """Produces one token id per sampled position.

    Args:
        model: For the vocabulary bound; a sampled id must be one the tokenizer and
            the logprobs schema consider valid.
        rng_factory: Per-request generators, so a request's output is independent of
            how it interleaved with others (R19.2).
        output_length_policy: See R11.2.
        content_policy: See R11.3.
    """

    model: ModelCard
    rng_factory: RngFactory
    output_length_policy: str = "from_request"
    content_policy: str = "pseudoword"
    output_length_fixed: int = 128
    output_length_range: tuple[int, int] = (16, 256)
    output_length_lognormal: tuple[float, float] = (4.0, 0.75)

    #: request_id -> how many tokens it will emit before stopping. Decided once, on
    #: first use, so the answer cannot drift mid-generation.
    _planned_lengths: dict[str, int] = field(default_factory=dict)

    # --- output length -------------------------------------------------------

    def planned_output_length(self, request_id: str, max_tokens: int) -> int:
        """How many tokens this request will emit. R11.2.

        Decided once per request and cached: drawing on every step would let the
        target wander, and a request would stop when the draw happened to be small
        rather than at a planned length.
        """
        planned = self._planned_lengths.get(request_id)
        if planned is not None:
            return planned

        rng = self.rng_factory.for_request(request_id)
        policy = self.output_length_policy

        if policy == "from_request":
            length = max_tokens
        elif policy == "fixed":
            length = self.output_length_fixed
        elif policy == "uniform":
            low, high = self.output_length_range
            length = int(rng.integers(low, high + 1))
        elif policy == "lognormal":
            mu, sigma = self.output_length_lognormal
            length = int(np.exp(rng.normal(mu, sigma)))
        elif policy == "from_fixture":
            raise NotImplementedError(
                "the from_fixture output length policy (requirement R11.2) needs the "
                "prompt-hash-to-output map, which lands with trace replay in M3"
            )
        else:
            raise ValueError(f"unknown output_length_policy {policy!r}")

        # Never exceed what the client asked for: a request emitting past max_tokens
        # would be a protocol violation regardless of the workload model.
        length = max(1, min(length, max_tokens))
        self._planned_lengths[request_id] = length
        return length

    # --- sampling ------------------------------------------------------------

    def sample_token(self, request_id: str, position: int, max_tokens: int) -> int:
        """One token for one position.

        Emits EOS at the planned length so the request stops through the *real*
        stop-detection path (R11.5) rather than being cut off out of band. That
        matters: the scheduler's finish accounting, the finish_reason, and the
        metrics all key off the normal path.
        """
        planned = self.planned_output_length(request_id, max_tokens)
        if position + 1 >= planned:
            return EOS_TOKEN_ID

        rng = self.rng_factory.for_request(request_id)
        if self.content_policy == "pseudoword":
            return int(
                rng.integers(
                    FIRST_CONTENT_TOKEN_ID,
                    max(FIRST_CONTENT_TOKEN_ID + 1, self.model.vocab_size),
                )
            )
        if self.content_policy == "echo":
            raise NotImplementedError(
                "the echo content policy (requirement R11.3) needs the prompt at "
                "sampling time; it lands in M3 with the trace viewer"
            )
        if self.content_policy == "fixture":
            raise NotImplementedError(
                "the fixture content policy (requirement R11.3) lands in M3"
            )
        raise ValueError(f"unknown content_policy {self.content_policy!r}")

    def sample_tokens(
        self, request_ids: list[str], positions: list[int], max_tokens: list[int]
    ) -> list[int]:
        """One token per request, in batch order."""
        return [
            self.sample_token(req_id, position, limit)
            for req_id, position, limit in zip(
                request_ids, positions, max_tokens, strict=True
            )
        ]

    # --- logprobs ------------------------------------------------------------

    def sample_logprobs(
        self, request_id: str, sampled_token_id: int, k: int
    ) -> tuple[list[int], list[float], int]:
        """Top-k logprobs for one position. Schema and shape only (NG3).

        The vocab-sized array R11.1 warns about is never built: only `k` entries are
        drawn. Values are synthetic but well-formed -- descending, negative, and with
        the sampled token present -- so a client that sorts or thresholds them
        behaves as it would against real output.
        """
        if k <= 0:
            return [sampled_token_id], [0.0], 0

        rng = self.rng_factory.for_request(request_id)
        magnitudes = np.sort(rng.exponential(1.0, size=k))
        logprobs = [float(-m) for m in magnitudes]

        token_ids = [sampled_token_id]
        while len(token_ids) < k:
            candidate = int(rng.integers(0, self.model.vocab_size))
            if candidate not in token_ids:
                token_ids.append(candidate)

        # The sampled token is rank 0 by construction, matching what a sampler
        # without temperature perturbation would report.
        return token_ids[:k], logprobs[:k], 0

    def forget_request(self, request_id: str) -> None:
        """Drop a finished request's planned length."""
        self._planned_lengths.pop(request_id, None)
        self.rng_factory.forget_request(request_id)
