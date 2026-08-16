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

from pvllm.sim.grammar import generate_for_constraint
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
    #: R14. How often a draft is accepted. See SimConfig.spec_acceptance_rate.
    spec_acceptance_rate: float = 0.7

    #: request_id -> how many tokens it will emit before stopping. Decided once, on
    #: first use, so the answer cannot drift mid-generation.
    _planned_lengths: dict[str, int] = field(default_factory=dict)

    #: R15. request_id -> the exact token sequence a constrained request must emit.
    #: See pvllm/sim/grammar.py for why the constraint is satisfied by generating
    #: conforming text rather than by masking a sampler that has no distribution to
    #: mask.
    _constrained_plans: dict[str, list[int]] = field(default_factory=dict)

    # --- speculative decoding (R14) ------------------------------------------

    def accepted_draft_count(self, request_id: str, num_drafts: int) -> int:
        """How many of `num_drafts` this step accepts. R14.

        Drawn per draft position at `acceptance_rate`, and stopping at the first
        rejection -- which is what verification actually does. A run of drafts is
        accepted only as a prefix, because each one conditions the next: rejecting
        the second invalidates the third whatever the target thought of it.

        That prefix rule is why the expected accepted count is
        `sum(rate**i for i in 1..k)` rather than `k * rate`, and why the return on
        `num_speculative_tokens` falls off so sharply once acceptance drops -- the
        curve a product tuning it needs to see.
        """
        if num_drafts <= 0 or self.spec_acceptance_rate <= 0.0:
            return 0
        rng = self.rng_factory.for_request(request_id)
        accepted = 0
        for _ in range(num_drafts):
            if float(rng.random()) >= self.spec_acceptance_rate:
                break
            accepted += 1
        return accepted

    def propose_drafts(
        self, request_id: str, position: int, num_drafts: int, max_tokens: int
    ) -> list[int]:
        """Draft continuations for the next step. R14.

        Content-free by construction: what a draft *is* only matters through whether
        the target accepts it, and that is drawn rather than compared. The ids are
        the ones this request would emit anyway, so a run with speculation on
        produces the same text as one with it off -- which is the property that makes
        the two comparable at all.
        """
        if num_drafts <= 0:
            return []
        planned = self.planned_output_length(request_id, max_tokens)
        return [
            self.sample_token(request_id, position + offset, max_tokens)
            for offset in range(1, num_drafts + 1)
            if position + offset < planned
        ]

    # --- structured output (R15) ---------------------------------------------

    def set_constraint(self, request_id: str, kind: str, spec: str) -> list[int]:
        """Decide what a constrained request will emit, once.

        Cached per request, for the same reason `planned_output_length` is: a
        request that is preempted and recomputed must produce the same string, or
        preemption would change the answer -- which R21.1 forbids and a product
        would experience as a nondeterministic API.

        Encoded here rather than through a tokenizer object because the mock
        vocabulary is byte-level by construction: ids `BYTE_TOKEN_OFFSET + b` are
        the 256 byte values, and this module already depends on that layout for EOS.
        Threading a tokenizer down to the worker would add a second instance of a
        vocabulary that has to agree with the frontend's exactly -- and two objects
        that must agree are a worse guarantee than one rule both follow.
        """
        planned = self._constrained_plans.get(request_id)
        if planned is not None:
            return planned

        text = generate_for_constraint(
            kind, spec, self.rng_factory.for_constraint(request_id)
        )
        planned = [BYTE_TOKEN_OFFSET + byte for byte in text.encode("utf-8")]
        if planned and max(planned) >= self.model.vocab_size:
            raise ValueError(
                f"the constrained output for {request_id} needs token ids up to "
                f"{max(planned)}, beyond this model card's vocab_size of "
                f"{self.model.vocab_size}"
            )
        self._constrained_plans[request_id] = planned
        return planned

    def constrained_plan(self, request_id: str) -> list[int] | None:
        return self._constrained_plans.get(request_id)

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

        # R15. A constrained request's length is the constraint's, not the workload
        # policy's: the output is a JSON document or a regex match, and stopping
        # partway through would emit something that does not parse. Truncation by
        # `max_tokens` still applies below, and is exactly what real vLLM does to a
        # schema that does not fit.
        constrained = self._constrained_plans.get(request_id)
        if constrained is not None:
            # The content, *plus one* for the EOS that ends the request. Without the
            # +1 the EOS branch in `sample_token` fires one position early and eats
            # the document's last character -- a JSON object missing its closing
            # brace, which parses nowhere and looks like a schema bug rather than an
            # off-by-one.
            length = min(len(constrained) + 1, max_tokens)
            self._planned_lengths[request_id] = length
            return length

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

        When the policy stops a request *early*, EOS is emitted at the planned
        length, so it stops through the real stop-detection path (R11.5) rather than
        being cut off out of band -- the scheduler's finish accounting, the
        finish_reason, and the metrics all key off that path.

        When the plan runs to `max_tokens`, **no EOS is emitted** and the length cap
        ends the request instead. That distinction is what makes `finish_reason`
        meaningful: a product testing "was I truncated?" must be able to observe
        `length`, and emitting EOS on the final allowed token would report `stop` for
        every request under the default policy -- a wrong answer about the one field
        a client uses to decide whether to continue.
        """
        # R15. The constrained plan decides its own end, *before* the generic
        # length logic. The two used to be entangled: `planned` was
        # `len(plan) + 1` capped at `max_tokens`, and the EOS branch below only
        # fires when `planned < max_tokens` -- so at `max_tokens == len(plan) + 1`
        # they collided, no EOS was emitted, and the position one past the plan
        # indexed off the end. That IndexError escaped `execute_model` into the
        # engine step, which then wedged the whole engine for every later request,
        # constrained or not. One client's choice of max_tokens took the server down.
        constrained = self._constrained_plans.get(request_id)
        if constrained is not None:
            if position >= len(constrained):
                return EOS_TOKEN_ID
            return constrained[position]

        planned = self.planned_output_length(request_id, max_tokens)
        if planned < max_tokens and position + 1 >= planned:
            return EOS_TOKEN_ID

        # Keyed by position, not drawn from the request's stream: sampling has to be
        # idempotent, or speculation would not be lossless and a recomputed request
        # would produce different tokens. See `RngFactory.for_position`.
        rng = self.rng_factory.for_position(request_id, position)
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
                "sampling time, which the sampler does not receive"
            )
        if self.content_policy == "fixture":
            raise NotImplementedError(
                "the fixture content policy (requirement R11.3) is not implemented"
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
        """Drop a finished request's cached state. R15, R11.2.

        Every map here is keyed by request id and none were pruned, so a server
        doing structured-output traffic grew by roughly one document per request
        forever -- and a harness that reuses request ids (a benchmark numbering them
        0..n per iteration) silently got the *previous* request's document back,
        because the cache hit before anything regenerated.
        """
        self._constrained_plans.pop(request_id, None)
        self._planned_lengths.pop(request_id, None)
        self.rng_factory.forget_request(request_id)
