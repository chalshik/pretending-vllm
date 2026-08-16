"""The simulated grammar backend. R15.1.

Upstream: (none -- simulator)
Tier: D

Upstream ships four grammar backends -- xgrammar, guidance, outlines,
lm-format-enforcer -- and every one of them works the same way: compile the constraint
into an automaton over the model's real vocabulary, then at each step mask the logits
so the sampler can only pick a token the automaton allows. The model's learned
distribution chooses among the allowed tokens.

**That mechanism cannot be ported, because the thing it steers does not exist here.**
`SimModel` has no distribution -- it invents tokens -- so an automaton admitting every
string that satisfies a schema would have nothing to choose with, and the sampler
would land on whichever synthetic pseudoword it was going to emit anyway. The result
would satisfy the bitmask and would not be JSON. A product calling `json.loads()` on
it fails, against the one feature it uses structured output for.

So the constraint is applied where a simulator can honestly apply it: `SimModel`
generates text that conforms (see `pvllm/sim/grammar.py`) and emits its tokens. This
backend keeps the half of the contract that *is* portable:

* **Compilation is real.** A malformed schema, an unsatisfiable range, an unsupported
  regex construct -- all fail here, on the compile thread, and fail only their own
  request. That is what makes the scheduler-side interaction R15.1 asks for real:
  `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`, asynchronous compilation, admission
  gating, and per-request compile errors all behave as upstream's do.
* **Termination is real.** The grammar knows when the constraint has been satisfied.

What is deliberately *absent* is the bitmask. `fill_bitmask` and
`allocate_token_bitmask` raise rather than returning a mask nothing consumes --
computing one would be ceremony that looks like fidelity and is not, and a mask that
never influenced a token is worse than an honest gap. `SchedulerOutput.grammar_bitmask`
stays `None` for the same reason.

The consequence worth stating: this backend tells you your structured-output plumbing
works. It cannot tell you whether a real model would struggle to satisfy your schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pvllm.logger import init_logger
from pvllm.sim.grammar import generate_for_constraint
from pvllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

logger = init_logger(__name__)

#: What the simulated backend charges for a compile, in seconds. Zero, and explicitly
#: so: R15.1 permits a modeled compile latency, but the compile happens on a thread
#: pool off the engine's clock, and a duration invented here would not appear on the
#: modeled timeline anyway. A schema that takes xgrammar 200 ms is a real cost this
#: does not reproduce.
COMPILE_SECONDS = 0.0


@dataclass
class SimGrammar(StructuredOutputGrammar):
    """One request's constraint, tracked as progress through its output."""

    kind: StructuredOutputOptions
    spec: str
    #: How many tokens the conforming output is. Filled by the worker's model when
    #: it generates the content; `None` until then.
    num_target_tokens: int | None = field(default=None, init=False)
    position: int = field(default=0, init=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Advance by `len(tokens)`.

        Always accepts. The tokens came from a model that generated them *from* this
        constraint, so rejecting them would mean the simulator disagreed with itself.
        A real backend's answer here is informative because its model might have
        produced anything.
        """
        self.position += len(tokens)
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        return list(tokens)

    def rollback(self, num_tokens: int) -> None:
        self.position = max(0, self.position - num_tokens)

    def fill_bitmask(self, bitmask: np.ndarray, batch_index: int) -> None:
        raise NotImplementedError(
            "the simulated grammar backend computes no token bitmask. Upstream masks "
            "logits so a learned distribution picks among allowed tokens; SimModel "
            "has no distribution, so the constraint is satisfied by generating "
            "conforming text instead (see pvllm/sim/grammar.py). A mask nothing "
            "consumes would look like fidelity without being it."
        )

    def is_terminated(self) -> bool:
        return (
            self.num_target_tokens is not None
            and self.position >= self.num_target_tokens
        )

    def reset(self) -> None:
        self.position = 0


class SimStructuredOutputBackend(StructuredOutputBackend):
    """Validates constraints. The compile step, and only the compile step."""

    def compile_grammar(
        self,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        request_id: str | None = None,
    ) -> StructuredOutputGrammar:
        """Prove the constraint is satisfiable, then return a fresh tracker.

        Satisfiability is checked by *generating* against it once with a throwaway
        seed -- which is the only check that matters, since generation is how the
        constraint will actually be met. An unsupported regex construct, an enum with
        no members, a maximum below its minimum: each raises here, on the compile
        thread, where it belongs to one request instead of aborting the step that
        happened to notice.
        """
        # With the *request's own* generator, not an arbitrary seed. The two used
        # to differ, so a schema whose conformance depends on the draw -- a
        # `pattern` beside a `minLength`, say -- could compile on seed 0 and then
        # fail validation at generation, inside `execute_model`, taking the engine
        # down instead of the request. Same generator, same string: a compile that
        # passes cannot be followed by a generation that fails.
        from pvllm.sim.rng import RngFactory

        seed = self.vllm_config.sim_config.seed
        rng = (
            RngFactory(seed).for_constraint(request_id)
            if request_id is not None
            else np.random.default_rng(seed)
        )
        generate_for_constraint(request_type.name.lower(), grammar_spec, rng)
        return SimGrammar(kind=request_type, spec=grammar_spec)

    def allocate_token_bitmask(self, max_num_seqs: int) -> Any:
        raise NotImplementedError(
            "the simulated grammar backend computes no token bitmask; see "
            "SimGrammar.fill_bitmask for why"
        )

    def destroy(self) -> None:
        return None
