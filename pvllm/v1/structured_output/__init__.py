"""Engine-level structured output. R15.

Upstream: vllm/v1/structured_output/__init__.py
Tier: B

`StructuredOutputManager` owns the backend and compiles grammars off the scheduler's
thread. The scheduler-side interaction is the part R15.1 requires to be real, and it
is: a request whose grammar is still compiling sits
in `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` and is skipped for admission, exactly as
upstream, so a product that submits a hundred requests against a slow-to-compile
schema sees the same admission behavior it would see against real vLLM.

Compilation runs on a thread pool, as upstream. That is not decoration: it is what
keeps one expensive schema from stalling every other request in the engine, and the
`Future` handling in `StructuredOutputRequest` exists to serve it.

Dropped from upstream: reasoning parsers, per-request backend selection, and the
logit bitmask. The first two raise where reached. The third is absent on purpose and
`grammar_bitmask` explains why -- there is no distribution to mask, so a mask would be
ceremony rather than fidelity.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

from pvllm.config import VllmConfig
from pvllm.logger import init_logger
from pvllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
)

if TYPE_CHECKING:
    from pvllm.v1.request import Request

logger = init_logger(__name__)

__all__ = [
    "StructuredOutputBackend",
    "StructuredOutputGrammar",
    "StructuredOutputManager",
]

#: The backends upstream ships. Named so that asking for one says what is missing
#: rather than silently substituting a different grammar engine -- two backends
#: disagree about edge cases in JSON Schema, and a product that pinned one did so
#: for a reason.
UPSTREAM_BACKENDS = ("xgrammar", "guidance", "outlines", "lm-format-enforcer")


class StructuredOutputManager:
    """Owns the grammar backend and the per-step bitmask."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.backend: StructuredOutputBackend | None = None

        # Half the CPUs, as upstream: grammar compilation is CPU-bound, and the
        # default pool size (CPUs * 5) would oversubscribe a machine that is also
        # running the engine.
        max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="pvllm-grammar"
        )
        self.tokenizer: Any = None

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Hand the manager the tokenizer the engine resolved.

        Passed in rather than constructed here, unlike upstream, because the engine
        core already built one and a second instance would be a second vocabulary --
        two objects that must agree exactly, where one shared object simply does.
        """
        self.tokenizer = tokenizer

    # --- compilation ---------------------------------------------------------

    def grammar_init(self, request: Request) -> None:
        """Start compiling this request's grammar. Returns immediately."""
        if request.structured_output_request is None:
            return

        if self.backend is None:
            self.backend = self._build_backend(request)

        request.structured_output_request.grammar = self.executor.submit(
            self._create_grammar, request
        )

    def _build_backend(self, request: Request) -> StructuredOutputBackend:
        assert request.sampling_params is not None
        params = request.sampling_params.structured_outputs
        assert params is not None
        backend = params._backend or self.vllm_config.structured_outputs_config.backend

        if backend in UPSTREAM_BACKENDS:
            raise NotImplementedError(
                f"structured output backend {backend!r} is a compiled grammar engine "
                f"over a real vocabulary and is not available here. pretending-vllm "
                f"provides the 'sim' backend, which generates output conforming to "
                f"the constraint (see pvllm/sim/structured_output.py for "
                f"what that does and does not tell you). Pass "
                f"--structured-outputs-backend auto or sim."
            )
        if backend not in ("auto", "sim"):
            raise ValueError(
                f"unknown structured output backend {backend!r}; expected 'auto' or "
                f"'sim' (upstream's backends are {list(UPSTREAM_BACKENDS)})"
            )

        if self.tokenizer is None:
            raise RuntimeError(
                "the structured output manager has no tokenizer; the engine core "
                "must call set_tokenizer before admitting a constrained request"
            )
        # Through the platform (B2), like the clock and the trace sink: the
        # backend is simulator-supplied, and this module must not import `pvllm.sim`
        # -- the purity lint enforces that, and enforced it when this was first
        # written the other way round.
        from pvllm.platforms import current_platform

        backend_obj = current_platform.build_structured_output_backend(
            self.vllm_config,
            tokenizer=self.tokenizer,
            vocab_size=self.vllm_config.model_config.get_vocab_size(),
        )
        # Narrowed rather than trusted: the platform resolves this at runtime (B2),
        # so its return type is Any, and a plugin returning the wrong thing should
        # fail here rather than at the first compile.
        assert isinstance(backend_obj, StructuredOutputBackend), (
            f"the platform returned {type(backend_obj).__name__}, which is not a "
            f"StructuredOutputBackend"
        )
        return backend_obj

    def _create_grammar(self, request: Request) -> StructuredOutputGrammar:
        structured = request.structured_output_request
        assert structured is not None and self.backend is not None
        request_type, grammar_spec = structured.structured_output_key
        try:
            return self.backend.compile_grammar(
                request_type, grammar_spec, request.request_id
            )
        except Exception:
            logger.exception(
                "failed to compile grammar for request %s", request.request_id
            )
            raise

    # --- the per-step bitmask ------------------------------------------------

    def grammar_bitmask(
        self,
        requests: dict[str, Request],
        structured_output_request_ids: dict[str, int],
    ) -> np.ndarray | None:
        """Always `None` here, and deliberately. R15.

        Upstream returns a per-step logit mask so the sampler can only pick tokens
        the grammar allows. The simulated backend has nothing to mask -- the model
        generates conforming text rather than sampling within a constraint -- so this
        returns `None` and `SchedulerOutput.grammar_bitmask` stays unset.

        The method is kept because it is the shape of the interaction: a consumer
        reading the scheduler output sees the same field, and a future backend with a
        real distribution behind it would fill it here without anything above
        changing. Returning a mask nothing consumes would be worse than returning
        nothing -- it would read as fidelity that is not there.
        """
        return None

    def should_advance(self, request: Request) -> bool:
        """Whether this request's grammar should consume the tokens just sampled."""
        if not request.use_structured_output:
            return False
        structured = request.structured_output_request
        if structured is None:
            return False
        return isinstance(structured.grammar, StructuredOutputGrammar)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.backend is not None:
            self.backend.destroy()
