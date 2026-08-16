"""Parallel sampling: `n > 1`, handled in the frontend. R11.7.

Upstream: vllm/v1/engine/parallel_sampling.py
Tier: B

The engine core has no notion of `n`. One request is one sequence, and asking for
four completions of one prompt is four requests that happen to share a prompt --
which is exactly how upstream does it, and why it belongs here rather than in the
scheduler.

That is not an implementation shortcut; it is the behaviour worth reproducing. The
four children queue independently, are preempted independently, and share KV blocks
through the ordinary prefix cache rather than through a special case. A product
sending `n=4` sees four times the KV pressure and one response, and both halves of
that are what a capacity plan needs.

The one thing the frontend owes the client is that the response arrives as *one*
`RequestOutput` carrying `n` `CompletionOutput`s, in index order, whatever order the
children actually finished in.
"""

from __future__ import annotations

from copy import copy

from pvllm.outputs import CompletionOutput
from pvllm.sampling_params import RequestOutputKind, SamplingParams


class ParentRequest:
    """One client request that fans out into `n` engine requests.

    Owns the child ids, the per-child sampling params, and the aggregation that
    turns their outputs back into one response.
    """

    def __init__(self, request_id: str, sampling_params: SamplingParams) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params

        #: Children still outstanding. The request is finished when this empties.
        self.child_requests: set[str] = set()
        #: Aggregate unless the client asked for deltas. `DELTA` is the only kind
        #: with a per-chunk meaning under `n > 1` -- `CUMULATIVE` would interleave
        #: `n` full texts under one request id, which no client can read. The HTTP
        #: layer already maps non-streaming to `FINAL_ONLY`, as upstream does; this
        #: covers the offline engine, whose default is `CUMULATIVE`.
        self.streaming = sampling_params.output_kind == RequestOutputKind.DELTA
        #: Slots for the final outputs, so they come back in index order however
        #: the children raced. Empty when streaming, where each chunk goes out as it
        #: arrives.
        self.output_aggregator: list[CompletionOutput | None] = (
            [] if self.streaming else [None] * sampling_params.n
        )
        self._cached_child_params: SamplingParams | None = None

    @property
    def n(self) -> int:
        return self.sampling_params.n

    # --- fan out -------------------------------------------------------------

    def child_info(self, index: int) -> tuple[str, SamplingParams]:
        """The `(request_id, sampling_params)` for child `index`."""
        child_id = f"{index}_{self.request_id}"
        self.child_requests.add(child_id)
        return child_id, self._child_params(index)

    def _child_params(self, index: int) -> SamplingParams:
        """`n = 1`, and a distinct seed per child when the parent set one.

        Shared between children when there is no seed, because then they are
        interchangeable and copying `n` of them per request is pure allocation.
        """
        if self._cached_child_params is not None:
            return self._cached_child_params
        child = copy(self.sampling_params)
        child.n = 1
        child.best_of = None
        if self.sampling_params.seed is None:
            self._cached_child_params = child
            return child
        # A seeded parent asks for reproducibility, not for four identical
        # completions -- so the children are offset rather than cloned.
        child.seed = self.sampling_params.seed + index
        return child

    # --- fan in --------------------------------------------------------------

    def collect(
        self, child_request_id: str, output: CompletionOutput
    ) -> tuple[list[CompletionOutput], bool]:
        """Fold one child's output in. Returns `(outputs, parent_finished)`.

        `outputs` is what the client should see *now*: the chunk itself while
        streaming, nothing until the last child lands otherwise.
        """
        already_returned = False
        if output.finish_reason is not None:
            if child_request_id in self.child_requests:
                self.child_requests.discard(child_request_id)
            else:
                # It finished in an earlier step and was already handed back. Its
                # output must not go out twice.
                already_returned = True

        if self.streaming:
            outputs = [] if already_returned else [output]
        else:
            self.output_aggregator[output.index] = output
            outputs = (
                []
                if self.child_requests
                else [item for item in self.output_aggregator if item is not None]
            )

        return outputs, not self.child_requests


__all__ = ["ParentRequest"]
