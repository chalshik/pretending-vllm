"""The scheduler's decision for one step.

Upstream: vllm/v1/core/sched/output.py
Tier: A

This is the structure that crosses the simulation boundary (section 4). Everything
above it is real; `SimModelRunner.execute_model` is the only thing below. Its field set
is therefore part of the C1 contract, and the trace records it verbatim (R5.10).

Note `scheduled_cached_reqs` is a single `CachedRequestData` holding parallel arrays,
not a list of per-request objects (F7). That shape is what makes the worker's
persistent batch update an incremental diff rather than a rebuild (R7.3): the runner
walks the arrays and patches only what changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from pvllm.sampling_params import SamplingParams
    from pvllm.v1.request import Request


@dataclass
class NewRequestData:
    """A request being scheduled for the first time.

    Carries everything the worker needs to build its persistent state, because the
    worker has never seen this request before.
    """

    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[Any]
    sampling_params: SamplingParams | None
    #: One list of block ids per KV cache group (R6.7). Single-element until hybrid
    #: models land in M4, but the tuple shape exists now so the group abstraction is
    #: right from the start.
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: Any = None

    @classmethod
    def from_request(
        cls, request: Request, block_ids: tuple[list[int], ...]
    ) -> NewRequestData:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
        )

    def __repr__(self) -> str:
        return (
            f"NewRequestData(req_id={self.req_id!r}, "
            f"prompt_len={len(self.prompt_token_ids or ())}, "
            f"num_computed_tokens={self.num_computed_tokens}, "
            f"block_ids={self.block_ids})"
        )


@dataclass
class CachedRequestData:
    """Requests the worker already knows about, as parallel arrays.

    Index `i` of every list refers to `req_ids[i]`. Kept as arrays rather than objects
    because this is the hot path -- it is rebuilt every step for every running request.
    """

    req_ids: list[str]
    #: Requests coming back from PREEMPTED, whose worker state must be rebuilt.
    resumed_req_ids: set[str]
    new_token_ids: list[list[int]]
    #: `None` where a request gained no new blocks this step.
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )

    def __repr__(self) -> str:
        return (
            f"CachedRequestData(num_reqs={self.num_reqs}, "
            f"resumed={sorted(self.resumed_req_ids)})"
        )


@dataclass
class SchedulerOutput:
    """One step's scheduling decision. F7."""

    #: Requests scheduled for the first time this step.
    scheduled_new_reqs: list[NewRequestData]
    #: Requests the worker already holds state for.
    scheduled_cached_reqs: CachedRequestData
    #: req_id -> number of tokens to process this step. The core decision (R5.1).
    num_scheduled_tokens: dict[str, int]
    #: Must never exceed `max_num_batched_tokens` (R5.3).
    total_num_scheduled_tokens: int
    #: req_id -> draft token ids being verified (R14).
    scheduled_spec_decode_tokens: dict[str, list[int]]
    #: req_id -> encoder input indices scheduled against the separate budget (R5.2).
    scheduled_encoder_inputs: dict[str, list[int]]
    #: Per KV cache group: blocks shared by every running request, for cascade
    #: attention (R5.9).
    num_common_prefix_blocks: list[int]
    #: Requests that finished, so the worker drops their cached state (R5.8).
    finished_req_ids: set[str]
    #: Encoder cache entries free to evict (R6.8).
    free_encoder_mm_hashes: list[str]

    #: Preempted this step (R5.5). Distinct from finished: their blocks are freed but
    #: they return to the waiting queue.
    preempted_req_ids: set[str] | None = None
    has_structured_output_requests: bool = False
    pending_structured_output_tokens: bool = False
    num_invalid_spec_tokens: dict[str, int] | None = None
    structured_output_request_ids: dict[str, int] = field(default_factory=dict)
    grammar_bitmask: npt.NDArray[np.int32] | None = None
    kv_connector_metadata: Any = None
    new_block_ids_to_zero: list[int] | None = None
    num_spec_tokens_to_schedule: int = 0

    @classmethod
    def make_empty(cls) -> SchedulerOutput:
        """A step in which nothing was scheduled.

        Returned when the waiting queue is empty or the budget is exhausted; the
        engine still counts it as a step so the trace stays gap-free.
        """
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

    def to_trace_dict(self) -> dict[str, Any]:
        """The stable trace representation of this decision. R5.10.

        Sets are sorted rather than dumped directly: set iteration order is not
        stable across runs, and an unstable ordering here would show up as a spurious
        difference in every C1 conformance diff.
        """
        return {
            "new_reqs": [r.req_id for r in self.scheduled_new_reqs],
            "cached_reqs": list(self.scheduled_cached_reqs.req_ids),
            "resumed_reqs": sorted(self.scheduled_cached_reqs.resumed_req_ids),
            "num_scheduled_tokens": dict(sorted(self.num_scheduled_tokens.items())),
            "total_num_scheduled_tokens": self.total_num_scheduled_tokens,
            "finished_req_ids": sorted(self.finished_req_ids),
            "preempted_req_ids": sorted(self.preempted_req_ids or ()),
            "num_common_prefix_blocks": list(self.num_common_prefix_blocks),
        }

    def __repr__(self) -> str:
        return (
            f"SchedulerOutput(new={len(self.scheduled_new_reqs)}, "
            f"cached={self.scheduled_cached_reqs.num_reqs}, "
            f"tokens={self.total_num_scheduled_tokens}, "
            f"finished={len(self.finished_req_ids)})"
        )
