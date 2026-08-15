"""Request state.

Upstream: vllm/v1/request.py
Tier: A

The scheduler reads and mutates this on every step, so its shape is part of the C1
contract. Two details matter more than they look:

**`RequestStatus` member order is load-bearing (F6).** `is_finished` is implemented as
`status > PREEMPTED`, not as membership in a set. Every finished state must therefore
sort after `PREEMPTED`, and no unfinished state may. Reorder the enum and
finished-request detection breaks silently -- no type error, no failing assertion, just
requests that never complete. `tests/v1/test_request.py` pins the ordering.

**Block hashing is injected (F8).** `block_hasher` arrives as a callable rather than
being computed inline, so the KV cache manager owns hashing policy (salt, algorithm,
extra keys) and `Request` only stores the result. R6.3 depends on this.

Diverging from upstream in one place, deliberately: `arrival_time` is required rather
than defaulting to `time.time()`. The engine core owns the clock (R19.1) and stamps it.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from pvllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from pvllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from pvllm.sampling_params import SamplingParams
    from pvllm.v1.core.kv_cache_utils import BlockHash


class RequestStatus(enum.IntEnum):
    """Status of a request.

    **The order of these members is part of the contract.** Anything after `PREEMPTED`
    counts as finished -- see `is_finished`. Insert new unfinished states before
    `PREEMPTED` and new finished states after `FINISHED_STOPPED`.
    """

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Anything after PREEMPTED is considered finished.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def is_finished(status: RequestStatus) -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: RequestStatus) -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


#: A request whose prompt exceeds the model's length cap is *ignored*, and the OpenAI
#: API reports that as "length" -- which is why FINISHED_IGNORED maps to LENGTH.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ERROR: FinishReason.ERROR,
    RequestStatus.FINISHED_REPETITION: FinishReason.REPETITION,
    RequestStatus.WAITING_FOR_STREAMING_REQ: FinishReason.STOP,
}


class Request:
    """A single generation request, as the engine core sees it."""

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        arrival_time: float,
        client_index: int = 0,
        lora_request: Any = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[[Request], list[BlockHash]] | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.lora_request = lora_request
        self.arrival_time = arrival_time

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None
        self.kv_transfer_params: dict[str, Any] | None = None

        if sampling_params is None:
            raise ValueError("sampling_params must be set")
        if sampling_params.max_tokens is None:
            raise ValueError(
                "sampling_params.max_tokens must be resolved before a Request is "
                "built; the processor resolves it against max_model_len"
            )
        self.max_tokens: int = sampling_params.max_tokens

        self.prompt_token_ids = prompt_token_ids or []
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = list(self.prompt_token_ids)

        self.num_computed_tokens = 0
        self.spec_token_ids: list[int] = []
        self.cache_salt = cache_salt
        self.mm_features: list[Any] = []

        #: True while this request is scheduled as a non-final prefill chunk (R5.4).
        self.is_prefill_chunk = False
        #: R5.5. Incremented on every preemption by recompute.
        self.num_preemptions = 0
        #: Set once a prefix cache lookup has run, for the metrics (R6.9).
        self.num_cached_tokens = 0

        # Read-only views over the lists above -- by reference, so they reflect later
        # appends. See ConstantList.
        self.output_token_ids: ConstantList[int] = ConstantList(self._output_token_ids)
        self.all_token_ids: ConstantList[int] = ConstantList(self._all_token_ids)
        self.trace_headers = trace_headers

        self.block_hashes: list[BlockHash] = []
        # Stored unbound so that Request -> partial -> Request never forms a reference
        # cycle; a server holding thousands of finished requests until the cycle
        # collector runs is a real leak, and upstream avoids it the same way.
        self._block_hasher = block_hasher
        self.update_block_hashes()

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        arrival_time: float,
        block_hasher: Callable[[Request], list[BlockHash]] | None = None,
    ) -> Request:
        """Build from the wire type.

        `arrival_time` is passed explicitly by the engine core rather than read from
        the request, because the frontend never reads a clock (R19.1). A request that
        already carries one -- a replayed trace, for instance -- keeps it.
        """
        return cls(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            arrival_time=(
                request.arrival_time
                if request.arrival_time is not None
                else arrival_time
            ),
            client_index=request.client_index,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
        )

    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)
        self.update_block_hashes()

    def attach_block_hasher(
        self, block_hasher: Callable[[Request], list[BlockHash]]
    ) -> None:
        """Bind the KV manager's hasher and hash whatever is already here (F8).

        Attached by the scheduler rather than passed at construction: the frontend
        builds the request but has no business knowing the block size or the hash
        algorithm, which are the KV manager's to choose.
        """
        self._block_hasher = block_hasher
        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Hash any newly-complete blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    # --- token accounting ----------------------------------------------------

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    @property
    def use_structured_output(self) -> bool:
        return False  # R15 lands in M4.

    # --- lifecycle -----------------------------------------------------------

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def record_event(self, event_type: EngineCoreEventType, timestamp: float) -> None:
        """Record a lifecycle event. The timestamp comes from the engine's clock."""
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def __lt__(self, other: Request) -> bool:
        """Priority scheduling order (R5.6): priority, then arrival, then id.

        The `request_id` tiebreak is what makes the order total, and therefore what
        makes C1 reproducible when two requests share an arrival time -- which is
        common under a virtual clock, where many requests can arrive at the same
        modeled instant.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id!r}, status={self.status}, "
            f"prompt={self.num_prompt_tokens}, output={self.num_output_tokens}, "
            f"computed={self.num_computed_tokens}, preemptions={self.num_preemptions})"
        )
