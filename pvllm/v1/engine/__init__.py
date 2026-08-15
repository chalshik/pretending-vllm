"""Wire types between the frontend and the engine core.

Upstream: vllm/v1/engine/__init__.py
Tier: B

These cross the `EngineCoreClient` boundary. Upstream uses msgspec Structs so that the
multiprocess client pays real serialization cost rather than a modeled one (R4.2); the
same types are used in process, so the in-process path exercises the same shapes.

**Clock ownership (R19.1/R4.4).** Upstream stamps `arrival_time` in the frontend, using
`time.time()`. pretending-vllm cannot: the engine core is the sole owner of the clock,
and a frontend that read a clock would break determinism the moment the engine ran
multiprocess. So `EngineCoreRequest.arrival_time` is optional here, and the engine core
stamps it on receipt. This is a deliberate divergence from upstream, and it is the one
place in the wire protocol where R19.1 shows.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from typing import Any

import msgspec

# Imported at runtime, not under TYPE_CHECKING. msgspec resolves a Struct's
# annotations when it builds a decoder, so a type that exists only for the type
# checker makes `Decoder(EngineCoreRequest)` raise `NameError` -- which the
# in-process client never triggers, because it never decodes anything. The
# multiprocess client does, on every request.
from pvllm.sampling_params import SamplingParams

FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")


class FinishReason(enum.IntEnum):
    """Why a request finished.

    Int rather than str for compact serialization, matching upstream.
    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self) -> str:
        return FINISH_REASON_STRINGS[self.value]


class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""

    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """A timestamped engine core event associated with a request.

    The timestamp comes from the engine core's clock, never from wall time, so the
    intervals the frontend computes from these are modeled durations (R12.4).
    """

    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(
        cls, event_type: EngineCoreEventType, timestamp: float
    ) -> EngineCoreEvent:
        return cls(event_type, timestamp)


class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    """A request as it crosses into the engine core."""

    request_id: str
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams | None
    #: Stamped by the engine core on receipt, not by the frontend (R19.1). `None`
    #: from the frontend is the normal case.
    arrival_time: float | None = None
    client_index: int = 0
    lora_request: Any = None
    cache_salt: str | None = None
    priority: int = 0
    trace_headers: Mapping[str, str] | None = None
    data_parallel_rank: int | None = None


class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    """One request's incremental result from one engine step."""

    request_id: str
    new_token_ids: list[int]
    new_logprobs: Any = None
    new_prompt_logprobs_tensors: Any = None
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, Any] | None = None
    trace_headers: Mapping[str, str] | None = None
    num_cached_tokens: int = 0

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
):
    """Everything one engine step produced for one frontend client."""

    engine_index: int = 0
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: Any = None
    timestamp: float = 0.0
    finished_requests: set[str] | None = None


class EngineCoreRequestType(enum.Enum):
    """What a frame on the input socket carries.

    Hex byte strings so the tag needs no encoding step of its own, matching upstream.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    UTILITY = b"\x02"
    #: Sentinel used inside the core process to wake a blocked input queue.
    WAKEUP = b"\x03"


class UtilityCall(msgspec.Struct, array_like=True, gc=False):
    """A synchronous call from the frontend to the core.

    Everything that is not "add a request" or "abort a request" -- reading the
    engine's clock, scraping stats, resetting the prefix cache. Correlated by
    `call_id` because the reply comes back interleaved with output frames on the
    same socket, and matching by arrival order alone would break the first time two
    calls were in flight.
    """

    call_id: int
    method: str
    args: list[Any] = []


class UtilityReply(msgspec.Struct, array_like=True, gc=False):
    """The answer to a `UtilityCall`, or the error it raised.

    Errors travel as text rather than being re-raised structurally: a traceback from
    another process is not reconstructable, and a string that names the method and
    the failure is more use than a plausible-looking exception with the wrong
    traceback attached.
    """

    call_id: int
    result: Any = None
    error: str | None = None
