"""Request state and the RequestStatus ordering contract. Section 7, F6."""

from __future__ import annotations

import pytest

from pvllm.sampling_params import SamplingParams
from pvllm.v1.engine import EngineCoreEventType, EngineCoreRequest, FinishReason
from pvllm.v1.request import Request, RequestStatus
from pvllm.v1.utils import ConstantList


def make_request(
    request_id="r0", prompt=(1, 2, 3), arrival_time=0.0, **kwargs
) -> Request:
    kwargs.setdefault("sampling_params", SamplingParams(max_tokens=8))
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        arrival_time=arrival_time,
        **kwargs,
    )


# --- the ordering contract -------------------------------------------------


def test_request_status_member_order_is_pinned():
    """F6. `is_finished` is `status > PREEMPTED`, so this order IS the semantics.

    Reordering breaks finished-request detection with no type error and no failing
    assertion elsewhere -- requests would simply never complete. Pinned explicitly so
    a reorder fails here instead.
    """
    assert [s.name for s in RequestStatus] == [
        "WAITING",
        "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR",
        "WAITING_FOR_REMOTE_KVS",
        "WAITING_FOR_STREAMING_REQ",
        "RUNNING",
        "PREEMPTED",
        "FINISHED_STOPPED",
        "FINISHED_LENGTH_CAPPED",
        "FINISHED_ABORTED",
        "FINISHED_IGNORED",
        "FINISHED_ERROR",
        "FINISHED_REPETITION",
    ]


def test_every_finished_status_sorts_after_preempted():
    """The invariant the ordering exists to serve."""
    for status in RequestStatus:
        expected = status.name.startswith("FINISHED_")
        assert RequestStatus.is_finished(status) is expected, (
            f"{status.name} sorts on the wrong side of PREEMPTED"
        )


def test_preempted_is_not_finished():
    """A preempted request returns to the waiting queue; treating it as finished
    would free its blocks and drop it (R5.5)."""
    assert not RequestStatus.is_finished(RequestStatus.PREEMPTED)
    assert not RequestStatus.is_finished(RequestStatus.RUNNING)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (RequestStatus.FINISHED_STOPPED, FinishReason.STOP),
        (RequestStatus.FINISHED_LENGTH_CAPPED, FinishReason.LENGTH),
        (RequestStatus.FINISHED_ABORTED, FinishReason.ABORT),
        # A prompt over the length cap is reported as "length" by the OpenAI API.
        (RequestStatus.FINISHED_IGNORED, FinishReason.LENGTH),
        (RequestStatus.FINISHED_ERROR, FinishReason.ERROR),
    ],
)
def test_finish_reason_mapping(status, reason):
    assert RequestStatus.get_finished_reason(status) is reason


def test_finish_reason_strings_match_the_openai_surface():
    assert str(FinishReason.STOP) == "stop"
    assert str(FinishReason.LENGTH) == "length"
    assert str(FinishReason.ABORT) == "abort"


# --- token accounting ------------------------------------------------------


def test_new_request_starts_waiting_with_the_prompt_counted():
    request = make_request(prompt=(1, 2, 3, 4))
    assert request.status is RequestStatus.WAITING
    assert request.num_prompt_tokens == 4
    assert request.num_tokens == 4
    assert request.num_output_tokens == 0
    assert request.num_computed_tokens == 0
    assert not request.is_finished()


def test_appending_output_advances_both_views_together():
    request = make_request(prompt=(1, 2))
    request.append_output_token_ids(7)
    request.append_output_token_ids([8, 9])
    assert list(request.output_token_ids) == [7, 8, 9]
    assert list(request.all_token_ids) == [1, 2, 7, 8, 9]
    assert request.num_tokens == 5
    assert request.num_output_tokens == 3


def test_token_views_refuse_direct_mutation():
    """Appending to one view without the other desyncs them, and the resulting bug
    surfaces far from its cause."""
    request = make_request()
    with pytest.raises(TypeError, match="read-only view"):
        request.output_token_ids.append(5)
    with pytest.raises(TypeError, match="read-only view"):
        request.all_token_ids.clear()


def test_constant_list_still_reads_like_a_list():
    view = ConstantList([1, 2, 3])
    assert view[0] == 1
    assert list(view) == [1, 2, 3]
    assert len(view) == 3


# --- construction ----------------------------------------------------------


def test_unresolved_max_tokens_is_rejected():
    """The processor resolves max_tokens against max_model_len. A Request built
    without that resolution would have no length cap at all."""
    params = SamplingParams(max_tokens=8)
    params.max_tokens = None
    with pytest.raises(ValueError, match="max_tokens must be resolved"):
        Request("r", [1], params, arrival_time=0.0)


def test_missing_sampling_params_is_rejected():
    with pytest.raises(ValueError, match="sampling_params must be set"):
        Request("r", [1], None, arrival_time=0.0)


def test_engine_core_stamps_arrival_time():
    """R19.1/R4.4: the frontend never reads a clock, so the core supplies the stamp."""
    wire = EngineCoreRequest(
        request_id="r0",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=4),
    )
    assert wire.arrival_time is None
    request = Request.from_engine_core_request(wire, arrival_time=1767225600.5)
    assert request.arrival_time == 1767225600.5


def test_an_arrival_time_already_on_the_wire_is_preserved():
    """Trace replay supplies its own arrival times (R20.2); the core must not
    overwrite them with the moment of receipt."""
    wire = EngineCoreRequest(
        request_id="r0",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=4),
        arrival_time=42.0,
    )
    request = Request.from_engine_core_request(wire, arrival_time=999.0)
    assert request.arrival_time == 42.0


# --- ordering --------------------------------------------------------------


def test_priority_ordering_is_priority_then_arrival_then_id():
    """R5.6."""
    high = make_request("a", arrival_time=10.0, priority=0)
    low = make_request("b", arrival_time=1.0, priority=5)
    assert high < low  # lower priority value wins regardless of arrival

    early = make_request("a", arrival_time=1.0)
    late = make_request("b", arrival_time=2.0)
    assert early < late


def test_request_id_breaks_arrival_ties():
    """Under a virtual clock many requests share an arrival instant, so without this
    tiebreak the scheduled order -- and therefore C1 -- would not be reproducible."""
    first = make_request("aaa", arrival_time=5.0)
    second = make_request("bbb", arrival_time=5.0)
    assert first < second
    assert sorted([second, first], key=lambda r: r.request_id)[0] is first


# --- events ----------------------------------------------------------------


def test_events_are_recorded_with_the_engine_clock_and_drained_once():
    request = make_request()
    request.record_event(EngineCoreEventType.QUEUED, timestamp=1.0)
    request.record_event(EngineCoreEventType.SCHEDULED, timestamp=2.0)

    events = request.take_events()
    assert events is not None
    assert [e.type for e in events] == [
        EngineCoreEventType.QUEUED,
        EngineCoreEventType.SCHEDULED,
    ]
    assert [e.timestamp for e in events] == [1.0, 2.0]
    assert request.take_events() is None


def test_block_hasher_is_injected_not_inlined():
    """F8/R6.3: the KV manager owns hashing policy; Request only stores the result."""
    calls = []

    def hasher(request: Request) -> list[int]:
        calls.append(request.num_tokens)
        return [request.num_tokens]

    request = make_request(prompt=(1, 2), block_hasher=hasher)
    assert calls == [2]
    request.append_output_token_ids(3)
    assert calls == [2, 3]
    assert request.block_hashes == [2, 3]


def test_without_a_hasher_no_hashes_accumulate():
    """Prefix caching lands in M2; until then the manager passes no hasher."""
    request = make_request()
    request.append_output_token_ids(9)
    assert request.block_hashes == []
