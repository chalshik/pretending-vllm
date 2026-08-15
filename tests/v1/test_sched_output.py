"""SchedulerOutput -- the structure that crosses the simulation boundary. F7, R5.10."""

from __future__ import annotations

from pvllm.sampling_params import SamplingParams
from pvllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from pvllm.v1.request import Request


def make_request(request_id="r0", prompt=(1, 2, 3)) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=8),
        arrival_time=0.0,
    )


def test_empty_step_is_representable():
    """A step that scheduled nothing is still a step; the engine counts it so the
    trace stays gap-free."""
    output = SchedulerOutput.make_empty()
    assert output.total_num_scheduled_tokens == 0
    assert output.scheduled_new_reqs == []
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert output.finished_req_ids == set()


def test_new_request_data_carries_block_ids_per_group():
    """R6.7: block ids are a tuple indexed by KV cache group, single-element until
    hybrid models land."""
    request = make_request(prompt=(1, 2, 3, 4))
    data = NewRequestData.from_request(request, block_ids=([0, 1],))
    assert data.req_id == "r0"
    assert data.block_ids == ([0, 1],)
    assert data.num_computed_tokens == 0
    assert data.prompt_token_ids == [1, 2, 3, 4]


def test_cached_request_data_is_parallel_arrays():
    """F7: an array-of-struct shape here would allocate one object per running
    request per step. This is the hot path."""
    data = CachedRequestData(
        req_ids=["a", "b"],
        resumed_req_ids={"b"},
        new_token_ids=[[5], [6]],
        new_block_ids=[([2],), None],
        num_computed_tokens=[10, 20],
        num_output_tokens=[1, 2],
    )
    assert data.num_reqs == 2
    # Index i of every array refers to req_ids[i].
    assert data.new_token_ids[data.req_ids.index("b")] == [6]
    assert data.new_block_ids[data.req_ids.index("b")] is None


def test_trace_dict_sorts_sets_for_reproducibility():
    """R5.10/C1. Set iteration order is not stable across runs; dumping a set
    directly would show up as a spurious difference in every conformance diff."""
    output = SchedulerOutput.make_empty()
    output.finished_req_ids = {"zeta", "alpha", "mid"}
    output.preempted_req_ids = {"q", "b"}
    output.num_scheduled_tokens = {"z": 1, "a": 2}

    record = output.to_trace_dict()
    assert record["finished_req_ids"] == ["alpha", "mid", "zeta"]
    assert record["preempted_req_ids"] == ["b", "q"]
    assert list(record["num_scheduled_tokens"]) == ["a", "z"]


def test_trace_dict_is_stable_across_equivalent_constructions():
    """Two runs that scheduled the same work must serialize identically, whatever
    order the sets happened to be built in."""

    def build(order):
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = set(order)
        return output.to_trace_dict()

    assert build(["a", "b", "c"]) == build(["c", "a", "b"])


def test_trace_dict_records_the_decision_not_the_payload():
    """The trace is for understanding and conformance, so it carries ids and counts
    rather than token arrays that would bloat every record."""
    request = make_request()
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [NewRequestData.from_request(request, ([0],))]
    output.num_scheduled_tokens = {"r0": 3}
    output.total_num_scheduled_tokens = 3

    record = output.to_trace_dict()
    assert record["new_reqs"] == ["r0"]
    assert record["num_scheduled_tokens"] == {"r0": 3}
    assert record["total_num_scheduled_tokens"] == 3
    assert "prompt_token_ids" not in record
