"""C1--C4, against recorded goldens. R21.3.

The contract calls these *exact*: a divergence is a bug by definition. That only
means something if a divergence is detected, which is what these tests do.

**They are in `asserted` mode.** The goldens were recorded from pretending-vllm, so a
pass means "we behave as we did", not "we behave as vLLM does". `test_goldens_declare_
their_source` is what stops that distinction from quietly eroding, and D4 describes
the promotion path.

When one of these fails, the failure names the conformance class and prints what the
workload exists to pin. Read that before regenerating anything: `capture_golden_trace
.py --force` is for intended changes, and reaching for it to make the suite green is
how a regression suite stops being one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pvllm import UPSTREAM_VERSION
from pvllm.conformance import ConformanceRecord, compare, record_workload
from pvllm.conformance_workloads import WORKLOADS

GOLDEN_DIR = Path(__file__).parent / "goldens"
WORKLOAD_NAMES = sorted(WORKLOADS)


@pytest.fixture(scope="module")
def recordings(tmp_path_factory):
    """Records a workload on first request, then reuses it.

    Cached because every test below asks a different question of the same recording
    and R21.5 budgets the whole suite at 30 seconds. Lazy per workload rather than
    all-up front so that a workload which *crashes* the engine fails only its own
    tests -- eagerly recording all four would turn one broken scheduler path into a
    dozen setup errors across unrelated workloads, and the real failure would be the
    hardest one to find in the output.
    """
    directory = tmp_path_factory.mktemp("conformance")
    cache: dict[str, ConformanceRecord] = {}

    def get(name: str) -> ConformanceRecord:
        if name not in cache:
            cache[name] = record_workload(WORKLOADS[name], trace_dir=directory)
        return cache[name]

    return get


def golden(name: str) -> ConformanceRecord:
    path = GOLDEN_DIR / f"{name}.json"
    if not path.exists():
        pytest.fail(
            f"no golden for workload {name!r}. Record one with:\n"
            f"    python tools/capture_golden_trace.py {name}"
        )
    return ConformanceRecord.read(path)


@pytest.mark.parametrize("name", WORKLOAD_NAMES)
def test_workload_matches_its_golden(name: str, recordings) -> None:
    """C1--C4 for one workload, all at once.

    One test rather than four, because the classes are not independent: a scheduler
    that admits a request one step early allocates its blocks one step early too, and
    four separate failures describing one cause is noise. `compare` orders the
    differences so the first one printed is the root.
    """
    differences = compare(recordings(name), golden(name))
    assert not differences, (
        f"\n{WORKLOADS[name].pins}\n\n"
        + "\n".join(f"  - {d}" for d in differences)
        + f"\n\nIf this change was intended:\n"
        f"    python tools/capture_golden_trace.py {name} --force"
    )


@pytest.mark.parametrize("name", WORKLOAD_NAMES)
def test_recording_is_deterministic(name: str, recordings, tmp_path) -> None:
    """B4 at the level the contract is asserted on.

    Comparing a recording to a golden proves nothing if the recording itself varies
    run to run -- the suite would be flaky and every failure ambiguous.
    """
    again = record_workload(WORKLOADS[name], trace_dir=tmp_path)
    assert not compare(again, recordings(name))


@pytest.mark.parametrize("name", WORKLOAD_NAMES)
def test_goldens_declare_their_source(name: str) -> None:
    """D4: `asserted` and `verified` must never be confusable.

    A golden that lost its provenance would let a self-consistency pass be read as
    conformance to upstream, which is the one claim this project must not make
    loosely.
    """
    record = golden(name)
    assert record.source in ("pretending-vllm", "vllm")
    assert record.upstream_version == UPSTREAM_VERSION, (
        f"golden for {name} was recorded against upstream "
        f"{record.upstream_version}, but this build pins {UPSTREAM_VERSION}. "
        f"See UPSTREAM.md, 'Bumping the pin'."
    )


def test_the_contract_is_still_asserted_not_verified() -> None:
    """The README's status line and the goldens must agree.

    If someone records goldens from real vLLM, this fails -- which is the prompt to
    promote the contract in the README rather than leaving it understated.
    """
    sources = {golden(name).source for name in WORKLOAD_NAMES}
    readme = (Path(__file__).parents[2] / "README.md").read_text(encoding="utf-8")

    if sources == {"vllm"}:
        pytest.fail(
            "every golden now comes from real vLLM. The contract can be promoted to "
            "`verified` in README.md and UPSTREAM.md."
        )
    assert "`asserted`, not `verified`" in readme, (
        "goldens are still self-recorded, so the README must not claim `verified`"
    )


# --- what the workloads are supposed to exercise ---------------------------
#
# A golden comparison passes just as happily against a workload that exercises
# nothing. These assert that each one still does the thing it was built for -- so a
# config change that quietly defuses a workload fails here rather than leaving a
# green test that checks nothing.


def test_the_preemption_workload_actually_preempts(recordings) -> None:
    record = recordings("preemption")
    assert record.preemptions["total"] > 0, (
        "the preemption workload no longer preempts, so C4 is unasserted. Its block "
        "budget is probably too generous now."
    )
    assert record.preemptions["by_step"], "preemptions were counted but not attributed"


def test_the_shared_prefix_workload_actually_hits(recordings) -> None:
    record = recordings("shared-prefix")
    assert record.prefix_cache["hits"] > 0, (
        "the shared-prefix workload no longer hits the cache, so C3 is unasserted."
    )
    assert record.block_hashes, "no resident block hashes, so C3's values are unpinned"


def test_the_chunked_prefill_workload_actually_chunks(recordings) -> None:
    record = recordings("chunked-prefill")
    long_request = "0"
    chunks = [
        step["num_scheduled_tokens"][long_request]
        for step in record.steps
        if long_request in step["num_scheduled_tokens"]
    ]
    # More than one prefill-sized chunk means the prompt really was split.
    assert sum(1 for size in chunks if size > 1) > 1, (
        f"the long prompt was not chunked; per-step sizes were {chunks}"
    )


def test_the_mixed_workload_queues(recordings) -> None:
    record = recordings("mixed-lengths")
    assert any(step["num_waiting"] > 0 for step in record.steps), (
        "no request ever waited, so admission order is unasserted"
    )
