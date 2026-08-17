"""The mutation runner's own tests. R21.6.

The catalogue is a safety net, and a net with a hole in it is worse than no net --
it reports "all mutations caught" either way. So the runner's own failure modes are
tested here rather than assumed: a mutation that goes unnoticed, an anchor that has
drifted, an anchor that matches twice, and a named test that does not exist.

Each case uses a throwaway file and a throwaway test, so nothing here depends on the
real catalogue staying the shape it is today.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import mutate

REAL_TEST = (
    "tests/entrypoints/test_responses_api.py::"
    "test_sequence_numbers_are_global_and_monotonic"
)


@pytest.fixture
def probe():
    """A throwaway module inside the package, so the runner's relative paths resolve."""
    target = mutate.REPO / "pvllm" / "_mutate_probe.py"
    yield target
    target.unlink(missing_ok=True)


def test_a_real_mutation_is_reported_as_caught():
    """The happy path, against a live catalogue entry: breaking the guarantee makes
    its test fail, and the runner calls that caught."""
    entry = next(
        m for m in mutate.load() if m.name == "responses-sequence-number-is-global"
    )
    caught, detail = mutate.run_one(entry)
    assert caught, detail
    assert detail == "caught"


def test_a_mutation_nothing_notices_is_reported_as_a_miss(probe):
    """The case the whole tool exists to surface: the edit lands, the named test still
    passes, and that is a finding rather than a success."""
    probe.write_text("VALUE = 1  # a comment\n")
    entry = mutate.Mutation(
        name="probe",
        why="a comment change cannot affect any test",
        file="pvllm/_mutate_probe.py",
        old="# a comment",
        new="# a different comment",
        test=REAL_TEST,
    )
    caught, detail = mutate.run_one(entry)
    assert not caught
    assert detail == "test PASSED with the guarantee broken"


def test_a_drifted_anchor_is_reported_rather_than_skipped(probe):
    """An entry whose `old` no longer appears has been pinning nothing, possibly for
    a long time. Silence here would be the worst outcome available."""
    probe.write_text("VALUE = 1\n")
    entry = mutate.Mutation(
        name="probe",
        why="...",
        file="pvllm/_mutate_probe.py",
        old="VALUE = 999",
        new="VALUE = 2",
        test=REAL_TEST,
    )
    caught, detail = mutate.run_one(entry)
    assert not caught
    assert "anchor not found" in detail


def test_an_ambiguous_anchor_is_reported(probe):
    """Two matches means the edit lands somewhere unintended, so the run proves
    nothing about the line the entry meant."""
    probe.write_text("VALUE = 1\nVALUE = 1\n")
    entry = mutate.Mutation(
        name="probe",
        why="...",
        file="pvllm/_mutate_probe.py",
        old="VALUE = 1",
        new="VALUE = 2",
        test=REAL_TEST,
    )
    caught, detail = mutate.run_one(entry)
    assert not caught
    assert "appears 2 times" in detail


def test_a_missing_test_is_not_mistaken_for_a_catch(probe):
    """pytest exits non-zero when it collects nothing, which naively reads as "the
    mutation was noticed". A typo in a node id would then look like coverage."""
    probe.write_text("VALUE = 1\n")
    entry = mutate.Mutation(
        name="probe",
        why="...",
        file="pvllm/_mutate_probe.py",
        old="VALUE = 1",
        new="VALUE = 2",
        test="tests/unit/test_mutate_tool.py::test_no_such_test_exists_anywhere",
    )
    caught, detail = mutate.run_one(entry)
    assert not caught
    assert "does not exist" in detail


def test_an_already_failing_test_is_not_mistaken_for_a_catch(probe, tmp_path):
    """Without a baseline the tool has no evidence the failure it saw was *caused* by
    the mutation -- only that the test was red while the mutation was applied.

    That is not hypothetical: this module's own dirty-tree guard deliberately permits
    running while a test file is being edited, on the grounds that this is exactly
    when you want to ask whether the test guards anything. A half-written test would
    then have every entry naming it certified as coverage.
    """
    probe.write_text("VALUE = 1\n")
    failing = mutate.REPO / "tests" / "unit" / "test_mutate_probe_failing.py"
    failing.write_text("def test_always_fails():\n    assert False\n")
    try:
        entry = mutate.Mutation(
            name="probe",
            why="...",
            file="pvllm/_mutate_probe.py",
            old="VALUE = 1",
            new="VALUE = 2",
            test=("tests/unit/test_mutate_probe_failing.py::test_always_fails"),
        )
        caught, detail = mutate.run_one(entry)
        assert not caught
        assert "ALREADY fails" in detail
    finally:
        failing.unlink(missing_ok=True)


def test_the_source_is_restored_even_when_the_test_fails(probe):
    """The edit is undone in a `finally`, so a failing run does not leave the tree
    mutated -- which is how a mutation could otherwise end up committed."""
    body = "VALUE = 1\n"
    probe.write_text(body)
    entry = mutate.Mutation(
        name="probe",
        why="...",
        file="pvllm/_mutate_probe.py",
        old="VALUE = 1",
        new="raise SystemExit('mutated')",
        test=REAL_TEST,
    )
    mutate.run_one(entry)
    assert probe.read_text() == body


def test_a_same_size_mutation_leaves_no_stale_bytecode():
    """The subtlest failure this tool had, and the one that makes it lie.

    CPython validates a `.pyc` against the source's mtime and size. A mutation that
    changes neither -- `+= 1` to `+= 0` -- can therefore run against the *original*
    bytecode (reporting a catch it never earned) and, worse, leave mutated bytecode
    behind after the restore, so a later unrelated run fails against code that is
    innocent on disk. That is how the bug was found.

    The check has to be set up deliberately: import the module first so a `.pyc` for
    the *unmutated* source exists, which is the state after any ordinary test run.
    Then a same-size mutation must still be detected. Removing the pre-run
    invalidation makes this report MISS, because the interpreter loads the cached
    original and the mutated source is never executed.
    """
    entry = next(
        m for m in mutate.load() if m.name == "responses-sequence-number-is-global"
    )
    assert len(entry.old) == len(entry.new), "this test needs a same-size mutation"

    target = mutate.REPO / entry.file
    subprocess.run(
        [sys.executable, "-c", "import pvllm.entrypoints.openai.responses.streaming"],
        cwd=mutate.REPO,
        capture_output=True,
        check=True,
    )
    assert list((target.parent / "__pycache__").glob(f"{target.stem}.*.pyc")), (
        "setup failed: no bytecode was cached, so this test proves nothing"
    )

    caught, detail = mutate.run_one(entry)
    assert caught, f"the mutation was not detected: {detail}"
    assert entry.old in target.read_text()


def test_every_catalogue_entry_names_a_test_that_exists():
    """Cheap, and it catches the commonest way an entry rots: a test renamed without
    the catalogue following. Checked by reading the files rather than by running
    pytest, so it costs nothing."""
    missing = []
    for entry in mutate.load():
        path = mutate.REPO / entry.test.split("::")[0]
        name = entry.test.split("::")[-1]
        if not path.exists() or f"def {name}(" not in path.read_text():
            missing.append(entry.name)
    assert not missing, f"catalogue entries naming a missing test: {missing}"


def test_every_catalogue_anchor_is_unique_in_its_file():
    """The same check `run_one` makes, hoisted so it runs in the ordinary suite: the
    catalogue rots when the code moves, and the fast suite should say so."""
    problems = []
    for entry in mutate.load():
        target = mutate.REPO / entry.file
        if not target.exists():
            problems.append(f"{entry.name}: {entry.file} does not exist")
            continue
        count = target.read_text().count(entry.old)
        if count != 1:
            problems.append(f"{entry.name}: anchor appears {count} times")
    assert not problems, "\n".join(problems)
