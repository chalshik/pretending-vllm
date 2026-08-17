"""Run the mutation catalogue: break a guarantee, assert its test notices. R21.6.

Tier: D (no upstream counterpart -- this is pretending-vllm's own tooling).

A green suite says the tests pass. It does not say they would fail if the code were
wrong, and on this project that gap has been real often enough to be worth automating:
mutation testing has caught a non-discriminating test nearly every time it has been
run, including three in the two commits before this tool existed. Until now it was run
from memory, on freshly written tests only.

Each catalogue entry names a guarantee, the minimal edit that breaks it, and the test
that should notice. The tool applies the edit, runs that one test, and expects a
FAILURE. A test that stays green is reported: it is not guarding what its name claims.

    python tools/mutate.py                 # every entry
    python tools/mutate.py -k responses    # entries whose name matches
    python tools/mutate.py --list          # names only, run nothing

The edits are applied to the working tree and restored in a `finally`. If the tool is
killed hard enough to skip that, `git checkout <file>` restores it -- and the tool
refuses to start when a file it would edit has uncommitted changes, so that recovery
can never cost you work.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATALOGUE = REPO / "tests" / "mutations.toml"


@dataclass(frozen=True)
class Mutation:
    name: str
    why: str
    file: str
    old: str
    new: str
    test: str


def load(path: Path = CATALOGUE) -> list[Mutation]:
    raw = tomllib.loads(path.read_text())
    return [Mutation(**entry) for entry in raw["mutation"]]


def _targets_are_unmodified(mutations: list[Mutation]) -> list[str]:
    """Uncommitted changes in the files this run will edit. Empty means safe.

    Scoped to the mutation targets, not the whole tree. The hazard is narrow: the tool
    rewrites a source file and restores it in a `finally`, and if it were killed hard
    enough to skip that, the recovery is `git checkout <file>` -- which would also
    discard uncommitted work in that same file. Nothing else is at risk. Guarding the
    whole tree instead would refuse to run while a *test* is being edited, which is
    exactly when you most want to ask whether that test guards anything.
    """
    targets = sorted({mutation.file for mutation in mutations})
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *targets],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _run_test(node_id: str) -> subprocess.CompletedProcess[str]:
    """One pytest invocation, used for both the baseline and the mutated run.

    Shared so the two are identical in every respect but the state of the source --
    a baseline run with different flags would be comparing two different things.
    """
    return subprocess.run(
        [sys.executable, "-B", "-m", "pytest", node_id, "-q", "--no-header"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _invalidate_bytecode(target: Path) -> None:
    """Drop any cached bytecode for `target`.

    Load-bearing, and the reason is nastier than it looks. CPython decides a `.pyc` is
    current by comparing the source's *mtime and size* against the values in the pyc
    header. A mutation like `sequence += 1` -> `sequence += 0` changes neither size nor
    -- within one filesystem tick -- mtime. Two distinct failures follow, and they
    point in OPPOSITE directions:

      * the mutated run loads the ORIGINAL bytecode, so the mutation never executes,
        the test passes, and the entry is reported as a MISS it did not deserve --
        noisy and self-correcting, because someone investigates a miss; and
      * the restore is equally invisible, leaving mutated bytecode live in
        `__pycache__` with an innocent source on disk, so a LATER unrelated run fails
        -- and that is the one that manufactures a false catch.

    An earlier version of this docstring had the first bullet producing the false
    catch. It cannot: a mutation that does not execute leaves the test green, and
    `run_one` classifies exit 0 as a miss unconditionally.

    Both were observed. The second is how it was found: a comment-only edit to an
    unrelated file "caught" a mutation, because a same-size edit from an earlier run
    was still live in `__pycache__`.
    """
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for compiled in cache.glob(f"{target.stem}.*.pyc"):
        compiled.unlink(missing_ok=True)


def run_one(mutation: Mutation, *, verbose: bool = False) -> tuple[bool, str]:
    """Apply, run the one test, restore. True means the test failed, which is a pass.

    The catalogue is only as good as its anchors, so a missing or ambiguous `old` is
    an error rather than a skip: it means the code moved and the entry has been
    silently pinning nothing.
    """
    target = REPO / mutation.file
    original = target.read_text()

    occurrences = original.count(mutation.old)
    if occurrences == 0:
        return False, "anchor not found -- the code moved; update the entry"
    if occurrences > 1:
        return False, f"anchor appears {occurrences} times -- make it unique"

    # The baseline. Without it the tool has no evidence that the failure it is about
    # to see was *caused* by the mutation -- only that the test is red while the
    # mutation is applied. An already-broken test would certify every entry naming it,
    # which is the exact opposite of what the tool is for, and the workflow this
    # module's own guard recommends (running while a test is being edited) is where
    # that bites. Cheap insurance: one extra run per entry.
    baseline = _run_test(mutation.test)
    if baseline.returncode != 0:
        if "no tests ran" in baseline.stdout or baseline.returncode in (4, 5):
            return False, "the named test does not exist"
        return False, "the named test ALREADY fails before the mutation is applied"

    try:
        target.write_text(original.replace(mutation.old, mutation.new, 1))
        _invalidate_bytecode(target)
        result = _run_test(mutation.test)
    finally:
        target.write_text(original)
        _invalidate_bytecode(target)

    if verbose:
        print(result.stdout[-2000:])

    # pytest exits 0 only when everything passed. Anything else -- a failure, a
    # collection error -- means the mutation was noticed.
    if result.returncode == 0:
        return False, "test PASSED with the guarantee broken"
    if "no tests ran" in result.stdout or result.returncode == 4:
        return False, "the named test does not exist"
    return True, "caught"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", dest="pattern", help="only entries whose name matches")
    parser.add_argument("--list", action="store_true", help="list entries and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    mutations = load()
    if args.pattern:
        mutations = [m for m in mutations if args.pattern in m.name]

    if args.list:
        for mutation in mutations:
            print(f"{mutation.name}\n    {mutation.why}\n    -> {mutation.test}")
        return 0

    if not mutations:
        print("no matching entries", file=sys.stderr)
        return 1

    dirty = _targets_are_unmodified(mutations)
    if dirty:
        print(
            "refusing to run: these files have uncommitted changes and this run would\n"
            "rewrite them:\n"
            + "".join(f"  {name}\n" for name in dirty)
            + "The tool restores what it edits, but if it were killed before that, the\n"
            "recovery is `git checkout <file>` -- which would discard your work too.",
            file=sys.stderr,
        )
        return 1

    failures: list[tuple[Mutation, str]] = []
    for index, mutation in enumerate(mutations, 1):
        caught, detail = run_one(mutation, verbose=args.verbose)
        mark = "ok  " if caught else "MISS"
        print(f"[{index:2}/{len(mutations)}] {mark} {mutation.name}")
        if not caught:
            print(f"          {detail}")
            print(f"          guarantee: {mutation.why}")
            print(f"          test:      {mutation.test}")
            failures.append((mutation, detail))

    print()
    if failures:
        print(f"{len(failures)} of {len(mutations)} mutations were NOT caught:")
        for mutation, detail in failures:
            print(f"  - {mutation.name}: {detail}")
        print(
            "\nA mutation that is not caught means the named test passes whether or not\n"
            "the behaviour holds. Fix the test, not the catalogue."
        )
        return 1

    print(f"all {len(mutations)} mutations caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
