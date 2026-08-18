"""The inert-mechanism lint. R21.7.

Defect class two, in this project's own taxonomy: a mechanism that is present,
commented, sometimes tested, and changes nothing. Six were found by hand across seven
adversarial reviews -- a load-balancer rotation that rotated nothing anyone read, a
"padding added" warning that padded nothing, a drain-tail counter, a lockstep counter,
a background-task registry, and four exception registrations that duplicated a
try/except already in the route.

**This catches ONE of those six.** An earlier version of this docstring said the six
"share a shape an AST can see", and that was written without checking. Retro-tested,
one at a time:

* the load-balancer rotation wrote `self.scan_start`, a name that *is* read elsewhere
  -- a redundant write to a live name is invisible here;
* the "padding added" warning was a local in a module-level function, not `self.X`;
* the drain tail was a method nobody called;
* `self.lockstep_rounds` -- **caught**, the only one;
* `self.background_tasks` was read via `.get()`, so the name looked live while the
  dict stayed permanently empty (fixed in the M9 review, along with this claim);
* the four duplicate exception registrations were calls, not attributes.

So this checks one narrow shape: every `self.X = ...` in `pvllm/`, against every `.X`
read in the package, its tests and its tools. Two further limits, stated here rather
than left to be discovered. `read` is a flat set of attribute *names* carrying no type
information, so an inert `self.name` or `self.config` is invisible whenever anything
anywhere reads a field of that name -- about 1,500 names are live against 336 written,
so an ordinary name is effectively exempt. And module constants, dataclass and pydantic
fields, unused parameters and uncalled methods are all outside it by construction.

Narrow is deliberate: widening to pydantic fields would drown the signal, since
serialisation reads those implicitly and several exist precisely so a key appears on
the wire. But narrow is not general, and this file used to imply it was. What it is
worth is measurable -- its first run found five real inert mechanisms, two of them
carrying comments claiming they fed metrics that were never built.

An entry in ALLOWED needs a reason, and the reason has to survive being read aloud.
"Somebody might want it later" is how the six got there.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "pvllm"
#: Read sites are collected from the tests and tools too. An attribute a test asserts
#: on is doing a job -- it is how the simulator's own observability is checked.
READ_ROOTS = (REPO / "pvllm", REPO / "tests", REPO / "tools")

#: name -> why it is written and never read, and why that is correct.
ALLOWED: dict[str, str] = {
    # Tier B keeps upstream's attribute even where pvllm's runner does not consult it,
    # because the *call* is the point: `get_attn_backend_cls` validates the requested
    # backend and refuses MLA on a platform that cannot serve it. Dropping the
    # assignment would invite dropping the call with it.
    "attn_backend_cls": (
        "upstream's model-runner attribute; the call it stores validates the backend"
    ),
    # Upstream's manager carries its group id and pvllm's mirrors it. Cheap, and it is
    # what makes a manager identifiable in a debugger when several groups are live.
    "kv_cache_group_id": "mirrors upstream's manager attribute; identifies the group",
    # The app owns the registry's lifetime even though only the stat logger reads it;
    # two apps in one process must not share one, which is a property of holding it.
    "registry": "held so the app owns the collector registry's lifetime",
}


def _collect() -> tuple[dict[str, list[tuple[Path, int]]], set[str]]:
    written: dict[str, list[tuple[Path, int]]] = {}
    read: set[str] = set()

    for root in READ_ROOTS:
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - the suite would be failing anyway
                continue
            record = path.is_relative_to(PACKAGE)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    read.add(node.attr)
                # `getattr(x, "name")` is a read the AST would otherwise miss, and
                # pvllm uses it for optional collaborators.
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ("getattr", "hasattr")
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    read.add(node.args[1].value)
                if not record:
                    continue
                targets: list[ast.expr] = []
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                elif isinstance(node, ast.AugAssign):
                    # `self.x += 1` both reads and writes; it is a write for our
                    # purposes, because a counter nobody consumes is the classic case.
                    targets = [node.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        written.setdefault(target.attr, []).append(
                            (path, target.lineno)
                        )
    return written, read


def test_no_attribute_is_written_and_never_read():
    """An attribute assigned in the package and read nowhere is doing no work.

    The failure message names every site, because the fix is usually one of three
    things and the sites tell you which: wire it to whatever its comment claims it
    feeds, assert on it in a test if it is observability, or delete it.
    """
    written, read = _collect()

    inert = {
        name: sites
        for name, sites in written.items()
        if name not in read and name not in ALLOWED and not name.startswith("__")
    }

    if inert:
        lines = ["attributes written but never read:"]
        for name, sites in sorted(inert.items()):
            lines.append(f"  self.{name}")
            for path, lineno in sites:
                lines.append(f"      {path.relative_to(REPO)}:{lineno}")
        lines.append(
            "\nEach is written and read nowhere. Wire it to what its comment says it\n"
            "feeds, assert on it if it is observability, or delete it. If it is\n"
            "genuinely correct as-is, add it to ALLOWED with a reason."
        )
        raise AssertionError("\n".join(lines))


def test_the_allowlist_does_not_rot():
    """An ALLOWED entry that is now read, or no longer written at all, is a stale
    exemption -- and a stale exemption is how a lint quietly stops linting."""
    written, read = _collect()

    stale = []
    for name, reason in ALLOWED.items():
        if name not in written:
            stale.append(f"{name}: no longer written anywhere ({reason})")
        elif name in read:
            stale.append(f"{name}: now read somewhere, so the exemption is obsolete")
    assert not stale, "stale ALLOWED entries:\n  " + "\n  ".join(stale)


def test_every_allowlist_entry_states_a_reason():
    """A reason that is blank, or that says nothing, is not a reason."""
    for name, reason in ALLOWED.items():
        assert len(reason) > 20, f"{name}: the exemption needs an actual reason"
