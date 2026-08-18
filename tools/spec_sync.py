#!/usr/bin/env python3
"""Check every module's declared upstream counterpart against the vendored tree.

pretending-vllm claims that module paths, class names, and method signatures mirror
upstream closely enough that a diff is a study exercise (G2). That claim decays
silently as upstream moves -- the requirements draft was written against an older
vLLM and had eight stale assumptions by the time the tree was actually checked (see
the delta table in UPSTREAM.md).

This turns that decay into a failing check. Every module declares its counterpart in
its docstring::

    Upstream: vllm/v1/core/sched/scheduler.py
    Tier: A

and this script verifies the counterpart still exists at the pin. Tier D modules
declare ``Upstream: (none -- simulator)``.

It also reports *coverage*: which upstream modules in the mirrored subset have no
pvllm counterpart yet. That is a progress tracker, not an error -- most of the tree
is intentionally never mirrored.

Usage::

    python tools/spec_sync.py             # check, human-readable
    python tools/spec_sync.py --coverage  # also list unported upstream modules
    python tools/spec_sync.py --json      # machine-readable

Exit code is non-zero when a declared counterpart is missing or a header is malformed.
Stdlib only: this runs in CI before the package is installed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "pvllm"
VENDOR_ROOT = REPO_ROOT / "vendor"

VALID_TIERS = {"A", "B", "C", "D"}

_UPSTREAM_RE = re.compile(r"^\s*Upstream:\s*(.+?)\s*$", re.MULTILINE)
_TIER_RE = re.compile(r"^\s*Tier:\s*([A-D])\s*$", re.MULTILINE)
#: Trailing notes like "(counterpart, not a port)" are documentation, not path.
_NOTE_RE = re.compile(r"\s*\(.*\)\s*$")

#: Upstream subtrees this project intends to mirror. Everything else upstream --
#: kernels, quantization, the model zoo, Ray, diffusion -- is out of scope by design,
#: so counting it as "unported" would drown the signal.
MIRRORED_SUBTREES = (
    "vllm/v1/core/",
    "vllm/v1/engine/",
    "vllm/v1/executor/",
    "vllm/v1/metrics/",
    "vllm/v1/sample/",
    "vllm/v1/worker/gpu/",
    "vllm/v1/structured_output/",
    "vllm/v1/spec_decode/",
    "vllm/config/",
    "vllm/entrypoints/openai/",
    "vllm/entrypoints/serve/",
    "vllm/entrypoints/cli/",
    "vllm/platforms/",
    "vllm/tokenizers/",
    "vllm/benchmarks/",
)
MIRRORED_FILES = (
    "vllm/v1/request.py",
    "vllm/v1/outputs.py",
    "vllm/v1/kv_cache_interface.py",
    "vllm/v1/serial_utils.py",
    "vllm/envs.py",
    "vllm/logger.py",
    "vllm/outputs.py",
    "vllm/sampling_params.py",
    "vllm/engine/arg_utils.py",
    "vllm/entrypoints/llm.py",
    "vllm/entrypoints/launcher.py",
)


@dataclass
class ModuleRecord:
    module: str
    upstream: str | None
    tier: str | None
    error: str | None = None
    #: Declared `(none -- pvllm addition)`: no upstream counterpart, but not
    #: simulator internals either.
    is_addition: bool = False


def find_upstream_dir() -> Path | None:
    """Locate the vendored tree without hardcoding the version twice."""
    candidates = sorted(VENDOR_ROOT.glob("vllm-*"))
    return candidates[-1] if candidates else None


def parse_module(path: Path) -> ModuleRecord:
    rel = str(path.relative_to(REPO_ROOT))
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return ModuleRecord(rel, None, None, f"syntax error: {exc}")

    docstring = ast.get_docstring(tree)
    if not docstring:
        if path.name == "__init__.py" and not path.read_text(encoding="utf-8").strip():
            return ModuleRecord(rel, None, None)  # empty package marker
        return ModuleRecord(rel, None, None, "no module docstring (NF5)")

    upstream_match = _UPSTREAM_RE.search(docstring)
    tier_match = _TIER_RE.search(docstring)

    if not upstream_match:
        return ModuleRecord(rel, None, None, "docstring has no `Upstream:` line (NF5)")
    if not tier_match:
        return ModuleRecord(rel, None, None, "docstring has no `Tier:` line")

    raw = upstream_match.group(1)
    tier = tier_match.group(1)
    is_addition = "pvllm addition" in raw
    upstream = _NOTE_RE.sub("", raw).strip()
    if upstream.startswith("(") or upstream.lower().startswith("none"):
        upstream = ""

    return ModuleRecord(rel, upstream or None, tier, is_addition=is_addition)


def check(upstream_dir: Path, records: list[ModuleRecord]) -> list[str]:
    problems: list[str] = []
    for record in records:
        if record.error:
            problems.append(f"{record.module}: {record.error}")
            continue
        if record.tier and record.tier not in VALID_TIERS:
            problems.append(f"{record.module}: invalid tier {record.tier!r}")

        if record.upstream is None:
            # A module with no counterpart must say why. Tier D (the simulator) is
            # the common case; a pvllm-only interface that sits *above* the boundary
            # -- the trace sink, for instance -- is neither a port nor simulator
            # internals, so it declares itself explicitly instead of being mistiered
            # into D just to satisfy this check.
            if record.tier and record.tier != "D" and not record.is_addition:
                problems.append(
                    f"{record.module}: tier {record.tier} declares no upstream "
                    f"counterpart. Either give it one, retier it to D (simulator "
                    f"internals), or mark it `Upstream: (none -- pvllm addition)` if "
                    f"it is a pvllm-only interface above the boundary."
                )
            continue

        if not (upstream_dir / record.upstream).is_file():
            problems.append(
                f"{record.module}: declared counterpart `{record.upstream}` does not "
                f"exist at the pin. It was renamed, moved, or removed upstream -- "
                f"find where it went and update the header, or retier the module."
            )
    return problems


def coverage(upstream_dir: Path, records: list[ModuleRecord]) -> list[str]:
    """Upstream modules in the mirrored subset with no pvllm counterpart yet."""
    declared = {r.upstream for r in records if r.upstream}
    unported: list[str] = []
    for path in sorted(upstream_dir.rglob("*.py")):
        rel = path.relative_to(upstream_dir).as_posix()
        if rel.endswith("__init__.py"):
            continue
        if not (rel.startswith(MIRRORED_SUBTREES) or rel in MIRRORED_FILES):
            continue
        if rel not in declared:
            unported.append(rel)
    return unported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage", action="store_true", help="list unported upstream modules"
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    upstream_dir = find_upstream_dir()
    if upstream_dir is None:
        print(
            "No vendored upstream tree found. Run `python tools/fetch_upstream.py`.",
            file=sys.stderr,
        )
        return 2

    records = [parse_module(p) for p in sorted(PACKAGE_ROOT.rglob("*.py"))]
    problems = check(upstream_dir, records)
    unported = coverage(upstream_dir, records) if args.coverage else []

    if args.as_json:
        print(
            json.dumps(
                {
                    "upstream_dir": str(upstream_dir.relative_to(REPO_ROOT)),
                    "modules": [asdict(r) for r in records],
                    "problems": problems,
                    "unported": unported,
                },
                indent=2,
            )
        )
        return 1 if problems else 0

    by_tier: dict[str, int] = {}
    for record in records:
        if record.tier:
            by_tier[record.tier] = by_tier.get(record.tier, 0) + 1

    print(f"upstream:  {upstream_dir.relative_to(REPO_ROOT)}")
    print(
        f"modules:   {len(records)}  "
        + "  ".join(f"tier {t}: {n}" for t, n in sorted(by_tier.items()))
    )

    if args.coverage:
        print(
            f"\nnot yet ported ({len(unported)} upstream modules in the mirrored subset):"
        )
        for rel in unported:
            print(f"  {rel}")

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print("\nall declared upstream counterparts exist at the pin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
