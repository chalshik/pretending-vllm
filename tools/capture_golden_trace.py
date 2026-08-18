#!/usr/bin/env python3
"""Record conformance goldens. R21.3, D4.

Upstream: (none -- pvllm addition)
Tier: D

Two jobs, and the difference between them is the whole of D4.

**Re-recording our own goldens** (`--source pretending-vllm`, the default) refreshes
what the suite compares against. That is only ever correct when a behavioral change
was *intended* -- a scheduler fix, a new upstream pin. Running it to make a red suite
green is how a regression suite stops being one, so the tool prints the diff it is
about to overwrite and requires `--force` when a golden already exists.

**Recording from real vLLM** (`--source vllm`) is the promotion path. The contract is
`asserted` today: our goldens catch drift from our own past behavior, not divergence
from upstream. Somebody with a GPU runs the same workloads against real vLLM at the
pin, and those recordings replace ours -- the tests do not change, only what they
compare against, and the README's contract state becomes `verified`.

That capture is not automated from here -- it needs a GPU and a `vllm` install, so
this tool refuses `--source vllm` rather than pretending it can. The procedure, on a
machine that has both:

1. `pip install vllm==0.27.1`.
2. Drive `vllm.LLM` with `WORKLOADS[name].prompts` and `engine_kwargs()`.
3. Attach `pvllm.conformance.BlockPoolRecorder` to the engine's `BlockPool`. It wraps
   `get_new_blocks` and `free_blocks` structurally, and both exist under those names
   upstream at the pin, so it attaches unchanged.
4. Build a `ConformanceRecord` with `source="vllm"` and write it over the golden.

The tests do not change. Only what they compare against does.

Note the one honest asymmetry: upstream salts its none-hash from `os.urandom` unless
`PYTHONHASHSEED` is set, so C3's hash *values* will not match ours. Set
`PYTHONHASHSEED=0` on the capture machine to compare them, or accept that hit rates
and allocation order compare and hash values do not (`compare(...,
compare_hash_values=False)`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pvllm.conformance import ConformanceRecord, compare, record_workload  # noqa: E402
from pvllm.conformance_workloads import WORKLOADS  # noqa: E402

DEFAULT_GOLDEN_DIR = REPO_ROOT / "tests" / "conformance" / "goldens"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "workloads",
        nargs="*",
        help=f"which workloads to record (default: all of {sorted(WORKLOADS)})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_GOLDEN_DIR),
        help="where goldens are written",
    )
    parser.add_argument(
        "--source",
        default="pretending-vllm",
        choices=["pretending-vllm", "vllm"],
        help=(
            "which engine produced the recording. Stamped into the golden, and it is "
            "what decides whether a passing suite means `asserted` or `verified`."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite goldens that already exist and differ",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report differences and write nothing; exits non-zero if any differ",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help=(
            "record the C6 metric surface (names, types, labels, bucket edges) "
            "instead of the C1--C4 workloads"
        ),
    )
    args = parser.parse_args(argv)

    if args.metrics:
        return _capture_metrics(Path(args.output_dir), force=args.force)

    if args.source == "vllm":
        parser.error(
            "recording from real vLLM is not automated from this repo -- it needs a "
            "GPU and a vllm install. See the module docstring for the procedure; it "
            "reuses pvllm.conformance.record_workload's recorder against vllm.LLM."
        )

    names = args.workloads or sorted(WORKLOADS)
    unknown = [name for name in names if name not in WORKLOADS]
    if unknown:
        parser.error(f"unknown workloads {unknown}; expected {sorted(WORKLOADS)}")

    output_dir = Path(args.output_dir)
    exit_code = 0

    for name in names:
        workload = WORKLOADS[name]
        record = record_workload(workload, source=args.source)
        path = output_dir / f"{name}.json"

        if not path.exists():
            if args.check:
                print(f"{name}: no golden at {path}")
                exit_code = 1
                continue
            record.write(path)
            print(f"{name}: recorded {len(record.steps)} steps -> {path}")
            continue

        differences = compare(record, ConformanceRecord.read(path))
        if not differences:
            print(f"{name}: unchanged ({len(record.steps)} steps)")
            continue

        print(f"\n{name}: {len(differences)} difference(s) from the existing golden")
        print(f"  what this workload pins: {workload.pins}")
        for difference in differences:
            print(f"  - {difference}")

        if args.check:
            exit_code = 1
        elif args.force:
            record.write(path)
            print(f"  overwritten -> {path}")
        else:
            print(
                "  NOT overwritten. If this change was intended, re-run with "
                "--force. If it was not, the suite just caught a regression."
            )
            exit_code = 1

    return exit_code


def _capture_metrics(output_dir: Path, *, force: bool) -> int:
    """Record the C6 surface.

    Imports the test's own scraper rather than reimplementing it: two definitions of
    "the metric surface" would drift, and the one in the test is the one that decides
    whether the suite passes.
    """
    import json

    sys.path.insert(0, str(REPO_ROOT / "tests" / "conformance"))
    from prometheus_client import CollectorRegistry
    from test_c5_to_c7 import (  # type: ignore[import-not-found]
        make_app,
        scrape_families,
    )

    registry = CollectorRegistry()
    make_app(registry)
    families = scrape_families(registry)

    path = output_dir / "metrics.json"
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == families:
            print(f"metrics: unchanged ({len(families)} families)")
            return 0
        print(f"metrics: surface differs from {path}")
        print("  NOT overwritten. Re-run with --force if the change was intended.")
        return 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(families, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"metrics: recorded {len(families)} families -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
