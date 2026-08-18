#!/usr/bin/env python3
"""Fetch the pinned upstream vLLM source tree into ``vendor/``.

pretending-vllm mirrors upstream module paths, class names, and method signatures
(G2). That only stays true if the upstream tree is available to diff against, so
every module carries an ``Upstream:`` header naming its counterpart and
``tools/spec_sync.py`` resolves those headers against the tree this script fetches.

The tree itself is *not* committed -- it is ~29 MB of third-party source. What is
committed is ``vendor/MANIFEST.sha256``, so a vendored tree can be verified as
byte-identical to the one the port was written against.

Usage::

    python tools/fetch_upstream.py            # fetch (no-op if manifest matches)
    python tools/fetch_upstream.py --force    # re-fetch even if present
    python tools/fetch_upstream.py --check    # verify only, never download
    python tools/fetch_upstream.py --write-manifest

Stdlib only, on purpose: this runs before ``pip install -e .``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

UPSTREAM_VERSION = "0.27.1"
UPSTREAM_TAG = f"v{UPSTREAM_VERSION}"
TARBALL_URL = (
    f"https://github.com/vllm-project/vllm/archive/refs/tags/{UPSTREAM_TAG}.tar.gz"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = REPO_ROOT / "vendor"
UPSTREAM_DIR = VENDOR_ROOT / f"vllm-{UPSTREAM_VERSION}"
MANIFEST_PATH = VENDOR_ROOT / "MANIFEST.sha256"

# Only text sources are vendored. Kernels (.cu/.cuh/.cpp) and test data are
# irrelevant to a port that has no device code, and they dominate the tarball.
KEEP_SUFFIXES = {".py", ".pyi", ".json", ".toml"}
KEEP_ROOTS = ("vllm/",)
KEEP_EXTRA_FILES = ("pyproject.toml",)


def _iter_vendored_files() -> list[Path]:
    if not UPSTREAM_DIR.is_dir():
        return []
    return sorted(p for p in UPSTREAM_DIR.rglob("*") if p.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_manifest() -> dict[str, str]:
    """Path -> digest, keyed with forward slashes on every platform.

    `str(Path)` renders `vllm\\benchmarks\\x.py` on Windows, so every entry read
    from the committed manifest counted as missing and every entry computed here
    counted as unexpected -- 2,735 of each, with 0 changed, which is the signature of
    a key mismatch rather than a corrupted tree. `as_posix()` is the whole fix; the
    manifest is a cross-platform artifact and has to be keyed like one.
    """
    return {
        p.relative_to(UPSTREAM_DIR).as_posix(): _sha256(p)
        for p in _iter_vendored_files()
    }


def read_manifest() -> dict[str, str]:
    if not MANIFEST_PATH.is_file():
        return {}
    entries: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        entries[rel] = digest
    return entries


def write_manifest(manifest: dict[str, str]) -> None:
    lines = [
        "# pretending-vllm vendored upstream manifest",
        f"# upstream: vllm {UPSTREAM_TAG} ({TARBALL_URL})",
        f"# files: {len(manifest)}",
        "",
    ]
    lines += [f"{digest}  {rel}" for rel, digest in sorted(manifest.items())]
    MANIFEST_PATH.write_text("\n".join(lines) + "\n")


def _wanted(member_path: str) -> bool:
    """Decide whether a tar member (path already stripped of its top dir) is kept."""
    if member_path in KEEP_EXTRA_FILES:
        return True
    if not member_path.startswith(KEEP_ROOTS):
        return False
    return Path(member_path).suffix in KEEP_SUFFIXES


def fetch() -> None:
    print(f"fetching {TARBALL_URL}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "upstream.tar.gz"
        with urllib.request.urlopen(TARBALL_URL) as response:
            tarball.write_bytes(response.read())

        staging = Path(tmp) / "staging"
        staging.mkdir()
        count = 0
        with tarfile.open(tarball, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                # Strip the "vllm-0.27.1/" prefix GitHub adds.
                rel = member.name.partition("/")[2]
                if not rel or not _wanted(rel):
                    continue
                target = staging / rel
                if not target.resolve().is_relative_to(staging.resolve()):
                    raise RuntimeError(f"unsafe tar member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                target.write_bytes(extracted.read())
                count += 1

        if count == 0:
            raise RuntimeError("tarball contained no matching files -- layout changed?")

        if UPSTREAM_DIR.exists():
            shutil.rmtree(UPSTREAM_DIR)
        UPSTREAM_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(UPSTREAM_DIR))

    print(
        f"vendored {count} files into {UPSTREAM_DIR.relative_to(REPO_ROOT)}",
        file=sys.stderr,
    )


def verify() -> int:
    """Compare the vendored tree against the committed manifest. Returns exit code."""
    expected = read_manifest()
    if not expected:
        print("no vendor/MANIFEST.sha256 -- nothing to verify against", file=sys.stderr)
        return 1
    actual = compute_manifest()
    if actual == expected:
        print(f"vendor tree matches manifest ({len(expected)} files)", file=sys.stderr)
        return 0

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(r for r in set(expected) & set(actual) if expected[r] != actual[r])
    for rel in missing[:20]:
        print(f"  missing: {rel}", file=sys.stderr)
    for rel in extra[:20]:
        print(f"  unexpected: {rel}", file=sys.stderr)
    for rel in changed[:20]:
        print(f"  changed: {rel}", file=sys.stderr)
    print(
        f"vendor tree does NOT match manifest "
        f"({len(missing)} missing, {len(extra)} unexpected, {len(changed)} changed)",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-fetch even if present")
    parser.add_argument(
        "--check", action="store_true", help="verify only, never download"
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="regenerate vendor/MANIFEST.sha256 from the vendored tree",
    )
    args = parser.parse_args()

    if args.check:
        return verify()

    if args.write_manifest:
        if not UPSTREAM_DIR.is_dir():
            print(f"{UPSTREAM_DIR} does not exist -- fetch first", file=sys.stderr)
            return 1
        write_manifest(compute_manifest())
        print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 0

    if UPSTREAM_DIR.is_dir() and not args.force:
        if verify() == 0:
            return 0
        print("re-fetching to reconcile", file=sys.stderr)

    fetch()
    if not MANIFEST_PATH.is_file():
        write_manifest(compute_manifest())
        print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 0
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
