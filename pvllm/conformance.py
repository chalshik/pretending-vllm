"""The fidelity contract, as a recordable artifact. R21.3, D4.

Upstream: (none -- pvllm addition)
Tier: B

The README states a contract: C1--C4 are *exact*, and a divergence is a bug by
definition. A claim like that is worth nothing unless something checks it, and
checking it means being able to record what an engine decided, in a form two engines
can be compared in.

That is what this module produces. `record_workload` runs a fixed workload and returns
a `ConformanceRecord` -- the decisions, and nothing else.

**Decisions, not durations.** The record deliberately excludes timestamps, step
durations, and everything else the cost model touches. C1--C4 are about what the
scheduler and KV manager *chose*; latency is approximate by construction (R9.5) and
carries a published error band. If both lived in one artifact, every cost-model
recalibration would fail the conformance suite, goldens would get regenerated
reflexively to make it green, and the signal that a real scheduler regression is
supposed to produce would be gone by the third time. Keeping them apart is what makes
a failure here mean something.

**`source` is load-bearing.** Every record says which engine produced it. While the
contract is `asserted` (D4), goldens are pretending-vllm recordings and the suite
catches drift from our own past behavior -- not divergence from upstream. Once
somebody with hardware records the same workloads from real vLLM, those goldens
replace these, the same tests compare against them, and the contract is `verified`.
`compare` refuses to let the difference go unnoticed.

The recorder attaches to a `BlockPool` by wrapping two bound methods and touches
nothing else, so `attach()` works on *upstream's* `BlockPool` unchanged --
`get_new_blocks` and `free_blocks` have the same names and signatures at the pin.
(It does not wrap `cache_full_blocks`; an earlier version of this note claimed it
did.)

`snapshot_hashes` is the part that does **not** transfer. The keys are the same type
in both trees, but upstream wraps the map in a `BlockHashToBlockMap` that is not
iterable at the pin, so a `vllm`-sourced capture has to supply its own hash snapshot.
Everything else in a record (C1, C2, C4, and hit *rates*) crosses unchanged, which is
why `compare(..., compare_hash_values=False)` exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pvllm.conformance_workloads import Workload

#: Bump when a section's meaning changes. Adding a section is backward compatible;
#: `compare` ignores sections a golden does not carry.
CONFORMANCE_SCHEMA_VERSION = 1


class _BlockPoolLike(Protocol):
    """The slice of a block pool the recorder wraps.

    Structural rather than an import of `BlockPool`, so the recorder attaches to
    upstream's pool unchanged.
    """

    def get_new_blocks(self, num_blocks: int) -> list[Any]: ...
    def free_blocks(self, ordered_blocks: Any) -> None: ...


@dataclass
class BlockPoolRecorder:
    """Records block allocation and free *order*. C2.

    Order, not just counts: two allocators that hand out the same number of blocks in
    a different order produce different eviction behavior later, and the counts would
    look identical right up until the cache started missing.

    Attaches by replacing bound methods on one pool instance. Nothing in the engine
    knows it is there, which is the point -- an engine that behaved differently while
    being recorded would make the recording worthless.
    """

    allocations: list[list[int]] = field(default_factory=list)
    frees: list[list[int]] = field(default_factory=list)
    cached_hashes: list[str] = field(default_factory=list)

    def attach(self, pool: _BlockPoolLike) -> None:
        original_get = pool.get_new_blocks
        original_free = pool.free_blocks

        def get_new_blocks(num_blocks: int) -> list[Any]:
            blocks = original_get(num_blocks)
            if blocks:
                self.allocations.append([b.block_id for b in blocks])
            return blocks

        def free_blocks(ordered_blocks: Any) -> None:
            # Materialized because upstream passes a `reversed(...)` iterator and
            # consuming it here would free nothing (R6.6's ordering is the thing
            # being recorded, so the iterator has to survive the recording).
            blocks = list(ordered_blocks)
            if blocks:
                self.frees.append([b.block_id for b in blocks])
            original_free(blocks)

        pool.get_new_blocks = get_new_blocks  # type: ignore[method-assign]
        pool.free_blocks = free_blocks  # type: ignore[method-assign]

    def snapshot_hashes(self, pool: Any) -> None:
        """Record the resident block hashes. C3.

        Sorted, because the cache map's iteration order is insertion order and two
        engines that cached the same content in a different step order would differ
        here for a reason C3 does not care about.

        Hash *values* only compare across engines when both derived the sentinel the
        same way -- see `compute_none_hash`. Hit *rates* compare unconditionally,
        which is why both are recorded.

        **This is the one method that does not transfer to upstream.** The *keys*
        are the same type there -- both are `NewType("BlockHashWithGroupId", bytes)`
        -- but at the pin upstream replaced the plain dict with a `BlockHashToBlockMap`
        class that defines `__len__` and no `__iter__`, so iterating it raises. It is
        named here rather than caught, because a capture that silently recorded no
        hashes would produce a golden whose C3 section is empty and compares equal to
        anything.
        """
        cache = pool.cached_block_hash_to_block
        try:
            keys = list(cache)
        except TypeError as exc:
            raise NotImplementedError(
                f"cannot iterate {type(cache).__name__} to snapshot block hashes. "
                f"Upstream's BlockHashToBlockMap is not iterable at the pin; reach "
                f"into its private `_cache` to record C3's hash values, or compare "
                f"with compare_hash_values=False, which is the honest option until "
                f"someone does."
            ) from exc
        self.cached_hashes = sorted(bytes(key).hex() for key in keys)


@dataclass
class ConformanceRecord:
    """One workload's decisions, in a form two engines can be diffed in."""

    workload: str
    #: `pretending-vllm` or `vllm`. See the module docstring: this decides whether a
    #: passing comparison means `asserted` or `verified`.
    source: str
    upstream_version: str
    config: dict[str, Any]
    #: C1: one entry per engine step, in order.
    steps: list[dict[str, Any]] = field(default_factory=list)
    #: C2: block ids per allocation and per free, in order.
    block_allocations: list[list[int]] = field(default_factory=list)
    block_frees: list[list[int]] = field(default_factory=list)
    #: C3.
    prefix_cache: dict[str, Any] = field(default_factory=dict)
    block_hashes: list[str] = field(default_factory=list)
    #: C4.
    preemptions: dict[str, Any] = field(default_factory=dict)
    #: The generated token ids per request, which is what preemption equivalence and
    #: determinism are ultimately asserted on.
    outputs: dict[str, list[int]] = field(default_factory=dict)
    schema_version: int = CONFORMANCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workload": self.workload,
            "source": self.source,
            "upstream_version": self.upstream_version,
            "config": self.config,
            "steps": self.steps,
            "block_allocations": self.block_allocations,
            "block_frees": self.block_frees,
            "prefix_cache": self.prefix_cache,
            "block_hashes": self.block_hashes,
            "preemptions": self.preemptions,
            "outputs": self.outputs,
        }

    def write(self, path: str | Path) -> None:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> ConformanceRecord:
        payload = json.loads(Path(path).read_text())
        version = payload.get("schema_version")
        if version != CONFORMANCE_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: golden was written with conformance schema v{version}, but "
                f"this build reads v{CONFORMANCE_SCHEMA_VERSION}. Re-record it with "
                f"tools/capture_golden_trace.py rather than comparing across "
                f"schemas -- a field whose meaning changed would diff as a "
                f"behavioural difference."
            )
        return cls(
            workload=payload["workload"],
            source=payload["source"],
            upstream_version=payload["upstream_version"],
            config=payload["config"],
            steps=payload["steps"],
            block_allocations=payload["block_allocations"],
            block_frees=payload["block_frees"],
            prefix_cache=payload["prefix_cache"],
            block_hashes=payload["block_hashes"],
            preemptions=payload["preemptions"],
            outputs=payload["outputs"],
        )


#: Which conformance class each section belongs to, for readable failures. A diff
#: that says "C4: preemption victim differs" points at the requirement; one that says
#: "list index 7 differs" points at nothing.
SECTION_CLASSES = {
    "steps": "C1 (scheduler decision sequence)",
    "block_allocations": "C2 (KV block allocation order)",
    "block_frees": "C2 (KV block free order)",
    "prefix_cache": "C3 (prefix cache hit rate)",
    "block_hashes": "C3 (block hash values)",
    "preemptions": "C4 (preemption count and victims)",
    "outputs": "output tokens",
}


def compare(
    recorded: ConformanceRecord,
    golden: ConformanceRecord,
    *,
    compare_hash_values: bool = True,
) -> list[str]:
    """Differences between a fresh recording and a golden, most specific first.

    `compare_hash_values` exists for the one part of C3 that is legitimately
    incomparable across engines: a real vLLM run salts its none-hash from
    `os.urandom` unless `PYTHONHASHSEED` is set, so hash *values* recorded there
    cannot match ours even when every decision does. Hit rates and allocation order
    still can, and do. Turning this off against a `vllm`-sourced golden is honest;
    turning it off against our own is hiding a bug.
    """
    differences: list[str] = []

    if recorded.workload != golden.workload:
        return [
            f"comparing different workloads: recorded {recorded.workload!r} against "
            f"golden {golden.workload!r}"
        ]

    if recorded.upstream_version != golden.upstream_version:
        differences.append(
            f"pin mismatch: golden was recorded against upstream "
            f"{golden.upstream_version}, this build targets "
            f"{recorded.upstream_version}. Behavioural differences below may be "
            f"upstream's, not ours (see UPSTREAM.md, 'Bumping the pin')."
        )

    if recorded.config != golden.config:
        changed = sorted(
            key
            for key in set(recorded.config) | set(golden.config)
            if recorded.config.get(key) != golden.config.get(key)
        )
        differences.append(
            f"config differs on {changed}; the workloads are not comparable until "
            f"that is reconciled"
        )

    # C1 first, and step-by-step: a scheduler that diverges at step 3 produces a
    # different everything from step 4 on, and reporting the first divergence is more
    # useful than reporting all 200 of its consequences.
    if len(recorded.steps) != len(golden.steps):
        differences.append(
            f"C1 (total steps to drain): {len(recorded.steps)} steps, golden has "
            f"{len(golden.steps)}"
        )
    # Non-strict on purpose: a length difference is already reported above as a C1
    # step-count difference, and raising here would replace that specific message
    # with a ValueError.
    for index, (mine, theirs) in enumerate(
        zip(recorded.steps, golden.steps, strict=False)
    ):
        if mine != theirs:
            differences.append(
                f"C1 (scheduler decision sequence): step {index} differs\n"
                f"    recorded: {json.dumps(mine, sort_keys=True)}\n"
                f"    golden:   {json.dumps(theirs, sort_keys=True)}"
            )
            break

    for section in ("block_allocations", "block_frees"):
        mine = getattr(recorded, section)
        theirs = getattr(golden, section)
        if mine != theirs:
            differences.append(
                f"{SECTION_CLASSES[section]}: {_first_difference(mine, theirs)}"
            )

    if recorded.prefix_cache != golden.prefix_cache:
        differences.append(
            f"{SECTION_CLASSES['prefix_cache']}: recorded {recorded.prefix_cache}, "
            f"golden {golden.prefix_cache}"
        )

    if compare_hash_values and recorded.block_hashes != golden.block_hashes:
        differences.append(
            f"{SECTION_CLASSES['block_hashes']}: "
            f"{len(recorded.block_hashes)} resident hashes vs "
            f"{len(golden.block_hashes)}; "
            f"{_first_difference(recorded.block_hashes, golden.block_hashes)}"
        )

    if recorded.preemptions != golden.preemptions:
        differences.append(
            f"{SECTION_CLASSES['preemptions']}: recorded {recorded.preemptions}, "
            f"golden {golden.preemptions}"
        )

    if recorded.outputs != golden.outputs:
        differing = sorted(
            key
            for key in set(recorded.outputs) | set(golden.outputs)
            if recorded.outputs.get(key) != golden.outputs.get(key)
        )
        differences.append(f"output tokens differ for requests {differing}")

    return differences


# --- recording -------------------------------------------------------------
#
# Fields dropped from every step record before it reaches a golden. `t` is modeled
# time and `kv_usage` is a float derived from a pool size the memory model computes --
# both move when the cost model or the device card is recalibrated, neither is a
# scheduling decision. See the module docstring on why mixing them in would rot the
# suite.
_NON_DECISION_STEP_FIELDS = frozenset({"v", "seq", "type", "t", "kv_usage"})


def record_workload(
    workload: Workload,
    *,
    source: str = "pretending-vllm",
    trace_dir: str | Path | None = None,
) -> ConformanceRecord:
    """Run one workload and record what the engine decided.

    Drives the offline `LLM` class rather than the HTTP server: C1--C4 are engine
    behavior, and a server in front of them adds an event loop whose interleaving is
    not part of the contract.
    """
    import tempfile

    from pvllm.entrypoints.llm import LLM

    directory = Path(trace_dir) if trace_dir else Path(tempfile.mkdtemp())
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / f"{workload.name}.jsonl"

    engine = LLM("tiny-test", trace_path=str(trace_path), **workload.engine_kwargs())
    try:
        return _record(engine, workload, source, trace_path)
    finally:
        # In a `finally` because the trace file is opened by the engine: a workload
        # that raises would otherwise leave a `BufferedWriter` open until the garbage
        # collector noticed, and on Windows the next test to touch that path fails
        # for reasons that have nothing to do with it.
        engine.shutdown()


def _record(
    engine: Any, workload: Workload, source: str, trace_path: Path
) -> ConformanceRecord:
    from pvllm import UPSTREAM_VERSION
    from pvllm.sampling_params import SamplingParams
    from pvllm.tracing import read_trace

    core = engine.llm_engine.engine_core.engine_core
    scheduler = core.scheduler
    pool = scheduler.kv_cache_manager.block_pool

    recorder = BlockPoolRecorder()
    recorder.attach(pool)

    outputs = engine.generate(
        list(workload.prompts), SamplingParams(max_tokens=workload.max_tokens)
    )

    cache_stats = scheduler.kv_cache_manager.make_prefix_cache_stats()
    prefix_cache = {
        "queries": cache_stats.queries,
        "hits": cache_stats.hits,
        # Rounded: a hit rate is a ratio of two integers already recorded above, and
        # carrying its full float precision into a golden invites a diff caused by
        # nothing but repr.
        "hit_rate": round(cache_stats.hit_rate, 6),
        "evictions": cache_stats.evictions,
    }
    recorder.snapshot_hashes(pool)
    preemptions = {"total": scheduler.num_preemptions_total}

    # Shut down *before* reading the trace: the writer flushes every 64 records, so
    # reading it while still open loses the tail and the recording silently comes up
    # short. (The outer `finally` shuts down again; both paths are idempotent.)
    engine.shutdown()

    steps: list[dict[str, Any]] = []
    for record in read_trace(trace_path):
        if record.get("type") != "step":
            continue
        steps.append(
            {k: v for k, v in record.items() if k not in _NON_DECISION_STEP_FIELDS}
        )

    # Per-step preemption detail, so C4 pins *which* request was chosen and when,
    # not just how many times it happened.
    preemptions["by_step"] = {
        str(step["step"]): step["preemptions"]
        for step in steps
        if "preemptions" in step
    }

    return ConformanceRecord(
        workload=workload.name,
        source=source,
        upstream_version=UPSTREAM_VERSION,
        config=dict(sorted(workload.engine_kwargs().items())),
        steps=steps,
        block_allocations=recorder.allocations,
        block_frees=recorder.frees,
        prefix_cache=prefix_cache,
        block_hashes=recorder.cached_hashes,
        preemptions=preemptions,
        outputs={
            output.request_id: list(output.outputs[0].token_ids) for output in outputs
        },
    )


def _first_difference(mine: list[Any], theirs: list[Any]) -> str:
    # Walks the common prefix first, then falls through to the length report --
    # "differs at index 3" is more useful than "one is longer" when both are true.
    for index, (a, b) in enumerate(zip(mine, theirs, strict=False)):
        if a != b:
            return f"first difference at index {index}: {a!r} vs {b!r}"
    if len(mine) != len(theirs):
        return f"lengths differ: {len(mine)} vs {len(theirs)}"
    return "no element differs"
