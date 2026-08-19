# 29. Conformance and fidelity

> **Files:** [`pvllm/conformance.py`](../../pvllm/conformance.py), [`pvllm/conformance_workloads.py`](../../pvllm/conformance_workloads.py), [`tests/conformance/`](../../tests/conformance), [`tools/capture_golden_trace.py`](../../tools/capture_golden_trace.py)
> **Upstream:** none — this is pvllm's own machinery (Tier B)
> **Prerequisites:** chapters [12](12-scheduler.md), [20](20-observability.md).

The README claims C1–C4 are *exact*, and that a divergence is a bug by definition.

> A claim like that is worth nothing unless something checks it, and checking it means being able to
> record what an engine decided, in a form two engines can be compared in.

This chapter is that machinery.

## The seven classes

**Exact — a divergence is a bug by definition:**

| | |
|---|---|
| **C1** | Scheduler decision sequence per step, and total engine steps to drain a workload |
| **C2** | KV block allocation and free order |
| **C3** | Prefix cache hit rate and block hash values |
| **C4** | Preemption count and victim selection |
| **C5** | OpenAI HTTP request/response schema for implemented endpoints, **errors included** |
| **C6** | Prometheus metric names, types, labels, and histogram bucket edges |
| **C7** | Error codes and failure modes at capacity |

**Approximate, labelled `modeled` everywhere it surfaces:** step latency, TTFT, ITL, throughput.

**Analytic, exact given the cards:** memory footprint, derived `num_gpu_blocks`, `max_concurrency` —
with the one modeled activation term (chapter [14](14-memory-model.md)).

**Not modeled at all:** generated text quality; logprob *values* (schema and shape only); embedding
*vectors* (chapter [27](27-pooling-and-embeddings.md)).

## Asserted, not verified — and why that distinction is the honest one

> **Current state: `asserted`, not `verified`.** Golden traces captured from a real vLLM run at the
> pinned version do not exist yet. Until they do, the conformance suite runs in self-consistency mode:
> it compares against previously recorded pretending-vllm traces, which catches *drift* but not
> *divergence from upstream*.

So today the suite answers "did this engine's behaviour change?" — a genuinely useful question — and not
"does this engine match vLLM?". `tools/capture_golden_trace.py` ships so the contract can be promoted
when hardware time becomes available, and `source` in every record is what keeps the difference
visible:

```python
#: `pretending-vllm` or `vllm`. See the module docstring: this decides whether a
#: passing comparison means `asserted` or `verified`.
source: str
```

`compare` refuses to let that difference go unnoticed. Read the whole contract as design intent until a
`vllm`-sourced golden exists.

## Decisions, not durations

This is the design decision that makes the suite worth having:

```python
@dataclass
class ConformanceRecord:
    workload: str
    source: str  # pretending-vllm | vllm
    upstream_version: str
    config: dict[str, Any]  # completely pinned
    steps: list[dict]  # C1
    block_allocations: list[list[int]]  # C2
    block_frees: list[list[int]]  # C2
    prefix_cache: dict  # C3
    block_hashes: list[str]  # C3
    preemptions: dict  # C4
    outputs: dict[str, list[int]]
```

**No timestamps. No durations. Nothing the cost model touches.** The reasoning:

> C1–C4 are about what the scheduler and KV manager *chose*; latency is approximate by construction and
> carries a published error band. If both lived in one artifact, every cost-model recalibration would
> fail the conformance suite, goldens would get regenerated reflexively to make it green, and the signal
> that a real scheduler regression is supposed to produce would be gone by the third time.

Keeping them apart is what makes a failure here **mean** something.

## What a golden looks like

[`tests/conformance/goldens/shared-prefix.json`](../../tests/conformance/goldens/shared-prefix.json),
abbreviated:

```json
{
  "schema_version": 1,
  "workload": "shared-prefix",
  "source": "pretending-vllm",
  "upstream_version": "0.27.1",
  "config": {"block_size": 8, "device_card": "tiny-2gb", "enable_prefix_caching": true, ...},
  "steps": [{"new_reqs": ["0"], "cached_reqs": [], "num_common_prefix_blocks": [8], ...}, ...],
  "block_allocations": [[0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15], [16,17,18,19,20], ...],
  "block_frees": [[30,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0], ...],
  "block_hashes": ["001c0a2b93eae244b0a4046d15e41b72a7502ae1642fb588cc1d6c6f92f39c5600000000", ...],
  "prefix_cache": {"queries": 825, "hits": 576, "hit_rate": 0.698182, "evictions": 0},
  "preemptions": {"total": 0, "by_step": {}},
  "outputs": {"0": [660, 676, 681, 803, ...], "1": [550, 903, ...]}
}
```

Look at `block_frees`: `[30, 20, 19, 18, ..., 1, 0]` — **descending**. That is chapter
[09](09-kv-cache-blocks.md)'s tail-first rule, recorded. If someone changed `KVCacheManager.free` to
release head-first, every count would still balance, the tests would still pass, the hit rate would
quietly collapse — and *this* file would show it immediately.

## The four workloads

Each is "the smallest thing that makes one conformance class *fail loudly* when the corresponding logic
drifts":

| Workload | Pins |
|---|---|
| `mixed-lengths` | C1 — the decision sequence over prompts of differing sizes |
| `shared-prefix` | C3 — cache hits, hash values, and the shared-prefix block count |
| `preemption` | C4 — a starved block budget, so victims and counts are forced |
| `chunked-prefill` | C1 — a prompt long enough to be split across steps |

> A workload that happens to exercise a code path is not the same as one whose recording changes when
> that path changes — the second is what a regression suite needs.

They are deliberately tiny — `tiny-test` on `tiny-2gb`, single-digit requests — because the whole suite
is budgeted at 30 seconds, and "a conformance run that took minutes would get skipped locally and only
fail in CI, which is where regressions become archaeology."

Every workload **pins its config completely** — block size, budget, device card, seed — because a golden
recorded under a different config is not comparable, and `compare` refuses to pretend otherwise. Each
also carries a `pins` string explaining what it would catch, "so a diff arrives with the reason the
workload exists attached."

## Failures that name the requirement

```python
SECTION_CLASSES = {
    "steps": "C1 (scheduler decision sequence)",
    "block_allocations": "C2 (KV block allocation order)",
    "block_frees": "C2 (KV block free order)",
    "prefix_cache": "C3 (prefix cache hit rate)",
    "block_hashes": "C3 (block hash values)",
    "preemptions": "C4 (preemption count and victims)",
    "outputs": "output tokens",
}
```

> A diff that says "C4: preemption victim differs" points at the requirement; one that says "list index
> 7 differs" points at nothing.

Schema mismatches refuse rather than comparing across versions: "a field whose meaning changed would
diff as a behavioural difference."

## The recorder attaches to *upstream's* block pool

This is the detail that makes promotion possible at all:

```python
class _BlockPoolLike(Protocol):
    def get_new_blocks(self, num_blocks: int) -> list[Any]: ...
    def free_blocks(self, ordered_blocks: Any) -> None: ...
```

Structural, not an import. `attach()` wraps two bound methods and touches nothing else, so it works on
**vLLM's own `BlockPool` unchanged** — those two methods have the same names and signatures at the pin.

The part that does **not** transfer is `snapshot_hashes`: upstream wraps its hash map in a
`BlockHashToBlockMap` that is not iterable at the pin, so a `vllm`-sourced capture has to supply its own
hash snapshot. Everything else in a record (C1, C2, C4, and hit *rates*) crosses unchanged — which is
why `compare(..., compare_hash_values=False)` exists.

And recall chapter [10](10-prefix-caching.md)'s `none_hash` divergence: hash *values* are only
comparable to a real run that had `PYTHONHASHSEED` set to a matching value. Hit rates and allocation
order are comparable unconditionally.

## C5–C7 are schema checks, not recordings

[`tests/conformance/test_c5_to_c7.py`](../../tests/conformance/test_c5_to_c7.py) — 499 lines against the
live HTTP app:

- **C5** — every implemented endpoint's request and response shape, including the embeddings envelope
  ("the vectors are synthetic; the envelope around them is the contract");
- **C6** — the metric surface against its own golden, so **any** name, type, label, or bucket edge that
  moves fails; plus a test that counters are declared *without* a `_total` suffix, and a test that every
  latency family carries the `modeled` note in its HELP text;
- **C7** — the error handlers (a malformed body is 400 in vLLM's envelope, not FastAPI's 422), the
  message sanitiser, and — the interesting one — **behaviour at capacity**:

> C7 at capacity: queueing, not failing. A test double that 503s under load teaches a product the wrong
> lesson.

That last test is a good example of what a conformance class is *for*: the correct behaviour when the
engine is saturated is to make requests wait, not to reject them, and a product built against an engine
that rejected them would ship the wrong retry logic.

## Running it

```bash
pytest tests/conformance/ -q                       # compare fresh recordings to the goldens
python tools/capture_golden_trace.py --check       # assert the recorder still produces what is checked in
python tools/capture_golden_trace.py --metrics     # the same for the metric surface
```

CI runs all three. The `--check` step is **redundant with the test suite by design**: the suite compares
a fresh recording to the goldens; `--check` asserts the *tool* that writes them still produces what is
checked in, "so a broken recorder cannot quietly make future re-recordings wrong while the existing
goldens keep passing."

## When a golden legitimately changes

After a deliberate behaviour change, or a pin bump:

```bash
python tools/capture_golden_trace.py --check       # 1. see WHICH workloads moved, and where
                                                   # 2. triage each one — was it the change or a bug?
python tools/capture_golden_trace.py --force       # 3. only then re-record
python tools/capture_golden_trace.py --metrics --force
```

Step 2 is the whole point and the step people skip. From [UPSTREAM.md](../../UPSTREAM.md):

> A behavioural change is expected after a pin bump; the point of the check is that you **decide** which
> changes were the port and which were bugs, workload by workload.

The goldens embed the pin, so `test_goldens_declare_their_source` fails until they are re-recorded —
"the suite will not let a stale golden pass under a new pin."

## Promoting to `verified`

Run the same four workloads against real vLLM at the pin, replace the goldens, and **the tests do not
change**. That is the property the whole design is arranged around: the recorder attaches to upstream's
`BlockPool` unchanged, the record carries only decisions, and `source` records which engine produced it.

## Check yourself

- Why does a conformance record deliberately exclude durations?
- What does `source: "pretending-vllm"` in a golden tell you about what a passing test proves?
- In `block_frees`, why are the block ids descending?
- Why are the workloads tiny?
- Why does `tools/capture_golden_trace.py --check` run in CI when the test suite already compares
  goldens?
- Which part of a record cannot be captured from upstream unchanged, and why?
- What is the correct behaviour at capacity, and which class pins it?

## Next

[30. Testing and tooling](30-testing-and-tooling.md).
