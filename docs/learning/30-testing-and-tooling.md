# 30. Testing and tooling

> **Files:** [`tests/`](../../tests), [`tests/mutations.toml`](../../tests/mutations.toml), [`tools/`](../../tools), [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
> **Prerequisites:** chapters [03](03-simulation-boundary.md), [29](29-conformance-and-fidelity.md).

A test double's own tests are load-bearing in an unusual way: **if this engine is wrong, everything
tested against it is wrong and nobody finds out.** So the suite is doing more than checking that the
code works — it is defending a set of claims.

## The layout

| Directory | What it defends |
|---|---|
| `tests/unit/` | config resolution, the platform, the **purity lint**, the mutation tool itself |
| `tests/v1/` | the engine: scheduler, block pool, prefix cache, worker, LoRA, spec decode, hybrid groups, data parallelism, KV connector, plus per-milestone review-regression files |
| `tests/sim/` | the simulator: clock, cost model, memory, RNG, parallelism arithmetic, MLA, state space, trace |
| `tests/entrypoints/` | the HTTP surface, the debug endpoints, the offline `LLM`, the CLI client |
| `tests/conformance/` | C1–C4 against goldens, C5–C7 as schema checks (chapter [29](29-conformance-and-fidelity.md)) |
| `tests/property/` | hypothesis properties over the scheduler |
| `tests/benchmarks/` | the benchmark harness and the arrival process |

```bash
pytest -q
```

```
1083 passed, 12 skipped in 61.68s
```

(The 12 skips are empty-package markers in the purity lint. The requirements draft budgets the suite at
30 seconds; on the machine these docs were written on it takes about a minute, so treat 30 s as the
target rather than an observed fact.)

Two settings in `tests/conftest.py` that change what the suite *means*:

```python
os.environ.setdefault("PVLLM_DEBUG_INVARIANTS", "1")   # block accounting, slot-mapping validation
os.environ.setdefault("PVLLM_LOGGING_COLOR", "0")
```

The invariant assertions run for the **whole** suite. They are the cheapest place to catch a KV manager
bug, "and they only pay off if they are always on in tests." Plus an autouse fixture that resets the
process-global device card between tests, so results cannot depend on execution order.

And in `pyproject.toml`:

```toml
filterwarnings = ["error"]
```

A warning is a failure. That is how a deprecation gets fixed on the commit that introduces it.

## The purity lint

[`tests/unit/test_purity.py`](../../tests/unit/test_purity.py) — 429 lines that turn the simulation
boundary into a failing build. Covered in chapter [03](03-simulation-boundary.md); the summary:

1. no wall-clock read outside `pvllm/sim/`;
2. no randomness outside `pvllm/sim/`;
3. no simulator awareness in `v1/core`, `v1/engine`, `entrypoints`;
4. no `torch` / `transformers` / `cupy` / `triton` at import time anywhere;
5. every module declares an upstream counterpart and a tier.

AST-based, not grep-based, so the docstrings in this repository — which discuss `time.time` constantly —
do not trip it. And it is itself covered by a mutation entry (`purity-lint-sees-dotted-time-access`),
because a lint that cannot see a violation is worse than no lint.

## The mutation catalogue

This is the most unusual thing in the repository and the most worth stealing.

> A green suite says the tests pass. It does not say they would fail if the code were wrong.

[`tests/mutations.toml`](../../tests/mutations.toml) holds **30 entries**. Each names a guarantee, the
minimal edit that breaks it, and the test that should notice. `tools/mutate.py` applies the edit, runs
that one test, and **expects a FAILURE**. A test that stays green is reported: it is not guarding what
its name claims.

```toml
[[mutation]]
name = '''responses-stream-sends-no-done-sentinel'''
why  = '''The Responses stream must not send `[DONE]`; a client that waits for one hangs.'''
file = '''...'''
old  = '''...'''
new  = '''...'''
test = '''tests/entrypoints/test_responses_api.py::test_the_stream_sends_no_done_sentinel'''
```

```bash
python tools/mutate.py                 # every entry
python tools/mutate.py -k responses    # entries whose name matches
python tools/mutate.py --list          # names and reasons, run nothing
```

Why it exists, from the tool's own docstring:

> Mutation testing has caught a non-discriminating test nearly every time it has been run on this
> project — including three the author wrote in the two commits immediately before this file. It was run
> by hand, from memory, only on freshly written tests. Written down, it becomes something CI enforces
> instead of something someone remembers.

The rules for an entry are worth reading before you add one:

- `old` must appear **exactly once** in the file — the tool fails loudly otherwise, "which is how the
  catalogue notices that the code moved underneath it";
- `new` must be a **realistic mistake** — a flipped comparison, a dropped `min`, a removed filter — not
  a syntax error. "Breaking the parser proves nothing."
- `test` should be a single node id, so a failure is unambiguous and the run is fast;
- `why` is the guarantee in one line. "If you cannot state it, the entry is not pinning anything worth
  pinning."

Safety: edits are applied to the working tree and restored in a `finally`, and **the tool refuses to
start when a file it would edit has uncommitted changes**, so recovery can never cost you work.

A sample of what the 30 entries defend — read this list as a summary of the project's sharp edges:

```
preemption-requeues-at-the-front            num-gpu-blocks-uses-the-real-block-cost
kv-cache-events-are-refused                 embeddings-bill-every-document
purity-lint-sees-dotted-time-access         async-stop-string-aborts-in-the-core
engine-rejects-a-duplicate-request-id       streamed-content-carries-its-choice-index
responses-store-is-off-by-default           param-drops-pydantic-union-markers
sanitize-strips-memory-addresses            mid-stream-failure-emits-an-error-frame
```

Every one of those is a bug that shipped, or nearly did, in a form the tests did not catch.

## Property tests

[`tests/property/test_scheduler_properties.py`](../../tests/property/test_scheduler_properties.py) —
hypothesis over the scheduler. Properties, not examples: the token budget is never exceeded, the running
count never exceeds `max_num_seqs`, blocks are conserved, a workload always drains. Hypothesis generates
the workloads; a `.hypothesis/` database keeps failing examples so a shrunk counterexample is replayed
on the next run.

This is the right tool for the scheduler specifically, because its state space is combinatorial —
prompt lengths × arrival order × budget × pool size — and hand-written examples cover a vanishing
fraction of it.

## The review-regression files

`tests/v1/test_m4_review_regressions.py`, `test_m5_review_regressions.py`, `test_m7_review_regressions.py`
and friends — around 2,500 lines. One test per bug found in review, named for what it defends.

Worth understanding as a pattern: the repository's docstrings are full of "an earlier version did X, and
here is what went wrong". These files are the executable half of that. Every one of those narratives has
a test, so the same mistake cannot come back quietly.

## `tools/spec_sync.py` — the anti-rot check

```bash
python tools/spec_sync.py
```

```
upstream:  vendor/vllm-0.27.1
modules:   167  tier A: 16  tier B: 89  tier C: 33  tier D: 17

all declared upstream counterparts exist at the pin
```

Resolves every module's declared `Upstream:` header against the vendored tree and fails when a
counterpart has been renamed, moved, or deleted.

> This turns that decay into a failing check. ... The requirements draft was written against an older
> vLLM and had eight stale assumptions by the time the tree was actually checked.

It also reports **coverage** — which upstream modules in the mirrored subset have no pvllm counterpart —
as a progress tracker, not an error. Stdlib-only, so it runs in CI before the package is installed.

The tier census is a useful map in itself: 16 Tier A modules are exactly the code the fidelity contract
calls exact.

## `tools/fetch_upstream.py`

```bash
python tools/fetch_upstream.py               # download and extract the pinned tag (~36 MB)
python tools/fetch_upstream.py --check       # verify byte-identical to vendor/MANIFEST.sha256
```

Stdlib-only, so it runs before `pip install -e .`. Kernels and test data are skipped — "a port with no
device code has no use for them." The tree is **not** committed; `vendor/MANIFEST.sha256` (2,736 files)
is, so anyone can verify their vendored copy matches the one this port was written against.

## CI, in order

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml), on `{ubuntu, macos, windows} × {3.11, 3.13}`
— "no device to detect, so the suite must pass everywhere":

```
1. fetch the vendored upstream tree (cached by tag — the pin is immutable, so a hit is always valid)
2. verify it against the manifest
3. spec_sync            — upstream drift
4. ruff check           — lint
5. ruff format --check
6. pytest -q --durations=10
7. python tools/mutate.py            — the mutation catalogue
8. capture_golden_trace --check / --metrics
9. mypy pvllm  (with the realtok extra)
```

The **ordering is deliberate** and the file says why:

- the mutation catalogue runs **after** the suite, "because a genuine failure above would make every
  entry 'caught' for the wrong reason";
- `mypy` runs **last** because it needs the optional `realtok` extra to resolve
  `pvllm/tokenizers/hf.py`, and `uv run --extra` installs into the shared environment — so running it
  earlier would leave the extra installed and the tests would stop exercising the default install,
  "the configuration a user actually gets."

`mypy` is `strict = true` with `warn_unreachable = true`. `ruff` selects `E, F, I, UP, B, SIM, C4, RUF`.

## Writing a test in this codebase

The conventions that make the suite what it is:

1. **Name the guarantee, not the function.** `test_preemption_requeues_at_the_front`, not
   `test_preempt_request`.
2. **Assert on decisions, not durations** — unless you are testing the cost model. Durations are modeled.
3. **Use `tiny-test` on `tiny-2gb`** for anything that does not need a realistic model. It keeps the
   suite fast.
4. **Force the interesting state.** `num_gpu_blocks_override` to cause preemption; `max_num_seqs=1` to
   cause queueing; `max_num_batched_tokens` small to force chunking.
5. **Add a mutation entry** for anything load-bearing, and check it fails before you trust it.
6. **Prefer a property** when the state space is combinatorial.

## Check yourself

- Why does the whole suite run with `PVLLM_DEBUG_INVARIANTS=1`?
- What question does the mutation catalogue answer that a green suite does not?
- Why must a mutation's `new` be a realistic mistake rather than a syntax error?
- Why does the purity lint walk the AST rather than grep?
- Why does `mypy` run last in CI, and what would break if it ran first?
- Why is `vendor/` gitignored while `vendor/MANIFEST.sha256` is committed?

## Next

[31. Extending the port](31-extending-the-port.md) — the last chapter.
