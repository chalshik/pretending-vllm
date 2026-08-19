# 31. Extending the port

> **Files:** [`UPSTREAM.md`](../../UPSTREAM.md), [`tools/`](../../tools), and whichever module you are about to touch
> **Prerequisites:** chapters [03](03-simulation-boundary.md), [29](29-conformance-and-fidelity.md), [30](30-testing-and-tooling.md).

You understand the engine. This chapter is how to change it without breaking the thing that makes it
worth having.

## The five rules

Everything below is an application of these.

**1. Fidelity tier first.** Decide the tier before you write code, and write the header. Tier A means
line-for-line with upstream — same method names, same order of operations, same branch structure. Tier D
means you may invent numbers. Getting this wrong is how a module that fabricates a duration ends up
somewhere the purity lint permits it.

**2. Refuse loudly, never no-op.** A dropped upstream code path raises `NotImplementedError` **naming the
upstream feature**. A flag that is accepted and ignored is worse than one that is refused: the user gets
numbers for a configuration they are not running.

**3. Keep the boundary.** No clock, no randomness, no `pvllm.sim` import above the line. If you need one
of those in the control plane, you need a *protocol* above the line and an implementation below it — the
way `Clock`, `TraceSink`, and `GammaSource` all work.

**4. Decisions are contract, durations are shape.** Anything that changes what the scheduler or the KV
manager *decides* touches C1–C4 and needs a conformance triage. Anything that changes a duration needs a
`modeled` label and touches nothing contractual.

**5. If it is load-bearing, add a mutation entry.** Write the test, then break the code and prove the test
notices.

## Adding a feature

The order that works, using multimodal (chapter [23](23-multimodal.md)) as the worked example:

**1. Read upstream first.**

```bash
python tools/fetch_upstream.py
grep -rn "encoder_cache" vendor/vllm-0.27.1/vllm/v1/core/
```

Find every place the feature touches. The shape of upstream's implementation is the specification — not
because it is optimal, but because *matching* it is the product.

**2. Decide what is real and what is modeled.** Split it explicitly. For multimodal: placeholders,
budgets, the encoder cache, and the prefix cache keys are **real**; the pixels and the encoder's FLOPs are
**modeled**. Write that sentence down before coding — it becomes the module docstring.

**3. Start with the types.** `MultiModalFeatureSpec` before the scheduler changes. Types crossing the
engine-core boundary must be msgspec-serializable, and their annotations must be importable at **runtime**
(chapter [07](07-requests-and-sampling.md)).

**4. Add the config, with validation.** In the right sub-config, with a `__post_init__` that refuses bad
values and derived defaults that mirror upstream's. Anything simulator-specific goes in `SimConfig` and
nowhere else.

**5. Thread it through, in upstream's order.** The order of operations inside `schedule()` and
`execute_model()` *is* the contract. If upstream checks the encoder budget before `allocate_slots`, so must
you — chapter [23](23-multimodal.md) has the bug that comes from getting that backwards.

**6. Model the cost as its own term.** Not folded into an existing one. Chapter
[15](15-cost-model.md)'s `encoder_seconds` explains why: a term added to the compute side flipped
`bound_by` on every step carrying an image.

**7. Surface it.** A metric if it has a rate or a hit ratio; a trace field if it is a decision; a
`/debug/*` field if you would want it live. "Both counted since M4 and reported by nothing until now, so
the features whose whole point is a hit rate had none on the surface a dashboard reads" — do not let that
be your feature.

**8. Test it four ways.** A unit test for the mechanism; a regression test for the bug you found in
review; a mutation entry for the guarantee; a conformance triage if you touched a decision.

**9. Run everything.**

```bash
pytest -q
python tools/spec_sync.py
python tools/mutate.py
python tools/capture_golden_trace.py --check
ruff check . && ruff format --check . && mypy pvllm
```

## Refusing well

The message is the deliverable. Compare:

```python
# bad — true, useless
raise NotImplementedError("not supported")

# good — names the feature, says why, says what to do instead
raise NotImplementedError(
    "KV cache event publishing (R12.5, --kv-events-config) is not modelled by "
    "pretending-vllm: no block store/remove events are emitted, so enabling it would "
    "report a stream that never arrives."
)
```

Three ingredients: **what** was asked for (with upstream's own flag name), **why** it is absent, and
**what the consequence would be** if it silently proceeded. Over HTTP, use `not_implemented()` so the
status is 501 and the envelope is vLLM's (chapter [19](19-openai-server.md)).

And the subtler case, worth internalising because it will come up: `--async-scheduling` is refused **not**
because it is hard, but because implementing it would produce a *misleading* answer. It exists to hide the
scheduler's CPU time behind the forward pass; this engine charges no CPU time; so comparing the flag on and
off would report that it buys nothing, which is the opposite of what real hardware says. **A feature whose
modeled answer would be actively wrong is better refused than approximated.**

## Bumping the pin

A deliberate, reviewed task — never a drive-by. From [UPSTREAM.md](../../UPSTREAM.md):

```bash
# 1. update UPSTREAM_VERSION in tools/fetch_upstream.py
python tools/fetch_upstream.py --force --write-manifest
python tools/spec_sync.py            # 2. triage every module whose counterpart moved or vanished
                                     # 3. diff the Tier A files and port behavioural changes
python tools/capture_golden_trace.py --check
                                     # 4. TRIAGE EACH FAILING WORKLOAD
python tools/capture_golden_trace.py --force
python tools/capture_golden_trace.py --metrics --force
                                     # 5. update UPSTREAM.md's delta table
```

Step 4 is the step people skip and the reason the check exists:

> A behavioural change is expected after a pin bump; the point of the check is that you **decide** which
> changes were the port and which were bugs, workload by workload.

Tier A is where fidelity is contractual, so that is where the diffing effort goes. A silent divergence
there breaks C1–C4 — and nothing else will tell you.

## Where the traps are

Ranked by how much time they have cost, judging by the docstrings that record them:

| Trap | Symptom | Chapter |
|---|---|---|
| Freeing blocks head-first instead of tail-first | hit rate collapses, every count balances | [09](09-kv-cache-blocks.md) |
| Appending unhashed blocks instead of prepending | same | [09](09-kv-cache-blocks.md) |
| Reordering `RequestStatus` | requests never finish, no error | [07](07-requests-and-sampling.md) |
| Not rolling back rejected drafts | negative `num_new_tokens`, infinite loop | [25](25-speculative-decoding.md) |
| Evicting windowed blocks at schedule time | cross-request KV corruption | [11](11-hybrid-kv-groups.md) |
| Two places deriving `num_gpu_blocks` differently | startup passes, then hangs forever | [14](14-memory-model.md) |
| A `_total` suffix on a counter | every dashboard panel empty | [20](20-observability.md) |
| Dividing cost terms by `pp_size` | reports a step `pp_size` times too fast | [15](15-cost-model.md), [24](24-parallelism.md) |
| Dividing MoE FLOPs by `ep_size` | EP looks free | [24](24-parallelism.md) |
| Reading a clock in the frontend | works in process, mixes timelines over IPC | [16](16-clock-and-determinism.md) |
| A `TYPE_CHECKING` import on a msgspec Struct | `NameError` only in multiprocess mode | [07](07-requests-and-sampling.md) |
| Registering a request queue before rejecting a duplicate id | one bad id tears down every endpoint | [17](17-engine-core-and-frontends.md) |

Notice the pattern: **almost every one of them balances.** Counts add up, tests pass, nothing raises — and
the engine reports something untrue. That is the failure mode this project is built to resist, which is
why the invariant assertions, the slot-mapping oracle, the conformance goldens, and the mutation catalogue
all exist.

## Promoting the fidelity contract

The single highest-value contribution available, and it needs a GPU for an afternoon:

1. install real vLLM at v0.27.1;
2. run the four conformance workloads against it, with the recorder attached to its `BlockPool` (which
   works unchanged — chapter [29](29-conformance-and-fidelity.md));
3. supply a hash snapshot, since upstream's map is not iterable at the pin;
4. replace the goldens; **the tests do not change**;
5. the contract becomes `verified` instead of `asserted`.

Second highest: `tools/calibrate_cost_model.py`, which does not exist yet. Fit `mfu`, `bw_eff`,
`link_eff`, and `launch_overhead` to observed latencies and every number in chapters
[15](15-cost-model.md), [24](24-parallelism.md), and [28](28-benchmarking.md) stops being shape and starts
being an estimate.

## Where to look things up

| Question | Source |
|---|---|
| What does `R6.7` mean? | [`pretending_vllm_requirements.md`](../../pretending_vllm_requirements.md) |
| What does `F7` mean? | [UPSTREAM.md](../../UPSTREAM.md)'s delta table |
| Which upstream file does this mirror? | the module's own docstring |
| Why is this code shaped like this? | the module's own docstring — it is the primary source |
| What is exact and what is modeled? | [README.md](../../README.md)'s fidelity contract |
| What guarantee does this line hold up? | `grep` it in [`tests/mutations.toml`](../../tests/mutations.toml) |

**The docstrings are the primary source for this whole series.** They explain not just what the code does
but what an earlier version got wrong and why the current shape is the way it is. Where a chapter here and
a docstring there disagree, the docstring is right and this is a bug.

## Check yourself

- You are adding a feature that needs randomness in the scheduler. What do you do?
- When is refusing a feature better than approximating it? Give the canonical example.
- Which step of a pin bump is the one that must not be skipped, and why?
- Name three traps whose symptom is "everything balances and the answer is wrong".
- What would make the fidelity contract `verified` rather than `asserted`?

## The end

You have read the whole engine. If you want to keep going:

- **read the real thing** — `python tools/fetch_upstream.py`, then diff the Tier A files. You now know
  what to look for;
- **break something on purpose** — pick a mutation entry, apply it by hand, and watch which test catches
  it;
- **answer a capacity question you actually have** — chapters [14](14-memory-model.md),
  [24](24-parallelism.md), and [28](28-benchmarking.md) are enough to do it today;
- **point your product at it** — `pvllm serve`, and find out what your client does when the engine
  saturates.

Back to the [index](README.md).
