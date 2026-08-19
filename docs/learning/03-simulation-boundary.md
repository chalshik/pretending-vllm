# 03. The simulation boundary

> **Files:** [`pvllm/platforms/`](../../pvllm/platforms), [`pvllm/sim/`](../../pvllm/sim), [`pvllm/timebase.py`](../../pvllm/timebase.py), [`pvllm/tracing.py`](../../pvllm/tracing.py), [`tests/unit/test_purity.py`](../../tests/unit/test_purity.py)
> **Upstream:** `vllm/platforms/interface.py` (Tier B), `vllm/platforms/cuda.py` (counterpart of `sim.py`, Tier D)
> **Prerequisites:** chapter [02](02-vllm-v1-architecture.md).

A test double is only trustworthy if you can point at the line between the real part and
the fake part. This chapter is that line: where it runs, how the simulated device gets
selected without anyone knowing, and the lint that fails the build when the line moves.

## The rule

> Nothing outside `pvllm/sim/` and `pvllm/platforms/sim.py` may invent a number, read a
> clock, or draw randomness.

That is one sentence with three clauses, and each clause has a reason.

**Invent a number.** If a duration, a memory footprint, or a token could be fabricated
anywhere in the engine, "which numbers can I trust?" would have no answer. Confining
fabrication to one subtree makes the answer mechanical: anything from `pvllm/sim/` is
modeled, everything else is computed.

**Read a clock.** The engine core owns the clock. A second component reading
`time.time()` would produce timestamps from a different timeline the moment the core ran
in another process — and every latency metric would become the sum of two clocks. This is
the clause that cannot be retrofitted: once a dozen call sites read the clock,
determinism is gone and getting it back means auditing all of them.

**Draw randomness.** One seed must reproduce an entire run: arrival times, output
lengths, sampled tokens, cost-model jitter. That only holds if `pvllm/sim/rng.py` is the
sole source.

## How the fake device gets selected

Here is the part that makes the whole design work, and it is not a trick — it is vLLM's
own extension mechanism.

vLLM resolves a **platform** at startup. The platform is asked to fill in details of the
resolved config, including which worker class to instantiate. `CudaPlatform` sets
`parallel_config.worker_cls` to `"vllm.v1.worker.gpu_worker.Worker"`. This port's
`SimPlatform` does the same thing with a different string
([`pvllm/platforms/sim.py`](../../pvllm/platforms/sim.py)):

```python
@classmethod
def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    if parallel_config.worker_cls == "auto":
        parallel_config.worker_cls = "pvllm.v1.worker.sim_worker.Worker"
```

That is the entire hinge. The executor loads a worker class by name; the name happens to
point at a simulated worker; nothing above the executor is aware. Upstream's
`PlatformEnum.OOT` exists precisely for out-of-tree backends, and out-of-tree plugins
registered under the entry-point group take precedence over builtins — so this port is
using the same door a real hardware vendor would.

The platform also answers the questions upstream answers by probing hardware:

```python
SimPlatform.get_device_name()          # from the device card's JSON
SimPlatform.get_device_total_memory()  # from the device card's JSON
SimPlatform.get_device_count()         # from the device card's JSON
```

**Hardware becomes a JSON file.** That is what makes "does 70B fit on eight 80 GB cards"
answerable without owning eight cards.

Two more things cross through the platform rather than being imported directly, for the
same reason:

```python
current_platform.build_clock(mode, time_scale=...)   # so EngineCore never imports sim
current_platform.build_trace_sink(path, ...)
current_platform.build_kv_connector(config, role)
```

`EngineCore` needs a clock, but a clock is a simulated thing. It asks the platform. The
`Clock` type it holds is the abstract one from
[`pvllm/timebase.py`](../../pvllm/timebase.py), which sits *above* the boundary; the
concrete `VirtualClock` lives below it. Same pattern for the trace sink, whose interface
is a `Protocol` in [`pvllm/tracing.py`](../../pvllm/tracing.py) satisfied structurally by
`pvllm.sim.trace.TraceWriter` — neither module imports the other.

## What is below the line

Everything in [`pvllm/sim/`](../../pvllm/sim). Nine files that matter:

| File | What it fabricates | Chapter |
|---|---|---|
| `clock.py` | time itself — virtual, real, or scaled | [16](16-clock-and-determinism.md) |
| `rng.py` | seeded, per-request random streams | [16](16-clock-and-determinism.md) |
| `device.py` | one accelerator: a memory ledger and a cost model | [14](14-memory-model.md) |
| `memory.py` | the analytic sizing that derives `num_gpu_blocks` | [14](14-memory-model.md) |
| `cost_model.py` | how long a step takes | [15](15-cost-model.md) |
| `model.py` | token ids, output lengths, draft acceptance, embeddings | [07](07-requests-and-sampling.md) |
| `model_db.py` / `hardware_db.py` | model cards and device cards (the JSON) | [06](06-configuration.md) |
| `weights.py` | "loading" weights, and the startup timeline | [14](14-memory-model.md) |
| `grammar.py` / `structured_output.py` | strings that satisfy a schema | [21](21-structured-output.md) |
| `kv_store.py` | an external KV store with a bandwidth | [26](26-kv-disaggregation.md) |
| `trace.py` | the JSONL writer | [20](20-observability.md) |

And one file just above the line that is the last real thing:
[`pvllm/v1/worker/sim_worker.py`](../../pvllm/v1/worker/sim_worker.py). It mirrors
`GPUWorker`'s lifecycle exactly — `init_device`, `load_model`,
`determine_available_memory`, `initialize_cache`, `compile_or_warm_up_model`,
`execute_model` — and it is where the `SimDevice` and `SimModel` are constructed.

## How the line is defended

Intentions rot. [`tests/unit/test_purity.py`](../../tests/unit/test_purity.py) turns all
three clauses into a failing build. It walks the **AST** of every file in `pvllm/`, not
the text, so that the docstrings in this repository — which discuss `time.time`
constantly — do not trip it.

What it checks:

```python
FORBIDDEN_TIME_ATTRS = {"time", "monotonic", "perf_counter", "process_time",
                        "time_ns", "now", "utcnow"}
FORBIDDEN_MODULES    = {"torch", "transformers", "cupy", "triton"}
RANDOM_MODULES       = {"random", "secrets"}
BOUNDARY_ENFORCED_SUBTREES = ("v1/core", "v1/engine", "entrypoints")
```

- No wall-clock read outside `pvllm/sim/`. (`now`/`utcnow` are in that set because
  `datetime.datetime.now()` is a wall clock too — they were missing once, and a
  `strftime_now` added to a chat template environment walked straight past the lint.)
- No `random`, no `numpy.random` outside `pvllm/sim/`. This is why
  [`benchmarks/lib/arrivals.py`](../../pvllm/benchmarks/lib/arrivals.py) takes its gamma
  generator as a structural `Protocol` instead of importing numpy's — the type annotation
  alone would fail the lint.
- No import of the `sim` package from `v1/core`, `v1/engine`, or `entrypoints`, and no
  `if simulated:` branching there.
- No `torch`, CUDA, or `transformers` import at module level anywhere. The base install
  really does not have them.
- Every module declares an upstream counterpart and a tier.

There is **exactly one deliberate exception**, and it is labelled as such:
[`entrypoints/serve/dev/introspect.py`](../../pvllm/entrypoints/serve/dev/introspect.py)
reaches through the worker into the simulator to report the cost-model breakdown for
`/debug/cost_model`. It has to — a cost-model breakdown *is* simulator state. The
exception is safe because the introspector decides nothing, so no engine behaviour can
depend on what it reports.

## Fidelity tiers

Every module declares one. This is the vocabulary the rest of the series uses, so it is
worth learning here.

| Tier | Meaning | The rule |
|---|---|---|
| **A** | Line-for-line | Same method names, same order of operations, same branch structure. Binds the C1–C4 contract. A behavioural divergence is a bug by definition. |
| **B** | Signature-faithful, body-thinned | Same public API and observable behaviour. Internals may drop unsupported paths. |
| **C** | Shape-only | Field names, types, and validation *intent* match. Implementation is ours. |
| **D** | Invented | No upstream counterpart. The only tier allowed randomness, wall-clock, or invented numbers. |

Where the tiers land, roughly:

- **Tier A**: the scheduler, the block pool, `kv_cache_utils`, the KV cache manager and
  coordinator, `Request`, the detokenizer, `kv_cache_interface`. In other words: exactly
  the code the fidelity contract calls exact.
- **Tier B**: the engine core, the executor, the frontends, the entrypoints, the metrics,
  the benchmarks.
- **Tier C**: the config dataclasses, the protocol models, `SamplingParams`.
- **Tier D**: everything in `pvllm/sim/`, plus `platforms/sim.py`, the mock tokenizer, and
  the simulated attention backend.

A module with no upstream counterpart declares one of two things:

```python
Upstream: (none -- simulator)        # Tier D
Upstream: (none -- pvllm addition)   # any tier: an interface above the boundary
```

The second form exists for pvllm-only *interfaces* that sit above the line — the trace
sink protocol, the timeline viewer, the conformance recorder. Calling those Tier D just
to satisfy `spec_sync` would wrongly mark them as places allowed to invent numbers.

## Unsupported-path discipline

One rule that shows up everywhere and is worth internalising early:

> A dropped upstream code path raises `NotImplementedError` naming the upstream feature.
> It never silently no-ops.

Examples you will meet:

```python
# pvllm/v1/executor/abstract.py
raise NotImplementedError(
    f"distributed_executor_backend={backend!r} is not implemented. ...")

# pvllm/v1/core/block_pool.py
raise NotImplementedError(
    "KV cache event publishing (R12.5, --kv-events-config) is not modelled ... "
    "so enabling it would report a stream that never arrives.")
```

For a test double, failing loudly beats behaving subtly wrongly. A flag that is accepted
and ignored is worse than one that is refused: the user gets numbers for a configuration
they are not running. `--async-scheduling` is refused for a subtler version of the same
reason — it exists to hide the scheduler's CPU time behind the forward pass, and this
engine charges no CPU time, so implementing it would report that it buys nothing, which
is the opposite of what real hardware says.

## Try it

Watch the boundary hold:

```bash
python -c "
import pvllm.v1.core.sched.scheduler, pvllm.v1.engine.core, sys
print('torch imported:', 'torch' in sys.modules)
print('transformers imported:', 'transformers' in sys.modules)
"
```

```
torch imported: False
transformers imported: False
```

And watch it be enforced:

```bash
pytest tests/unit/test_purity.py -q
```

Then try to break it — add `import time` and a `time.time()` call to
`pvllm/v1/core/sched/scheduler.py` and rerun. The failure names the file and the line.

## Check yourself

- What single assignment causes a simulated worker to be used instead of a real one?
- Why does `EngineCore` ask the platform for a clock instead of importing `VirtualClock`?
- Why is the trace sink a `Protocol` in `pvllm/tracing.py` rather than a class in
  `pvllm/sim/`?
- Which tier is the scheduler, and what does that tier promise?
- Why is a silently ignored flag worse than a refused one?

## Next

[04. Repository tour](04-repository-tour.md) — every directory and file, and why it is
there.
