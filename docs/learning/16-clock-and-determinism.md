# 16. Clock and determinism

> **Files:** [`pvllm/timebase.py`](../../pvllm/timebase.py), [`pvllm/sim/clock.py`](../../pvllm/sim/clock.py), [`pvllm/sim/rng.py`](../../pvllm/sim/rng.py), [`tests/unit/test_purity.py`](../../tests/unit/test_purity.py)
> **Upstream:** none — Tier **D** (upstream has real hardware and real time)
> **Prerequisites:** chapters [03](03-simulation-boundary.md), [15](15-cost-model.md).

Two properties make this project useful rather than merely plausible: **a 30-minute load test
runs in seconds**, and **the same seed reproduces the run exactly**. Both come from this
chapter, and both are the kind of property you cannot retrofit.

## Three clock modes, one timeline

```python
class Clock(ABC):
    def time(self) -> float:  # modeled timeline, as a Unix timestamp
        return self._epoch + self._elapsed

    def advance(self, duration: float) -> float:
        self._sleep(duration)  # ← the only thing that differs between modes
        self._elapsed += duration
        return self.time()
```

| Mode | `_sleep(d)` | Use for |
|---|---|---|
| `virtual` (default) | nothing | CI, sweeps, anything where waiting is waste |
| `real` | `time.sleep(d)` | exercising *your* client's timeouts, retries, streaming |
| `scaled` | `time.sleep(d / time_scale)` | a demo that should feel real but finish sooner |

**All three share one modeled timeline.** `time()` returns accumulated modeled time in every
mode; the modes differ only in whether, and for how long, the process waits.

```bash
python -c "
import time
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
for mode, scale in (('virtual', 1.0), ('scaled', 50.0), ('real', 1.0)):
    t0 = time.perf_counter()
    llm = LLM(model='dense-0.6b', max_model_len=512, clock_mode=mode,
              time_scale=scale, cost_model_profile='roofline')
    out = llm.generate(['hello there'], SamplingParams(max_tokens=10))[0]
    print(f'{mode:8s} wall={time.perf_counter()-t0:6.3f}s '
          f'modeled={llm.llm_engine.make_stats()[\"elapsed\"]:.4f}s '
          f'text={out.outputs[0].text[:12]!r}')
    llm.shutdown()
"
```

```
virtual  wall= 0.041s modeled=0.7988s text='vugapibasure'
scaled   wall= 0.031s modeled=0.7988s text='vugapibasure'
real     wall= 0.860s modeled=0.7988s text='vugapibasure'
```

**Identical modeled time, identical output, 20× difference in wall time.** That is the property
that lets a virtual-clock CI run and a real-clock demo be compared directly. (Most of the 0.7988
s here is the modeled weight load: 1.11 GiB over the card's declared 2 GB/s.)

One honest caveat, stated in the source: in `real` mode the **interpreter's own execution time is
not added to the timeline**. If a step models 100 ms but Python takes 130 ms, the engine reports
100 ms while a client measuring with its own wall clock sees 130. The modeled number is the
honest one to report — the extra 30 ms is simulator overhead a real vLLM would not have — but a
consumer comparing the two will see the gap. Above, `real` took 0.860 s of wall time for 0.7988
s of modeled time: that 60 ms is the simulator.

## The fixed epoch

```python
DEFAULT_EPOCH = 1767225600.0  # 2026-01-01T00:00:00Z
```

Fixed so a run is reproducible **down to its timestamps**; plausible so timestamps that reach an
API consumer do not look absurd. It is why every `created` field in chapter
[05](05-first-run.md) reads `1767225600`.

## Clock ownership

> The engine core owns the clock and is the only component that advances it.

Every consequence of that rule:

- `EngineCoreRequest.arrival_time` is optional, and the core stamps it on receipt — a divergence
  from upstream, which stamps it in the frontend with `time.time()`.
- The `Worker` **refuses to be constructed without a clock**: `ValueError: Worker requires the
  engine core's clock; it must not create one`.
- `SimDevice` takes the clock as a field, and `SimDevice.execute` is the *only* place time
  advances during inference. If a duration appeared in a trace, that call put it there.
- The scheduler collects the requests it admitted and the *core* dates them; the scheduler builds
  the step trace record and the core dates that too.
- Across a process boundary the frontend learns the time from the `timestamp` on every
  `EngineCoreOutputs` frame, plus an explicit round trip when it needs a fresh one — never from
  its own clock (chapter [18](18-multiprocess-engine.md)).

Why this could not be added later, in the source's words:

> In process, a frontend that read `time.time()` would appear to work; over a process boundary it
> would silently mix two timelines. Making every timestamp come back across this interface from
> the start is what made the multiprocess implementation a transport change rather than a
> redesign.

And it is enforced, not merely intended — `tests/unit/test_purity.py` fails the build if any
module outside `pvllm/sim/` touches `time.time`, `time.monotonic`, `time.perf_counter`,
`time.time_ns`, `datetime.now`, or `utcnow`.

## Async: where the time is actually spent

Under a real or scaled clock, *something has to wait*. If that something blocks the event loop,
an HTTP server stops streaming for exactly as long as the step it is streaming through — which
defeats the reason to run a real clock at all. So the whole path has an async twin:

```
AsyncLLM._run_output_handler
  → engine_core.get_output_async()
    → EngineCore.step_async()
      → executor.execute_model_async()
        → worker.execute_model_async()
          → SimModelRunner.execute_model_async()
            → SimDevice.execute_async()
              → clock.advance_async()  →  await asyncio.sleep(d)
```

Two design notes:

- `EngineCore.step` and `step_async` **share `_plan_step` and `_finish_step`**, so a clock mode
  can never change what the engine decided — only how long the process spent deciding it.
- `Executor.execute_model_async` is abstract rather than defaulting to the sync version,
  precisely so nobody can accidentally block the loop in a way that looks correct in every
  virtual-clock test.

`SimModelRunner.execute_model_async` has no upstream counterpart, and its docstring says why: on
real hardware the forward pass is launched asynchronously and awaited by CUDA, so there is no
interval during which Python holds the loop. Here the interval *is* the whole simulation.

## Determinism: the RNG design

```python
def _derive_entropy(seed: int, namespace: str, key: str) -> int:
    payload = f"{seed}\x00{namespace}\x00{key}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")
```

One global seed reproduces an entire run: arrival times, output lengths, sampled tokens,
cost-model jitter.

**The load-bearing property is that a request's RNG derives from `(seed, request_id)` rather
than being drawn from a shared stream.** A shared stream would make every request's tokens
depend on how requests happened to interleave — so adding one request to a workload would change
the output of all the others, and any test asserting on a specific request's output would be
coupled to the whole schedule.

With per-request derivation, request `abc` produces the same tokens whether it ran alone or
seventeenth in a batch of two hundred.

Three flavours of stream:

| Method | Derived from | For |
|---|---|---|
| `for_request(request_id)` | `(seed, "request", id)`, cached | output length, draft acceptance |
| `for_position(request_id, position, seed=None)` | `(seed, id, position)` | the token at one output position |
| `stream(name)` | `(seed, name)` | engine-level: `"jitter"`, `"arrival"`, `"workload"` |

`for_position` is **idempotent** — the same position always yields the same token, however many
times it is asked and whatever was asked before it. That is not a nicety: a preempted request
recomputes, and if recomputation produced different tokens, preemption would change the answer.
It is also what makes a client-supplied `seed` mean something.

BLAKE2b rather than Python's `hash()`, which is salted per process by `PYTHONHASHSEED` and would
make runs irreproducible across interpreter restarts.

## What determinism buys, and what breaks it

Determinism is what the conformance suite compares (chapter
[29](29-conformance-and-fidelity.md)) and what makes two runs of a sweep differ only in what was
configured. Two things trade it away, and both are opt-in for that reason:

**`PVLLM_ENABLE_V1_MULTIPROCESSING=1`.** The engine core steps whenever it has work, so whether
request seven arrived before step three depends on OS scheduling. Engine *decisions* stay
deterministic given an arrival order — but the arrival order is not. Upstream defaults this
**on**; this project defaults it **off**. Chapter [18](18-multiprocess-engine.md).

**`prefix_caching_hash_algo="builtin"`.** Python's `hash()` is salted per process, so block hash
values are not reproducible across runs unless `PYTHONHASHSEED` is set.

There is also the documented `none_hash` divergence from chapter
[10](10-prefix-caching.md): upstream seeds it from `os.urandom(32)`, which would make block hash
values differ every run; here it is derived from the seed.

## Try it

Reproducibility, and the one thing that is *not* the seed's job:

```python
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams


def run(seed):
    llm = LLM(model="dense-0.6b", max_model_len=512, seed=seed)
    out = llm.generate(["hello"], SamplingParams(max_tokens=8))[0]
    llm.shutdown()
    return out.outputs[0].text


print(run(0) == run(0))  # True  — same seed, same run
print(run(0) == run(1))  # False — different seed, different draw
```

And a per-request seed, which survives being run alongside anything else:

```python
llm = LLM(model="dense-0.6b", max_model_len=512)
a = llm.generate(["x"], SamplingParams(max_tokens=6, seed=99))[0].outputs[0].text
batched = llm.generate(["x", "y", "z"], SamplingParams(max_tokens=6, seed=99))
print(a == batched[0].outputs[0].text)  # True
```

## Check yourself

- Which mode sleeps, and which numbers change between modes?
- Why is `arrival_time` stamped by the engine core rather than the frontend?
- Why does the `Worker` refuse to construct without a clock?
- Why is a request's RNG derived from its id rather than drawn from a shared stream?
- Why must `for_position` be idempotent? What behaviour would break otherwise?
- Name the two opt-in settings that give up byte-identical determinism.

## Next

[17. Engine core and frontends](17-engine-core-and-frontends.md) — the loop, and the two ways to
drive it.
