# 00. Orientation

> **Files:** [`pvllm/__init__.py`](../../pvllm/__init__.py), [`README.md`](../../README.md), [`UPSTREAM.md`](../../UPSTREAM.md)
> **Prerequisites:** you can read Python. You do not need to know anything about vLLM,
> GPUs, or transformers yet.

## What this repository is

`pretending-vllm` is a **reimplementation of the vLLM V1 inference engine in which
every layer above the device is real, and only the device and the model are simulated.**

It is not a mock. A mock returns canned answers. This engine really schedules, really
allocates KV cache blocks, really preempts requests when memory runs short, really
serves the OpenAI HTTP API, and really exports Prometheus metrics. What it does not do
is multiply any matrices. Where a real engine would launch a CUDA kernel, this one asks
a cost model how long that kernel would have taken, advances a clock by that much, and
returns synthetic token ids.

```
REAL  entrypoints → processor → EngineCoreClient → EngineCore → Scheduler → KVCacheManager
                                       │
                                       ▼
REAL                        Executor → Worker
                                       │
============= SIMULATION BOUNDARY =====│=====================================
                                       ▼
FAKE                     SimModelRunner → SimDevice (memory ledger + cost model)
                                          SimModel  (token generator)
```

Exactly one function crosses that line, and it is the same one upstream uses:

```
SimModelRunner.execute_model(scheduler_output: SchedulerOutput) -> ModelRunnerOutput
```

Nothing above the boundary knows it is talking to a simulator. There is no
`if simulated:` branch anywhere in `v1/core`, `v1/engine`, or `entrypoints` — the
simulated worker is selected through vLLM's own platform plugin mechanism, exactly the
way an out-of-tree hardware backend would be. Chapter [03](03-simulation-boundary.md)
shows how that is enforced rather than merely intended.

## Why anyone would want this

Three problems, all of which cost GPU hours to work on otherwise.

**1. You are building a product on top of vLLM and you need to test it.**
Your product cares about streaming, cancellation, queueing, backpressure, capacity
limits, error codes at saturation, and what `/metrics` says. None of those need a real
model — but exercising them against real vLLM needs a GPU, several minutes of model
loading, and non-reproducible timing. Here, point your OpenAI client at
`localhost:8000` and it works in CI on a laptop, deterministically.

**2. You are capacity planning and the hardware is not available.**
"Does a 70B model at 128k context fit on eight 80 GB cards at `gpu_memory_utilization
0.92`, and how many concurrent requests does that leave?" is arithmetic on a model
architecture and a device spec. This project turns hardware into a JSON file so you can
answer it in a second, then sweep the knobs. Chapter [14](14-memory-model.md) walks the
arithmetic; chapter [28](28-benchmarking.md) sweeps it.

**3. You are trying to learn how vLLM works.**
vLLM's scheduler is one file of nearly three thousand lines and it is where the
interesting decisions live. Reading it is hard partly because you cannot easily run it:
the imports pull in torch, the code paths need a device, and the interesting behaviour
(preemption, prefix cache eviction order) only shows up under memory pressure you cannot
easily create. Here the same logic runs in-process under `pytest`, with a JSONL trace of
every decision and a timeline renderer. Module paths and class names mirror upstream
closely enough that diffing the two trees is a study exercise.

## The mental model to hold

Learn this and most of the rest follows.

**A serving engine's job is bookkeeping, not arithmetic.** The matrix multiplications
are the GPU's problem. Everything that makes an LLM server good or bad — how many
requests run at once, which one gets evicted when memory is tight, whether two requests
sharing a prompt prefix pay for it twice, how long a request waits before its first
token — is bookkeeping performed by the CPU, in Python, in code you can read.

That is why a simulator like this can be *useful* rather than a toy: it keeps 100% of
the bookkeeping and throws away 100% of the arithmetic. The parts it keeps are the parts
that decide whether your deployment works.

## What is exactly right, and what is not

The project publishes a **fidelity contract** (full version in
[README.md](../../README.md#fidelity-contract), explained in chapter
[29](29-conformance-and-fidelity.md)). The short version:

| Behaviour | Status |
|---|---|
| Scheduler decisions per step, and total steps to drain a workload | **exact** — a divergence is a bug |
| KV block allocation and free order | **exact** |
| Prefix cache hit rate and block hash values | **exact** |
| Preemption count and victim selection | **exact** |
| OpenAI HTTP schemas and error codes | **exact** |
| Prometheus metric names, types, labels, bucket edges | **exact** |
| Memory footprint, `num_gpu_blocks`, `max_concurrency` | **analytic** — exact given the cards, except a modeled activation term |
| Step latency, TTFT, inter-token latency, throughput | **modeled** — right shape, wrong values |
| Generated text | **not modeled at all** — synthetic tokens, no weights are ever read |
| Embedding vectors | **not modeled** — stable and distinct, but carrying no semantics |

The line to internalise: **latency here is a shape, not a measurement.** It will
reproduce the qualitative regimes correctly — prefill compute-bound and linear in
tokens, decode memory-bound and nearly flat, throughput saturating with batch size, TTFT
rising under queueing — and it will not tell you what your p99 will be. Chapter
[15](15-cost-model.md) is unusually blunt about this because it is the one place the
project can mislead you.

One more honesty note that applies to the whole series: the contract is currently
**asserted, not verified.** Golden traces captured from a real vLLM run at the pin do
not exist yet, so the conformance suite compares against previously recorded
*pretending-vllm* traces. That catches drift from our own past behaviour, not divergence
from upstream. The machinery to promote it ships in `tools/capture_golden_trace.py`.

## The pin

Structural fidelity is only meaningful against a specific upstream version.

| | |
|---|---|
| Upstream | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| Pinned version | **v0.27.1** (released 2026-08-11) |
| Vendored at | `vendor/vllm-0.27.1/` (gitignored, fetched by `tools/fetch_upstream.py`) |

Every module in this repository declares its upstream counterpart and a fidelity tier in
its docstring:

```python
"""The scheduler. The centerpiece.

Upstream: vllm/v1/core/sched/scheduler.py
Tier: A
"""
```

`tools/spec_sync.py` resolves each of those against the vendored tree and fails CI when
a counterpart has been renamed, moved, or deleted. Upstream drift becomes a failing
check instead of a discovery years later.

**This matters for you as a reader**: when a chapter says "upstream does X", it means
v0.27.1 does X. vLLM moves fast. In particular, if you have read older material about
vLLM, note that:

- The **V0 engine is gone**. Everything here is V1: one engine core, one scheduler, no
  `SequenceGroup`, no separate prefill/decode phases.
- The **V2 model runner has landed and is the default** for dense generate models
  (`vllm/v1/worker/gpu/`). The legacy 7,900-line `gpu_model_runner.py` is not what
  runs. This port mirrors V2.
- Prefix caching and chunked prefill are **on by default**, not opt-in.

## What this is not

- Not a drop-in import replacement for `vllm`. Compatibility is at the HTTP layer and
  the offline `LLM` class. `import vllm` code will not run unchanged.
- Not meaningful text. Outputs are drawn pseudowords. `pvllm` will happily tell you
  `'vugapibasurenanofulobudivupasegi'`.
- Not kernel-level or microsecond-accurate.
- Not real multi-machine distribution. Parallelism is modeled, not performed.

## Try it

The one-minute version of the whole project:

```bash
python -c "
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
llm = LLM(model='dense-0.6b', max_model_len=1024, trace_path='run.jsonl')
for out in llm.generate(['hello there', 'hello there, friend'], SamplingParams(max_tokens=8)):
    print(out.request_id, repr(out.outputs[0].text), out.outputs[0].finish_reason)
"
```

```
INFO ... Memory profile: capacity=80.00GiB, usable=73.60GiB, weights=1.11GiB,
         activation_peak=0.77GiB (modeled), non_torch=1.00GiB, kv_pool=70.72GiB,
         num_gpu_blocks=41382, max_concurrency=646.59x
0 'vugapibasurenanofulobudivupasegi' length
1 'nadipororebikilesekumopakoromole' length
```

Nonsense text, real block accounting. Then look at what the engine actually did:

```bash
pvllm trace view run.jsonl
```

```
pretending-vllm trace  (upstream 0.27.1, seed 0, clock virtual)
  model='dense-0.6b' card='dense-0.6b' device='datacenter-80gb' block_size=16 cost_model='constant'

  0  #=======  length
  1  #=======  length

  steps=8  tokens=46  preemptions=0  peak_kv=0.0%
  prefix cache: 0/32 tokens (0.0%)
  legend: # prefill  : small prefill  = decode  . waiting  ! preempted  ^ resumed
```

Two requests, eight engine steps: one prefill step each (`#`), then seven decode steps
(`=`). By the end of chapter [12](12-scheduler.md) you will be able to predict that
picture before running it, and by chapter [20](20-observability.md) you will be able to
find out why any run looks the way it does.

> **Note on imports.** The offline entrypoint lives at `pvllm.entrypoints.llm.LLM`.
> `pvllm/__init__.py` exports only `__version__` and `UPSTREAM_VERSION`, so
> `from pvllm import LLM` does *not* work today even though a couple of docstrings write
> it that way. Use the full path, as above.

## Check yourself

- What is the single function that crosses the simulation boundary, and why does having
  exactly one matter?
- Which of these can you trust from this engine: the number of engine steps, the KV
  block count, the p99 latency, the generated text?
- Why does the project pin an upstream version instead of tracking `main`?

## Next

[01. LLM inference fundamentals](01-llm-inference-fundamentals.md) — what the engine is
actually managing, from first principles.
