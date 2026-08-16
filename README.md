# pretending-vllm

A structurally faithful reimplementation of the vLLM V1 engine in which every layer above the device
is real logic, and only the device and the model are simulated.

Point a product that speaks the OpenAI API at it and exercise streaming, cancellation, queueing,
backpressure, capacity limits, and metrics scraping — deterministically, in CI, on a laptop, with no
GPU and no model weights.

**Upstream pin: [vLLM v0.27.1](https://github.com/vllm-project/vllm/tree/v0.27.1)** (2026-08-11).
See [UPSTREAM.md](UPSTREAM.md).

> **Status: M1–M4 complete.** The OpenAI server, the offline `LLM` class,
> `AsyncLLM`, `/metrics`, prefix caching, chunked prefill, preemption, the debug surface (JSONL
> trace, timeline viewer, `/debug/*` endpoints), the conformance suite, `pvllm bench`, the
> multiprocess engine core, real/scaled clocks, structured output, LoRA, tensor and pipeline
> parallelism, speculative decoding, sliding-window attention, multimodal, and KV
> disaggregation all work. Mixed full/windowed models, data and expert
> parallelism, real KV transports, and a KV connector paired with a sliding window are not
> implemented and refuse by name.

```bash
pvllm serve --model meta-llama/Llama-3.1-8B-Instruct --device-card datacenter-80gb
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
```

## What is real, and what is not

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

Exactly one interface crosses the boundary, the same one upstream uses:

```python
SimModelRunner.execute_model(scheduler_output: SchedulerOutput) -> ModelRunnerOutput
```

No code above the boundary knows it is talking to a simulator. There is no `if simulated:` anywhere
in `v1/core`, `v1/engine`, or `entrypoints` — selection happens through the platform abstraction,
exactly as an out-of-tree hardware backend is selected upstream.

## Seeing what the engine is doing

Three ways in, all of them structured — there is no verbose stdout narration to grep.

**A JSONL trace** of every step and every request transition:

```bash
pvllm serve --model dense-0.6b --trace-path run.jsonl
```

**A timeline** rendered from that trace, as text or SVG:

```bash
pvllm trace view run.jsonl
```

```
pretending-vllm trace  (upstream 0.27.1, seed 0, clock virtual)
  model='dense-0.6b' device='workstation-24gb' block_size=16 cost_model='constant'

  0  #====            length
  1  :====            length
  2  .....:====       length

  steps=15  tokens=46  preemptions=0  peak_kv=28.6%
  prefix cache: 64/90 tokens (71.1%)
  legend: # prefill  : small prefill  = decode  . waiting  ! preempted  ^ resumed
```

**Live HTTP introspection**, for asking questions while a product is driving the engine:

```bash
pvllm serve --model dense-0.6b --enable-debug-endpoints
```

| endpoint | answers |
|---|---|
| `GET /debug/scheduler` | what is running, what is waiting, and in what order |
| `GET /debug/requests` | every tracked request, counted by state |
| `GET /debug/requests/{id}` | one request's state machine and block table |
| `GET /debug/blocks` | the block pool, and which requests hold which blocks |
| `GET /debug/prefix_cache` | hit rate overall, and per live request |
| `GET /debug/cost_model` | the term-by-term breakdown of recent steps |
| `GET /debug/config` | the fully resolved config, including what was *derived* |

All read-only, all off by default — they expose prompt token ids, and the gate mirrors upstream's
`VLLM_SERVER_DEV_MODE`. `/debug/cost_model` reports modeled durations; see the fidelity contract.

## Making time real

By default the clock is **virtual**: the engine models each step's duration without
spending it, so a workload representing minutes of serving runs in seconds. That is what makes
sweeps and CI runs cheap.

When you want your own client's timeouts, retries, and streaming behavior exercised against
something that actually takes time:

```bash
pvllm serve --model meta-llama/Llama-3.1-8B-Instruct --clock-mode real
```

`--clock-mode scaled --time-scale 10` replays the same timeline ten times faster. All three modes
report **identical** numbers — they differ only in whether the process waits. So a virtual-clock CI
run and a real-clock demo can be compared directly.

Real mode is faithful all the way through startup — but only with the roofline cost model, which is
not the default:

```bash
pvllm serve --model meta-llama/Llama-3.1-8B-Instruct --clock-mode real --cost-model-profile roofline
```

That takes about eight seconds to become ready, because that is what the cost model says loading an
8B model's weights over `datacenter-80gb`'s memory bandwidth costs. Under the default `constant`
profile weight loading is free and the server is ready in about a tenth of a second, which will not
exercise a client's readiness timeout at all.

## Running the engine in its own process

```bash
PVLLM_ENABLE_V1_MULTIPROCESSING=1 pvllm serve --model dense-8b
```

Mirrors upstream's `VLLM_ENABLE_V1_MULTIPROCESSING`: the engine core runs in a separate process and
the frontend talks to it over ZeroMQ with msgspec-encoded frames. Serialization and backpressure
become real, so a product that sends something the wire format cannot carry finds out here.

**It defaults off, unlike upstream.** The core steps whenever it has work, so batch composition
depends on OS scheduling rather than only on the workload — which costs the byte-identical
determinism the conformance suite and the sweep runner both rely on. Upstream defaults it on because
there it buys overlap with the GPU; here it buys realism at the cost of reproducibility, so turning
it on is a deliberate act.

## Modeling a deployment's shape

Each of these changes a number a capacity plan turns on, and each is modeled where it is
observable rather than approximated.

| | what it changes |
|---|---|
| `--tensor-parallel-size` | shards KV and weights per device; near-linear speedup on prefill, sublinear on decode |
| `--pipeline-parallel-size` | shards layers per device; **same step latency**, less memory (no microbatch overlap is modeled) |
| `--enable-lora --max-loras N` | adapters cost KV pool memory, and `max_loras` bounds *distinct* adapters — a real source of queueing |
| `--lora-modules name=path` | each adapter is served under its own model name: `/v1/models` lists it, and a request naming it routes to it |
| `--sliding-window N` | KV per request stops growing with the conversation; concurrency rises in proportion |
| `n > 1` | fans out into `n` engine requests sharing the prompt through the prefix cache — one response, `n` times the decode pressure |
| speculative decoding | fewer steps when acceptance is high, wasted work when it is not; lossless either way |
| an `image_url` content part | 256 placeholder tokens, a separate encoder budget, an encoder pass priced at ViT-L scale, and a cache the second request with the same image hits |
| a KV connector | a second engine pulls a published prefix instead of recomputing it; both sides pay — the producer for its writes, the consumer for its reads — so `kv_role` and your store's bandwidth decide whether it wins against recomputing |
| `--enable-lora` + prefix caching | adapter id partitions the cache, so two tenants with the same prompt do not share blocks |

Two of these carry a knob a simulator cannot derive from anything, because it depends on a model
this engine does not have: `--spec-acceptance-rate` (the agreement between a draft model and its
target) and the structured-output backend's conformance. Measure them on your real pair and
everything downstream is faithful.

## Comparing configurations

This is the part that costs a GPU reservation otherwise. `pvllm bench` mirrors upstream's
`vllm bench` layout — `latency`, `throughput`, `serve` — plus a sweep runner:

```bash
pvllm bench sweep --model meta-llama/Llama-3.1-8B-Instruct --device-card datacenter-80gb --sweep max-num-seqs=1,2,4,8,16 -o sweep.csv
```

One tidy CSV row per cell, ready to plot. Sweepable: `max-num-seqs`, `max-num-batched-tokens`,
`block-size`, `gpu-memory-utilization`, `device-card`, `enable-prefix-caching`,
`enable-chunked-prefill`, `request-rate`, `num-prompts`, `input-len`, `output-len`.

`bench serve` generates arrivals from a gamma process (`--request-rate`, `--burstiness`) and reports
TTFT split into **queue wait** and **prefill** — the distinction a capacity decision turns on, since
a request that waited 200 ms and prefilled in 5 needs more concurrency while one that prefilled for
205 ms needs a smaller batch.

Two things to know. Durations are modeled, so read a sweep for where the knee is and which way a
knob moves things, never for the number itself. And arrivals are seeded from `--seed`, so rerunning
a comparison reruns the same workload — unlike upstream, where the arrival process draws from global
random state.

## Fidelity contract

**Current state: `asserted`, not `verified`.** Golden traces captured from a real vLLM run at the
pinned version do not exist yet. Until they do, the conformance suite runs in self-consistency mode:
it compares against previously recorded pretending-vllm traces, which catches *drift* but not
*divergence from upstream*. `tools/capture_golden_trace.py` ships so the contract can be promoted
when hardware time becomes available. Read this section as design intent, not as a guarantee.

**Exact — a divergence is a bug by definition:**

| | |
|---|---|
| C1 | Scheduler decision sequence per step, and total engine steps to drain a workload |
| C2 | KV block allocation and free order |
| C3 | Prefix cache hit rate and block hash values |
| C4 | Preemption count and victim selection |
| C5 | OpenAI HTTP request and response schema for implemented endpoints |
| C6 | Prometheus metric names, types, labels, and histogram bucket edges |
| C7 | Error codes and failure modes at capacity |

**Approximate, and labeled `modeled` everywhere it surfaces:** step latency, TTFT, ITL, throughput.

**Analytic, exact given the model card and device card:** memory footprint, derived
`num_gpu_blocks`, `max_concurrency`.

**Not modeled at all:** generated text quality; logprob *values* (schema and shape only).

### How the contract is checked

`tests/conformance/` runs four workloads — mixed lengths, a shared prefix, a starved block budget
that forces preemption, and a prompt long enough to chunk — and compares each against a recorded
golden. C5–C7 are schema checks rather than recordings.

The recordings carry **decisions only**: which requests were scheduled, how many tokens each got,
which blocks were allocated and freed in what order, the cache hit rate, which request was preempted
and when. No timestamps, no durations. That separation is deliberate — latency is approximate by
construction, and if a cost-model recalibration failed the conformance suite, the goldens would get
regenerated reflexively and stop catching anything.

```bash
python tools/capture_golden_trace.py --check
```

To promote the contract from `asserted` to `verified`, run the same four workloads against real vLLM
at the pin and replace the goldens; the tests do not change. The recorder attaches to upstream's
`BlockPool` unchanged, and `tools/capture_golden_trace.py` documents the procedure.

### The latency numbers are modeled, not measured

The cost model is a roofline approximation — a compute term, a memory term, and overheads, with a
published error band. It is the one place this project can be wrong in a way that misleads you;
everything else is either exactly right or obviously fake. It ships **uncalibrated** until
`tools/calibrate_cost_model.py` is run against real hardware.

Treat latency output as *shape*, not truth. It will reproduce the qualitative regimes — prefill
compute-bound and linear in tokens, decode memory-bound and nearly flat until KV traffic dominates,
throughput saturating with batch size, TTFT rising under queueing — and it will not tell you what
your p99 is going to be.

One known bias: cascade attention is not modeled. The common-prefix block count is computed and
carried through `SchedulerOutput` exactly as upstream does, and you can read it at
`/debug/cost_model`, but the cost model ignores it. Shared-prefix workloads are therefore modeled
*pessimistically* — a real backend taking the optimization would be faster than this says. Prefix
caching itself, which is the much larger effect, is fully modeled.

## Non-goals

Meaningful text (outputs are synthetic tokens; no weights are ever read). Kernel-level or
microsecond accuracy. Numerical parity of sampling. Import-level drop-in replacement for `vllm` —
compatibility is at the HTTP layer and the offline `LLM` class only. Real multi-machine distribution.

## Development

```bash
python tools/fetch_upstream.py     # vendor the reference tree (~36 MB, gitignored)
uv venv && uv pip install -e ".[dev]"
pytest
```

`tools/spec_sync.py` checks that every module's declared upstream counterpart still exists at the
pin. `tests/unit/test_purity.py` enforces the simulation boundary: no clock, no randomness, and no
`torch`/`transformers` import outside `pvllm/sim/`.

## License

Apache-2.0, matching upstream vLLM.
