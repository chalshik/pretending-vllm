# pretending-vllm

A structurally faithful reimplementation of the vLLM V1 engine in which every layer above the device
is real logic, and only the device and the model are simulated.

Point a product that speaks the OpenAI API at it and exercise streaming, cancellation, queueing,
backpressure, capacity limits, and metrics scraping — deterministically, in CI, on a laptop, with no
GPU and no model weights.

**Upstream pin: [vLLM v0.27.1](https://github.com/vllm-project/vllm/tree/v0.27.1)** (2026-08-11).
See [UPSTREAM.md](UPSTREAM.md).

> **Status: M1 complete.** The OpenAI server, the offline `LLM` class, `AsyncLLM`, and `/metrics`
> all work. Prefix caching, chunked prefill, and preemption-under-load land in M2; tensor
> parallelism, LoRA, and speculative decoding in M4. Run with `--no-enable-prefix-caching` until M2 —
> the engine refuses to pretend a cache it does not have.

```bash
pvllm serve --model meta-llama/Llama-3.1-8B-Instruct --device-card datacenter-80gb \
            --no-enable-prefix-caching
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

### The latency numbers are modeled, not measured

The cost model is a roofline approximation — a compute term, a memory term, and overheads, with a
published error band. It is the one place this project can be wrong in a way that misleads you;
everything else is either exactly right or obviously fake. It ships **uncalibrated** until
`tools/calibrate_cost_model.py` is run against real hardware.

Treat latency output as *shape*, not truth. It will reproduce the qualitative regimes — prefill
compute-bound and linear in tokens, decode memory-bound and nearly flat until KV traffic dominates,
throughput saturating with batch size, TTFT rising under queueing — and it will not tell you what
your p99 is going to be.

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
