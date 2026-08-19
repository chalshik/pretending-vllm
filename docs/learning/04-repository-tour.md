# 04. Repository tour

> **Files:** all of them.
> **Prerequisites:** chapter [03](03-simulation-boundary.md).

This is the reference chapter. Read it once to build a map, then come back to it whenever
you meet a file you do not recognise. Every entry answers "why does this exist" in one
line, with a pointer to the chapter that goes deep.

## Top level

| Path | What it is |
|---|---|
| [`pvllm/`](../../pvllm) | the package |
| [`tests/`](../../tests) | the suite — unit, v1, sim, entrypoints, conformance, property, benchmarks |
| [`tools/`](../../tools) | four maintenance scripts: fetch upstream, check drift, record goldens, run mutations |
| [`vendor/`](../../vendor) | the vendored upstream tree (gitignored, ~36 MB) plus a committed `MANIFEST.sha256` |
| [`README.md`](../../README.md) | the product-facing description and the fidelity contract |
| [`UPSTREAM.md`](../../UPSTREAM.md) | the pin, the tier system, and the delta table of stale spec assumptions |
| [`pretending_vllm_requirements.md`](../../pretending_vllm_requirements.md) | the original requirements draft. Every `R*` / `C*` / `D*` / `F*` reference in the code points here |
| [`pyproject.toml`](../../pyproject.toml) | dependencies (note: no torch, no transformers), lint and type settings |
| [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) | the eight-stage build — see chapter [30](30-testing-and-tooling.md) |

**Reading the requirement tags.** The codebase is dense with references like `R6.5`,
`C3`, `D9`, `F7`. They are not noise:

- `R*` — a numbered requirement from the requirements draft (`R6` is KV cache management,
  `R9` the cost model, and so on).
- `C1`–`C7` — the fidelity contract's conformance classes (chapter
  [29](29-conformance-and-fidelity.md)).
- `B*`, `D*`, `G*`, `NG*` — settled design decisions, deliberate divergences, goals, and
  non-goals.
- `F1`–`F11` — corrections discovered when the real v0.27.1 tree was first read. The
  table in [UPSTREAM.md](../../UPSTREAM.md) records what the draft assumed and what
  upstream actually does. Worth reading: it is a list of the things that are easy to get
  wrong about modern vLLM.

## `pvllm/` — the root modules

| File | Why it exists | Chapter |
|---|---|---|
| [`__init__.py`](../../pvllm/__init__.py) | version, `UPSTREAM_VERSION`, and the boundary diagram | [00](00-orientation.md) |
| [`envs.py`](../../pvllm/envs.py) | the `PVLLM_*` environment surface, mirroring `VLLM_*` one-to-one. Prefixed differently so a real vLLM install side by side cannot be reconfigured by accident | [06](06-configuration.md) |
| [`sampling_params.py`](../../pvllm/sampling_params.py) | what a client asks for: `temperature`, `max_tokens`, `stop`, `n`, `seed`, `logprobs`, output kind | [07](07-requests-and-sampling.md) |
| [`pooling_params.py`](../../pvllm/pooling_params.py) | the same, for a request that returns a vector instead of tokens | [27](27-pooling-and-embeddings.md) |
| [`outputs.py`](../../pvllm/outputs.py) | `RequestOutput`, `CompletionOutput`, `PoolingRequestOutput` — the public result types | [07](07-requests-and-sampling.md) |
| [`timebase.py`](../../pvllm/timebase.py) | the abstract `Clock`. Above the boundary so the engine can hold one without importing the simulator | [16](16-clock-and-determinism.md) |
| [`tracing.py`](../../pvllm/tracing.py) | the `TraceSink` protocol and the trace reader. Above the boundary for the same reason | [20](20-observability.md) |
| [`trace_viewer.py`](../../pvllm/trace_viewer.py) | renders a JSONL trace as a step timeline, text or SVG | [20](20-observability.md) |
| [`conformance.py`](../../pvllm/conformance.py) | records what an engine *decided*, in a form two engines can be compared in | [29](29-conformance-and-fidelity.md) |
| [`conformance_workloads.py`](../../pvllm/conformance_workloads.py) | the four fixed workloads the contract is asserted on | [29](29-conformance-and-fidelity.md) |
| [`logger.py`](../../pvllm/logger.py), [`logging_utils/`](../../pvllm/logging_utils) | logging setup and formatters, mirroring upstream's | — |
| [`utils/`](../../pvllm/utils), [`v1/utils.py`](../../pvllm/v1/utils.py) | small shared helpers (`resolve_obj_by_qualname`, `ConstantList` — the read-only view over a request's growing token list) | — |
| [`plugins/`](../../pvllm/plugins) | entry-point plugin loading, so an out-of-tree platform can be registered | [03](03-simulation-boundary.md) |
| [`py.typed`](../../pvllm/py.typed) | marks the package as typed; `mypy --strict` passes | [30](30-testing-and-tooling.md) |

## `pvllm/config/` — configuration

One file per upstream sub-config, all Tier C: field names, types, and validation intent
match; the resolution logic is ours. → chapter [06](06-configuration.md)

| File | Holds |
|---|---|
| [`vllm.py`](../../pvllm/config/vllm.py) | `VllmConfig`, the composite root. Its `__post_init__` is where the platform gets the last word |
| [`model.py`](../../pvllm/config/model.py) | model name, dtype, `max_model_len`, tokenizer selection |
| [`cache.py`](../../pvllm/config/cache.py) | `block_size` (16), `gpu_memory_utilization` (0.92), prefix caching (on), hash algo (sha256) |
| [`scheduler.py`](../../pvllm/config/scheduler.py) | `max_num_seqs` (1024), `max_num_batched_tokens` (8192, derived), chunked prefill (on), policy |
| [`parallel.py`](../../pvllm/config/parallel.py) | TP / PP / DP sizes, expert parallelism, `worker_cls` |
| [`device.py`](../../pvllm/config/device.py) | `DeviceConfig` and **`SimConfig`** — every simulator knob in one place |
| [`lora.py`](../../pvllm/config/lora.py), [`speculative.py`](../../pvllm/config/speculative.py), [`structured_outputs.py`](../../pvllm/config/structured_outputs.py), [`kv_transfer.py`](../../pvllm/config/kv_transfer.py) | the optional feature configs |
| [`load.py`](../../pvllm/config/load.py), [`observability.py`](../../pvllm/config/observability.py) | weight loading and observability settings |

`SimConfig` is the one to remember: it is the entire "what is fake" surface in a single
dataclass — device card, clock mode, cost model profile, jitter, output length policy,
content policy, spec acceptance rate, seed, trace path.

## `pvllm/engine/` and `pvllm/platforms/`

| File | Why |
|---|---|
| [`engine/arg_utils.py`](../../pvllm/engine/arg_utils.py) | `EngineArgs` / `AsyncEngineArgs`: the flat, user-facing surface (CLI flags, `LLM(**kwargs)`) that resolves into a `VllmConfig` |
| [`platforms/interface.py`](../../pvllm/platforms/interface.py) | the `Platform` abstraction — the boundary's *selection* mechanism |
| [`platforms/sim.py`](../../pvllm/platforms/sim.py) | `SimPlatform`: sets `worker_cls`, answers device questions from a JSON card, aligns block size for state-space models |
| [`platforms/__init__.py`](../../pvllm/platforms/__init__.py) | lazy `current_platform` resolution, with out-of-tree plugins beating builtins |

## `pvllm/v1/` — the engine

### `v1/core/` — scheduling and KV cache

| File | Why | Chapter |
|---|---|---|
| [`sched/scheduler.py`](../../pvllm/v1/core/sched/scheduler.py) | **the centerpiece.** What runs each step, and folding results back | [12](12-scheduler.md) |
| [`sched/output.py`](../../pvllm/v1/core/sched/output.py) | `SchedulerOutput` — the decision that crosses the boundary | [12](12-scheduler.md) |
| [`sched/request_queue.py`](../../pvllm/v1/core/sched/request_queue.py) | FCFS deque and priority heap. The queue's order *is* the admission order | [12](12-scheduler.md) |
| [`sched/utils.py`](../../pvllm/v1/core/sched/utils.py) | `check_stop` — token-level stop conditions, in a load-bearing order | [07](07-requests-and-sampling.md) |
| [`kv_cache_utils.py`](../../pvllm/v1/core/kv_cache_utils.py) | `KVCacheBlock`, the intrusive free-block queue, block hashing, group partitioning | [09](09-kv-cache-blocks.md), [10](10-prefix-caching.md) |
| [`block_pool.py`](../../pvllm/v1/core/block_pool.py) | who owns which block, and what gets evicted next | [09](09-kv-cache-blocks.md) |
| [`kv_cache_manager.py`](../../pvllm/v1/core/kv_cache_manager.py) | the scheduler's view: `get_computed_blocks`, `allocate_slots`, `free` | [09](09-kv-cache-blocks.md) |
| [`kv_cache_coordinator.py`](../../pvllm/v1/core/kv_cache_coordinator.py) | reconciles a cache hit across several KV groups | [11](11-hybrid-kv-groups.md) |
| [`single_type_kv_cache_manager.py`](../../pvllm/v1/core/single_type_kv_cache_manager.py) | per-group bookkeeping: full attention, sliding window, Mamba | [11](11-hybrid-kv-groups.md) |
| [`kv_cache_metrics.py`](../../pvllm/v1/core/kv_cache_metrics.py) | prefix cache effectiveness counters | [20](20-observability.md) |
| [`encoder_cache_manager.py`](../../pvllm/v1/core/encoder_cache_manager.py) | the vision encoder's output cache | [23](23-multimodal.md) |

### `v1/engine/` — the core and the frontends

| File | Why | Chapter |
|---|---|---|
| [`core.py`](../../pvllm/v1/engine/core.py) | `EngineCore`: the step loop, **the clock**, the trace | [17](17-engine-core-and-frontends.md) |
| [`__init__.py`](../../pvllm/v1/engine/__init__.py) | the wire types: `EngineCoreRequest`, `EngineCoreOutput(s)`, `UtilityCall` — msgspec structs | [17](17-engine-core-and-frontends.md) |
| [`core_client.py`](../../pvllm/v1/engine/core_client.py) | how the frontend reaches the core. `InprocClient` is the default | [17](17-engine-core-and-frontends.md) |
| [`core_proc.py`](../../pvllm/v1/engine/core_proc.py), [`core_client_mp.py`](../../pvllm/v1/engine/core_client_mp.py) | the core in its own process, over ZeroMQ | [18](18-multiprocess-engine.md) |
| [`dp_client.py`](../../pvllm/v1/engine/dp_client.py) | several whole engines behind a load-aware router | [24](24-parallelism.md) |
| [`input_processor.py`](../../pvllm/v1/engine/input_processor.py) | validate, tokenize, build the wire request | [07](07-requests-and-sampling.md) |
| [`output_processor.py`](../../pvllm/v1/engine/output_processor.py) | assemble what the client sees; own the detokenizer state and the per-request stats | [17](17-engine-core-and-frontends.md) |
| [`detokenizer.py`](../../pvllm/v1/engine/detokenizer.py) | incremental detokenization and **stop strings** — including the held-back text a stream must not emit | [08](08-tokenizers.md) |
| [`llm_engine.py`](../../pvllm/v1/engine/llm_engine.py) | the synchronous engine the offline `LLM` drives | [17](17-engine-core-and-frontends.md) |
| [`async_llm.py`](../../pvllm/v1/engine/async_llm.py) | the asyncio engine the HTTP server sits on. Cancellation lives here | [17](17-engine-core-and-frontends.md) |
| [`parallel_sampling.py`](../../pvllm/v1/engine/parallel_sampling.py) | `n > 1` fanned out into *n* engine requests | [07](07-requests-and-sampling.md) |

### `v1/executor/`, `v1/worker/`, `v1/attention/`

| File | Why | Chapter |
|---|---|---|
| [`executor/abstract.py`](../../pvllm/v1/executor/abstract.py) | the executor interface; hides how many workers there are |[13](13-worker-and-model-runner.md) |
| [`executor/uniproc_executor.py`](../../pvllm/v1/executor/uniproc_executor.py) | the in-process executor — the only implementation | [13](13-worker-and-model-runner.md) |
| [`worker/sim_worker.py`](../../pvllm/v1/worker/sim_worker.py) | the last object above the boundary; owns the `SimDevice` and `SimModel` | [13](13-worker-and-model-runner.md) |
| [`worker/gpu/model_runner.py`](../../pvllm/v1/worker/gpu/model_runner.py) | **the boundary itself.** Persistent batch, input prep, attention metadata, then the fake forward pass | [13](13-worker-and-model-runner.md) |
| [`worker/gpu/input_batch.py`](../../pvllm/v1/worker/gpu/input_batch.py) | one step's input, prepared (the numpy half of upstream's V2 runner) | [13](13-worker-and-model-runner.md) |
| [`worker/gpu/block_table.py`](../../pvllm/v1/worker/gpu/block_table.py) | block tables and slot mapping — which physical slot each token writes to | [13](13-worker-and-model-runner.md) |
| [`worker/gpu/states.py`](../../pvllm/v1/worker/gpu/states.py) | persistent per-request state held by the worker | [13](13-worker-and-model-runner.md) |
| [`worker/gpu/attn_utils.py`](../../pvllm/v1/worker/gpu/attn_utils.py) | builds attention metadata and the per-layer KV cache spec | [11](11-hybrid-kv-groups.md) |
| [`attention/backends/sim_attn.py`](../../pvllm/v1/attention/backends/sim_attn.py) | the simulated attention backend: metadata, no math | [13](13-worker-and-model-runner.md) |

### `v1/` odds and ends

| File | Why | Chapter |
|---|---|---|
| [`request.py`](../../pvllm/v1/request.py) | `Request` and `RequestStatus` — the state the scheduler mutates every step | [07](07-requests-and-sampling.md) |
| [`outputs.py`](../../pvllm/v1/outputs.py) | `ModelRunnerOutput` — the boundary's return half | [13](13-worker-and-model-runner.md) |
| [`kv_cache_interface.py`](../../pvllm/v1/kv_cache_interface.py) | the specs: full attention, sliding window, MLA, Mamba, and the group/config types | [11](11-hybrid-kv-groups.md) |
| [`metrics/loggers.py`](../../pvllm/v1/metrics/loggers.py) | the Prometheus surface — names, types, labels, bucket edges | [20](20-observability.md) |
| [`metrics/stats.py`](../../pvllm/v1/metrics/stats.py) | per-step and per-request statistics | [20](20-observability.md) |
| [`structured_output/`](../../pvllm/v1/structured_output) | the grammar manager, backend interface, and per-request state | [21](21-structured-output.md) |

## `pvllm/entrypoints/` — the surfaces a product touches

| Path | Why | Chapter |
|---|---|---|
| [`llm.py`](../../pvllm/entrypoints/llm.py) | the offline `LLM` class: `generate`, `chat`, `embed` | [17](17-engine-core-and-frontends.md) |
| [`openai/api_server.py`](../../pvllm/entrypoints/openai/api_server.py) | the FastAPI app: every route, the lifespan, `/metrics`, `/health` | [19](19-openai-server.md) |
| [`openai/completion/`](../../pvllm/entrypoints/openai/completion), [`openai/chat_completion/`](../../pvllm/entrypoints/openai/chat_completion) | request/response schemas and the serving logic for the two classic endpoints | [19](19-openai-server.md) |
| [`openai/responses/`](../../pvllm/entrypoints/openai/responses) | the Responses API: named SSE events, no `[DONE]`, an optional response store | [19](19-openai-server.md) |
| [`openai/models/serving.py`](../../pvllm/entrypoints/openai/models/serving.py) | `/v1/models`, including LoRA adapters served under their own names | [22](22-lora.md) |
| [`openai/multimodal.py`](../../pvllm/entrypoints/openai/multimodal.py) | turns OpenAI content parts into placeholder token runs | [23](23-multimodal.md) |
| [`openai/structured_outputs.py`](../../pvllm/entrypoints/openai/structured_outputs.py) | maps `response_format` / `guided_*` fields onto a grammar constraint | [21](21-structured-output.md) |
| [`pooling/embed/`](../../pvllm/entrypoints/pooling/embed) | `/v1/embeddings` | [27](27-pooling-and-embeddings.md) |
| [`serve/tokenize/serving.py`](../../pvllm/entrypoints/serve/tokenize/serving.py) | `/tokenize`, `/detokenize` | [08](08-tokenizers.md) |
| [`serve/utils/`](../../pvllm/entrypoints/serve/utils) | the error path: every failure leaves in vLLM's `{"error": {...}}` envelope | [19](19-openai-server.md) |
| [`serve/dev/api_router.py`](../../pvllm/entrypoints/serve/dev/api_router.py), [`serve/dev/introspect.py`](../../pvllm/entrypoints/serve/dev/introspect.py) | the read-only `/debug/*` routes and the live introspector behind them | [20](20-observability.md) |
| [`cli/main.py`](../../pvllm/entrypoints/cli/main.py) | the `pvllm` command: `serve`, `trace`, `bench`, `complete`, `chat` | [05](05-first-run.md) |
| [`cli/openai.py`](../../pvllm/entrypoints/cli/openai.py) | `pvllm complete` / `pvllm chat` — stdlib-only clients for a running server | [05](05-first-run.md) |
| [`cli/benchmark/`](../../pvllm/entrypoints/cli/benchmark) | `pvllm bench {latency,throughput,serve,sweep}` argument parsing | [28](28-benchmarking.md) |

## `pvllm/sim/` — the simulator

Covered in chapter [03](03-simulation-boundary.md) and unpacked in chapters
[14](14-memory-model.md)–[16](16-clock-and-determinism.md). Two subdirectories hold the
data:

```
pvllm/sim/models/     dense-0.6b, dense-8b, dense-70b, moe-8x7b,
                      mla-16b, hybrid-4b, hybrid-ssm-8b, tiny-test
pvllm/sim/hardware/   datacenter-80gb, workstation-24gb, tiny-2gb
```

Each is a JSON file of declared architecture or declared device capability, with a
`provenance` field that says how much to trust it. This is the "hardware is a JSON file"
idea made literal:

```json
{
  "name": "datacenter-80gb",
  "memory_bytes": 85899345920,
  "memory_bandwidth": 3350000000000,
  "peak_flops": { "bfloat16": 494700000000000, ... },
  "mfu": 0.45, "bw_eff": 0.8, "link_eff": 0.75,
  "provenance": "Uncalibrated approximation ... Figures are published vendor peaks
                 for the device class, NOT measurements ..."
}
```

## The other supporting packages

| Path | Why |
|---|---|
| [`tokenizers/`](../../pvllm/tokenizers) | the tokenizer protocol, registry, byte-level mock, real HF tokenizer, incremental detokenization helpers → chapter [08](08-tokenizers.md) |
| [`transformers_utils/config.py`](../../pvllm/transformers_utils/config.py) | model metadata loading — resolves a name to a model card |
| [`multimodal/inputs.py`](../../pvllm/multimodal/inputs.py) | `MultiModalFeatureSpec`: how many tokens an image occupies, where, and its hash → chapter [23](23-multimodal.md) |
| [`lora/request.py`](../../pvllm/lora/request.py) | `LoRARequest`, whose `lora_int_id` joins the prefix cache key → chapter [22](22-lora.md) |
| [`distributed/kv_transfer/`](../../pvllm/distributed/kv_transfer) | the KV connector interface and a connector over a simulated shared store → chapter [26](26-kv-disaggregation.md) |
| [`benchmarks/`](../../pvllm/benchmarks) | `latency`, `throughput`, `serve`, `sweep`, plus `lib/` (arrival processes, metric shapes, the runner) → chapter [28](28-benchmarking.md) |

## `tests/` and `tools/`

| Path | Why | Chapter |
|---|---|---|
| `tests/unit/` | config, platform, purity lint, the mutation tool's own tests | [30](30-testing-and-tooling.md) |
| `tests/v1/` | the engine: scheduler, block pool, prefix cache, worker, LoRA, spec decode, hybrid groups, and per-milestone review-regression files | [30](30-testing-and-tooling.md) |
| `tests/sim/` | the simulator: clock, cost model, memory, RNG, parallelism arithmetic, trace | [30](30-testing-and-tooling.md) |
| `tests/entrypoints/` | HTTP surface, debug endpoints, offline `LLM`, the CLI client | [19](19-openai-server.md) |
| `tests/conformance/` | C1–C4 against recorded goldens; C5–C7 as schema checks | [29](29-conformance-and-fidelity.md) |
| `tests/property/` | hypothesis properties over the scheduler | [30](30-testing-and-tooling.md) |
| `tests/mutations.toml` | 30 entries: a guarantee, the edit that breaks it, the test that must notice | [30](30-testing-and-tooling.md) |
| `tools/fetch_upstream.py` | vendors the reference tree; stdlib-only so it runs before install | [31](31-extending-the-port.md) |
| `tools/spec_sync.py` | fails when a declared upstream counterpart moved or vanished | [31](31-extending-the-port.md) |
| `tools/capture_golden_trace.py` | records and checks the conformance goldens | [29](29-conformance-and-fidelity.md) |
| `tools/mutate.py` | runs the mutation catalogue | [30](30-testing-and-tooling.md) |

## Reading order for the source itself

If you want to read the *code* rather than the docs, this order introduces nothing before
it is needed:

1. `pvllm/v1/request.py` — the object everything else manipulates
2. `pvllm/v1/core/kv_cache_utils.py` then `block_pool.py` — the memory bookkeeping
3. `pvllm/v1/core/kv_cache_manager.py` — the scheduler's view of it
4. `pvllm/v1/core/sched/output.py` then `scheduler.py` — the decisions
5. `pvllm/v1/engine/core.py` — the loop that drives it all
6. `pvllm/v1/worker/gpu/model_runner.py` — the boundary
7. `pvllm/sim/device.py`, `cost_model.py`, `memory.py` — the fake half
8. `pvllm/v1/engine/async_llm.py` then `entrypoints/openai/api_server.py` — the surface

Every one of those files opens with a docstring that explains *why* it is shaped the way
it is. Those docstrings are the primary source for this whole series; when a chapter here
and a docstring there disagree, the docstring is right and this is a bug.

## Check yourself

- Where does the entire "what is fake" configuration surface live?
- Which two directories hold JSON rather than code, and what do they replace?
- You see `R6.7` in a comment. Where do you look it up?
- Which file would you open first to find out why a request was preempted?

## Next

[05. Your first run](05-first-run.md) — install it, generate, serve, and read a trace.
