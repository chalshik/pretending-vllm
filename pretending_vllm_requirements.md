# pretending-vllm

Requirements Specification, draft v2.

> **Amended 2026-08-15 against the real upstream tree.** Draft v1 was written before the v0.27.1
> source was available and extrapolated forward; eleven assumptions were checked against
> `vendor/vllm-0.27.1/` and eight were stale. Amendments are marked **[F1]**–**[F11]** inline and
> catalogued with evidence in [UPSTREAM.md](UPSTREAM.md). The largest are: Model Runner V2 has
> already landed and is the default (D5 → **D6**), the R12.1 metric names carry no `_total` suffix,
> and the section 5 size budget is unreachable by a factor of three (replaced by fidelity tiers).

A structurally faithful reimplementation of the vLLM V1 engine in which every layer above the device
is real logic, and only the device and the model are simulated.

Upstream reference pinned at vLLM v0.27.1 (released 2026 08 11). The pin is recorded in
`UPSTREAM.md` at the repo root and in every golden trace.

## 0. Settled decisions

These were open. They are now closed. Rationale is included so they can be reopened deliberately
rather than accidentally.

* D1. Distribution name `pretending-vllm`. Import package `pvllm`. Every module path beneath `pvllm/`
  mirrors the upstream path beneath `vllm/` exactly, so `pvllm/v1/core/sched/scheduler.py` is a
  direct counterpart of `vllm/v1/core/sched/scheduler.py`. A short import root is chosen over the
  full name so that call sites read like upstream call sites, and a distinct root is chosen over
  shadowing `vllm` so both can be installed in the same environment for diffing and conformance.
* D2. The multiprocess engine core is deferred to P2. P0 and P1 ship the `EngineCoreClient`
  abstraction with only the in process implementation. The clock ownership rule (R19.1) is enforced
  from the first commit, because that is the part that cannot be retrofitted cheaply.
* D3. Default tokenizer is `MockTokenizer`, so the base install has no `transformers` dependency.
  A real Hugging Face tokenizer is supported through the optional `realtok` extra and is mandatory
  for conformance class C3, because prefix cache hit rates on real text depend on the exact
  tokenization.
* D4. Golden traces from a real vLLM run are assumed unavailable at project start. Therefore the
  fidelity contract in section 3 ships in `asserted` state, marked as such in the README, and the
  repo includes `tools/capture_golden_trace.py`, an instrumentation script to be run against a real
  vLLM when hardware becomes available. The contract is promoted to `verified` only when the traces
  are checked in. Until then C1 to C4 run in self consistency mode (regression against previously
  recorded pretending-vllm traces), which catches drift but not divergence from upstream.
* ~~D5. Upstream pin is v0.27.1. Note that upstream is mid migration to Model Runner V2 [...]. This
  spec implements the classic model runner shape as it exists at the pin.~~ **Superseded by D6.**
* D6 **[F1]**. Upstream pin is v0.27.1, and **Model Runner V2 has already landed and is the
  default**. `VllmConfig.use_v2_model_runner` returns True for any dense, non-MoE, non-hybrid
  generate model. This spec therefore mirrors the V2 shape at `vllm/v1/worker/gpu/`, not the legacy
  `gpu_model_runner.py`. Three reasons, in order of weight: V2 is what "latest official vLLM
  behavior" now means; V2 is 1,723 lines against the legacy runner's 7,928, so the port is
  tractable; and V2's real logic is *already* the numpy path (`query_start_loc_np`, `idx_mapping_np`,
  `is_prefilling_np`) with torch only mirroring results to device, so `SimModelRunner` keeps
  upstream's numpy half near-verbatim and simply drops the device copies. `PVLLM_USE_V2_MODEL_RUNNER=0`
  raises rather than silently falling back — there is no V1 runner here to fall back to.
* D7 **[F9]**. The global LOC budget in section 5 is replaced by four fidelity tiers. See
  *Fidelity tiers* below.
* D8. All four consumption surfaces are first class: the OpenAI HTTP server, the offline `LLM`
  class, `AsyncLLM`, and Prometheus `/metrics`. The milestones in section 9 are reordered so the
  HTTP surface lands in M1 rather than P2 — C5 (HTTP schema) is independent of C1–C4 (scheduler
  fidelity), so the product surface can stabilize while depth is added underneath it.
* D9. "Full debug mode" means the structured JSONL trace (R19.3), HTTP introspection endpoints, and
  the trace viewer (R19.4) — machine-readable and inspectable, not a verbose stdout narration.

## 1. Executive architecture

One sentence: keep the entire control plane of vLLM real, and replace exactly one leaf, the thing
that turns scheduled token positions into logits while consuming time and memory.

```
REAL  entrypoints => processor => EngineCoreClient => EngineCore => Scheduler => KVCacheManager
                                                        |
                                                        v
REAL                                        Executor => Worker
                                                        |
================== SIMULATION BOUNDARY ==================|=========================
                                                        v
FAKE                          SimModelRunner => SimDevice (memory ledger + cost model)
                                                SimModel  (token generator)
```

Three consequences that justify the design:

1. Scheduler and cache behavior is not approximated, it is reproduced. The same workload yields the
   same block allocation trace and the same prefix cache hit rate as real vLLM.
2. Hardware becomes a JSON file. "Does 70B at 128k context fit at gpu_memory_utilization 0.92 on
   eight devices" is answerable without owning the hardware.
3. Wall clock becomes a dial. A 30 minute load test runs in 4 seconds under a virtual clock, or in
   real time with sleeps when a product under test needs true latency.

Highest risk requirement is R9, the cost model. It is the only place the project can be wrong in a
way that misleads a downstream consumer. Everything else is either exactly right or obviously fake.

## 2. Goals and non goals

### Goals

* G1. Understanding. A request traceable from HTTP to token in under 6000 lines of Python, with no
  torch, no CUDA, no kernels.
* G2. Structural fidelity. Module paths, class names, and method signatures mirror upstream closely
  enough that a diff against the real repo is a study exercise and knowledge transfers both ways.
* G3. Control plane behavioral fidelity. Scheduling decisions, KV block accounting, prefix cache
  hits, preemption events, and metric semantics match real vLLM for the same inputs.
* G4. Usable as a test double. A product speaking the OpenAI API points at it and exercises
  streaming, cancellation, queueing, backpressure, capacity limits, and metrics scraping,
  deterministically, in CI.
* G5. Cheap experiments. Sweep max_num_batched_tokens, block_size, gpu_memory_utilization,
  scheduling policy, and speculative acceptance rate over thousands of runs in minutes.

### Non goals

* NG1. Meaningful text. Outputs are synthetic tokens. No weights are ever read.
* NG2. Kernel level or microsecond accuracy. The cost model is a roofline approximation with a
  published error band.
* NG3. Numerical parity of sampling. Same parameter surface, different floating point path.
* NG4. Import level drop in replacement for `vllm`. Compatibility is at the HTTP layer and the
  offline `LLM` class only.
* NG5. Real multi machine distribution. Parallelism is simulated in process.

## 3. Fidelity contract

Lives in the README. Enforced by the conformance suite (R21). Current state: `asserted` (see D4).

Exact, and a divergence is a bug by definition:

* C1. Scheduler decision sequence per step, and total engine steps to drain a workload.
* C2. KV block allocation and free order.
* C3. Prefix cache hit rate and block hash values.
* C4. Preemption count and victim selection.
* C5. OpenAI HTTP request and response schema for implemented endpoints.
* C6. Prometheus metric names, types, labels, and histogram bucket edges.
* C7. Error codes and failure modes at capacity.

Approximate, and labeled as modeled everywhere it surfaces:

* Step latency, TTFT, ITL, throughput. Calibrated, error band published.

Analytic, exact given the model card and device card:

* Memory footprint, derived num_gpu_blocks, max_concurrency.

Not modeled at all:

* Generated text quality. Logprob values (schema and shape only).

## 4. The simulation boundary

Exactly one interface crosses it, the same one upstream uses:

```
SimModelRunner.execute_model(scheduler_output: SchedulerOutput) => ModelRunnerOutput
```

* B1. No code above the boundary knows it is talking to a simulator. No `if simulated:` anywhere in
  `v1/core`, `v1/engine`, or `entrypoints`.
* B2. Selection happens through the platform abstraction, exactly as an out of tree hardware backend
  is selected upstream. `current_platform` resolves to `SimPlatform`, which supplies the worker
  class, the attention backend class, and the device communicator. This mirrors the real
  `vllm.platform_plugins` entry point mechanism, so the seam is the one hardware vendors actually
  use.
* B3. The `sim` package is the only place allowed to invent numbers: memory ledger, clock, cost
  model, hardware database, model database, token generator. Nothing else imports randomness or wall
  clock time.
* B4. Below the boundary is pure. Same seed plus same SchedulerOutput sequence gives byte identical
  output and identical elapsed virtual time.

## 5. Repository layout

Upstream paths keep upstream names. New paths exist only under `platforms/` and `sim/`.

```
pretending-vllm/
  pyproject.toml
  README.md
  UPSTREAM.md                      pinned upstream version and docs tree
  pvllm/
    __init__.py
    envs.py                        env var surface mirroring VLLM_*
    logger.py
    outputs.py                     RequestOutput, CompletionOutput
    sampling_params.py
    config/
      model.py cache.py parallel.py scheduler.py device.py load.py
      lora.py speculative.py observability.py structured_outputs.py
      kv_transfer.py vllm.py       VllmConfig composite root
    tokenizers/                    [F2] upstream moved these out of transformers_utils/
      protocol.py registry.py mock.py
    engine/
      arg_utils.py                 EngineArgs, AsyncEngineArgs
    entrypoints/
      llm.py launcher.py
      cli/main.py cli/serve.py
      cli/benchmark/{main.py, latency.py, serve.py, throughput.py, sweep.py}   [F4]
      openai/api_server.py openai/cli_args.py
      openai/completion/{api_router.py, protocol.py, serving.py}
      openai/chat_completion/{api_router.py, protocol.py, serving.py}
      openai/models/{api_router.py, protocol.py, serving.py}
      serve/tokenize/{api_router.py, protocol.py, serving.py}
      serve/instrumentator/{health.py, metrics.py}
      serve/utils/{request_logger.py, error_response.py}
    v1/
      request.py                   Request, RequestStatus
      outputs.py                   ModelRunnerOutput, EngineCoreOutput(s), SamplerOutput
      kv_cache_interface.py        KVCacheSpec, KVCacheConfig, KVCacheGroupSpec
      serial_utils.py              msgspec encode and decode
      core/
        sched/{interface.py, scheduler.py, output.py, request_queue.py, utils.py}
        block_pool.py
        kv_cache_manager.py
        kv_cache_coordinator.py
        single_type_kv_cache_manager.py
        kv_cache_utils.py
        kv_cache_metrics.py                [F5] prefix cache metrics live here now
        encoder_cache_manager.py
      engine/
        core.py core_client.py async_llm.py llm_engine.py coordinator.py
        input_processor.py                 [F3] was processor.py
        output_processor.py detokenizer.py parallel_sampling.py
      executor/{abstract.py, uniproc_executor.py, multiproc_executor.py}
      worker/
        worker_base.py sim_worker.py
        gpu/{model_runner.py, input_batch.py, attn_utils.py, block_table.py}
                                           [D6/F1] mirrors upstream's V2 tree layout
      attention/backends/sim_attn.py      metadata builder only, no math
      sample/{sampler.py, metadata.py, logits_processor/}
      spec_decode/{ngram_proposer.py, draft_proposer.py, metrics.py}
      structured_output/{__init__.py, backend_types.py, request.py}   [F4] not manager/grammar
      metrics/{stats.py, loggers.py, prometheus.py}
    platforms/{interface.py, sim.py, __init__.py}
    model_executor/
      models/{sim_causal_lm.py, registry.py}
      model_loader/sim_loader.py
      layers/{sim_attention.py, logits_processor.py}
    transformers_utils/config.py   [F2] tokenizers moved to the top-level package above
    distributed/{parallel_state.py, sim_communicator.py}
    lora/{request.py, sim_manager.py}
    multimodal/{registry.py, inputs.py}
    sim/                           the only unreal code in the repo
      clock.py       VirtualClock, RealClock, ScaledClock
      device.py      SimDevice, streams, sync, OOM
      memory.py      memory ledger, profiling run
      cost_model.py  roofline latency
      hardware_db.py hardware/*.json
      model_db.py    models/*.json
      weights.py     fake weight materialization and load time
      model.py       SimModel token generator
      rng.py         seeded, request scoped
      trace.py       JSONL event trace writer
    benchmarks/{latency.py, throughput.py, serve.py, datasets.py, sweep.py}
  tools/
    capture_golden_trace.py        instrumentation patch for a real vLLM run
    calibrate_cost_model.py
  tests/{unit/, v1/, entrypoints/, sim/, conformance/, property/}
  examples/
  docs/
```

### Fidelity tiers **[F9, D7]**

~~Size budget: no module over 400 lines. Engine plus scheduler plus KV manager under 2500 lines. The
package under 6000 lines excluding tests and JSON data.~~

That budget was unreachable and in direct conflict with R5's "a faithful port, not an approximation"
and the C1–C4 exactness contract. Measured against the vendored tree: the upstream subset being
mirrored is **~148,000 lines**, and `vllm/v1/core/sched/scheduler.py` alone is **2,915** — larger
than the entire "engine plus scheduler plus KV manager" allowance. Holding the budget would have
meant an approximating port, which forfeits the strongest claim in this document.

Fidelity is instead spent where the contract demands it. Every module declares its tier in its
header alongside its upstream counterpart, and `tools/spec_sync.py` reads both.

| Tier | Meaning | Rule | Applies to |
|---|---|---|---|
| **A** | Line-for-line | Same method names, same order of operations, same branch structure. A behavioral divergence is a bug by definition. | Everything C1–C4 binds: `v1/core/sched/*`, `block_pool.py`, `kv_cache_{manager,coordinator,utils,metrics}.py`, `single_type_kv_cache_manager.py`, `v1/request.py` |
| **B** | Signature-faithful, body-thinned | Same public API and observable behavior; internals may drop unsupported paths. | `v1/engine/*`, `v1/executor/*`, `v1/worker/*`, `v1/metrics/*`, `entrypoints/*` |
| **C** | Shape-only | Field names, types, and validation *intent* match; implementation is ours. | `config/*`, `v1/sample/*`, `tokenizers/*`, `engine/arg_utils.py` |
| **D** | Invented | No upstream counterpart. The only place allowed randomness, wall-clock, or invented numbers (B3). | `sim/*`, `platforms/sim.py` |

**Unsupported-path discipline.** A dropped upstream code path raises `NotImplementedError` naming the
upstream feature. It never silently no-ops. For a test double this matters more than it looks: a
product sending an unmodeled sampling parameter must find out immediately, not through subtly wrong
behavior three layers downstream.

Per-module cap: **600 lines**, with `v1/core/sched/scheduler.py` exempt (target ~1,200 against
upstream's 2,915). Revised totals for M0–M3, by subsystem:

| subsystem | upstream ~LOC | target | tier |
|---|---:|---:|---|
| `v1/core` (sched + KV) | 13,700 | 3,500 | A |
| `v1/engine` | 12,000 | 2,200 | B |
| `v1/worker` + sim runner | 17,700 | 1,400 | B |
| `v1/executor` | 4,300 | 400 | B |
| `config/` + `engine/arg_utils.py` | 19,200 | 2,000 | C |
| `entrypoints/` | 23,500 | 3,000 | B |
| `v1/metrics` | 4,200 | 900 | B |
| `v1/sample` | 5,300 | 600 | C |
| `tokenizers/` | 3,200 | 400 | C |
| `platforms/` | 5,000 | 300 | B/D |
| `sim/` | — | 1,800 | D |
| **total** | **~108,000** | **~16,500** | |

G1 survives in a form that is actually defensible: the *read path* for a single request —
entrypoint → processor → engine core → scheduler → KV manager → runner — stays under 6,000 lines.
That is the number worth claiming, because it is the one a person actually reads.

## 6. Component requirements

Each requirement is numbered and testable. Untagged requirements are P0 or P1.

### R1. Configuration

* R1.1. `VllmConfig` composes ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig,
  DeviceConfig, LoadConfig, LoRAConfig, SpeculativeConfig, ObservabilityConfig,
  StructuredOutputsConfig, KVTransferConfig. Field names match upstream.
* R1.2. `EngineArgs` exposes at minimum: model, tokenizer, dtype, max_model_len, block_size,
  gpu_memory_utilization, max_num_batched_tokens, max_num_seqs, max_num_partial_prefills,
  long_prefill_token_threshold, enable_prefix_caching, enable_chunked_prefill,
  tensor_parallel_size, pipeline_parallel_size, data_parallel_size, seed, disable_log_stats,
  scheduling_policy, max_logprobs, speculative_config, kv_transfer_config.
* R1.3. Simulator fields live in `SimConfig`, reached through DeviceConfig: device_card,
  num_devices, clock_mode, time_scale, jitter_sigma, cost_model_profile, model_card,
  output_length_policy, seed.
* R1.4. Defaults and resolution order match upstream wherever the field exists upstream. Prefix
  caching and chunked prefill default to enabled.
* R1.5. Derived value computation (max_num_batched_tokens defaults, max_model_len clamping,
  validation failures) reproduces upstream error intent.

### R2. Entrypoints

* R2.1. Offline `LLM` with `generate(prompts, sampling_params)` returning RequestOutput, plus
  `chat()`.
* R2.2. Server endpoints: POST /v1/completions, POST /v1/chat/completions, GET /v1/models,
  POST /tokenize, POST /detokenize, GET /health, GET /ping, GET /metrics, GET /version.
  POST /v1/embeddings at P3, returning a synthetic vector.
* R2.3. Streaming over server sent events including stream_options.include_usage, correct chunk
  shape, finish_reason values stop, length, abort.
* R2.4. Client disconnect aborts the request in the engine and frees its blocks within one step.
* R2.5. Error parity: context length exceeded, unsupported sampling parameter, model not found,
  invalid grammar.
* R2.6. CLI: `pvllm serve`, `pvllm bench {latency, throughput, serve}`, `pvllm complete`.
* R2.7. Startup does not block on load. /health reports ready only after simulated load and
  profiling complete.

### R3. Input processing

* R3.1. `Processor` validates params, applies the chat template (P2), tokenizes, and builds
  EngineCoreRequest.
* R3.2. Tokenizer is pluggable. Default `MockTokenizer` is deterministic and reversible with vocab
  size taken from the model card, whitespace and byte fallback rules, and a fixed set of special
  ids. A real Hugging Face tokenizer is available through the `realtok` extra and is required for
  conformance class C3.
* R3.3. Prompt token ids may be supplied directly, bypassing tokenization, for trace replay.
* R3.4. Multimodal inputs accepted as opaque placeholders with a declared token count and a content
  hash (P3), enough to drive the encoder cache and the prefix cache extra keys.

### R4. Engine core and IPC

* R4.1. `EngineCore` owns the scheduler, the executor, the structured output manager, and runs
  `step()`.
* R4.2. `EngineCoreClient` has three intended implementations: in process (P0), multiprocess
  synchronous and multiprocess asynchronous (P2). Multiprocess uses real OS processes and ZeroMQ
  with msgspec framing, so serialization cost and backpressure are real rather than modeled.
* R4.3. `AsyncLLM` runs the frontend loop: submission, output queue draining, per request asyncio
  queues, abort propagation.
* R4.4. Determinism under multiprocess: the engine core is the sole owner of the clock and stamps
  every output. The frontend never reads the clock. In process is the default for tests,
  multiprocess is the default for serving.
* R4.5. Graceful shutdown, engine dead detection, and error propagation to in flight requests.

### R5. Scheduler

The centerpiece. A faithful port, not an approximation.

* R5.1. Unified queue model: `waiting` and `running`, no prefill versus decode distinction. A
  decision is `{request_id: num_tokens}` plus block tables.
* R5.2. `schedule()` phases in upstream order: running requests first, then encoder inputs against a
  separate budget, then admission from waiting.
* R5.3. Token budget of max_num_batched_tokens per step, decremented per scheduled token, never
  exceeded. max_num_seqs caps concurrent running requests.
* R5.4. Chunked prefill splits a prompt longer than the remaining budget across steps.
  long_prefill_token_threshold and max_num_partial_prefills are respected.
* R5.5. Preemption by recompute when allocate_slots fails for a running request: free all blocks,
  reset the computed token count, push to the front of waiting, increment num_preemptions. Victim
  order matches upstream.
* R5.6. Policies: fcfs and priority (priority then arrival time).
* R5.7. `update_from_output` maps ModelRunnerOutput back to request state, appends tokens, delegates
  stop detection, and produces EngineCoreOutputs.
* R5.8. Finished, aborted, and preempted bookkeeping, including finished_req_ids propagation so the
  worker drops cached state.
* R5.9. Cascade attention decision, encoder budget, and structured output bitmask hooks exist as
  call sites even when the work below them is simulated.
* R5.10. Emits one structured trace event per step containing the full SchedulerOutput in a stable
  schema.

### R6. KV cache management

* R6.1. `BlockPool` with a free block queue implemented as a doubly linked list, giving upstream
  eviction order (least recently freed first, cached blocks evicted only when reached).
* R6.2. `KVCacheBlock` carries block_id, ref_cnt, and a block hash assigned when the block is full
  and cleared on eviction.
* R6.3. Block hash construction matches upstream: parent block hash, the tuple of token ids in the
  block, and extra keys (LoRA id, multimodal content hash, cache_salt). Hash function selectable
  between the builtin hash and sha256.
* R6.4. `get_computed_blocks()` returns the longest cached prefix, honoring the rule that at least
  one token must be recomputed.
* R6.5. `allocate_slots()`: compute the new blocks needed, fail early if insufficient, touch computed
  blocks (increment ref_cnt, remove from the free queue), pop new blocks from the free queue head,
  cache full blocks immediately.
* R6.6. `free()` returns blocks in reverse order so the sequence tail is evicted first.
* R6.7. `KVCacheCoordinator` plus per group managers so that hybrid models (full attention plus
  sliding window plus state space layers) are expressible. Hybrid support lands at P3, but the group
  abstraction exists from P1 so the shape is right.
* R6.8. `EncoderCacheManager` with its own budget (P3).
* R6.9. Prefix cache metrics: queries, hits, hit rate, cached block count, resettable.
* R6.10. `reset_prefix_cache()` method and endpoint.

### R7. Executor and worker

* R7.1. `Executor` abstract with UniProc and MultiProc implementations, `collective_rpc`,
  `determine_available_memory`, `initialize_from_config`, `execute_model`.
* R7.2. `SimWorker` mirrors GPUWorker: init_device, load_model, determine_available_memory,
  initialize_cache, compile_or_warm_up_model, execute_model.
* R7.3. The worker holds persistent per request state (`SimInputBatch`) updated incrementally from
  the scheduler diff, exactly as upstream does, so the cost of state churn is visible in the design.
* R7.4. Block tables materialized as integer arrays per request, sized max_model_len over
  block_size, so metadata memory is accounted for and the indexing logic is real.

### R8. Model runner and attention metadata

* R8.1. `execute_model` order: update the persistent batch, build attention metadata, resolve which
  slots are read and written, ask the cost model for a duration, advance the clock, invoke SimModel,
  run the sampler, return ModelRunnerOutput.
* R8.2. Attention metadata is real: query_start_loc, seq_lens, slot_mapping, block_table,
  num_prefill_tokens, num_decode_tokens. It is the cost model input, so bugs in it are observable.
* R8.3. Slot mapping validation: every written slot must lie inside an allocated block owned by that
  request. Violations raise, they do not warn. This turns the simulator into a correctness oracle for
  the KV manager.
* R8.4. Graph capture is simulated: a set of captured batch sizes, a capture time cost at startup,
  and a lower per step launch overhead when the batch size matches a captured shape and no chunked
  prefill is present.

### R9. Cost model, highest risk

* R9.1. The device card supplies peak dense FLOPs at the model dtype, HBM bandwidth, HBM capacity,
  interconnect bandwidth per direction, kernel launch overhead, and achievable efficiency factors.
* R9.2. Per step latency is computed from first principles as the maximum of a compute term and a
  memory term, plus overheads:

```
flops_step  = 2 * P_active_local * T_step
            + 4 * L_local * n_heads_local * head_dim * sum_r(T_new_r * ctx_r)
t_compute   = flops_step / (mfu * peak_flops)

bytes_step  = weight_bytes_local + kv_bytes_touched(batch) + activation_traffic(T_step)
t_memory    = bytes_step / (bw_eff * hbm_bandwidth)

t_comm      = tp_allreduce_volume(T_step) / (link_eff * link_bandwidth) * L_local
t_overhead  = launch_cost * (n_kernels_captured if graph_hit else n_kernels_eager)

t_step      = max(t_compute, t_memory) + t_comm + t_overhead
t_step      = t_step * (1 + jitter),  jitter ~ N(0, sigma), seeded
```

* R9.3. The model must reproduce the qualitative regimes with no special casing: prefill compute
  bound and linear in tokens, decode memory bound and nearly flat until KV traffic dominates,
  throughput saturating with batch size, TTFT rising under queueing.
* R9.4. Calibration: ship at least three profiles derived from published benchmark numbers, plus
  `tools/calibrate_cost_model.py`, which fits mfu, bw_eff, and launch_cost to a CSV of observed
  (config, batch, tokens, latency) rows from a real vLLM run. Publish the residual error.
* R9.5. Honesty: the README and the /metrics help text state that latency is modeled. No consumer
  should mistake it for measured.
* R9.6. A `constant` profile exists for tests wanting pure determinism and speed.

### R10. Memory model

* R10.1. Memory ledger with named pools: weights, activation peak, KV cache, non torch overhead,
  graph memory. Exceeding capacity raises SimOutOfMemoryError, shaped like the upstream message.
* R10.2. Analytic sizes:

```
weight_bytes       = P * dtype_bytes / TP  (plus unsharded embedding terms)
kv_bytes_per_token = 2 * n_kv_heads_local * head_dim * kv_dtype_bytes * L_local
kv_bytes_per_block = block_size * kv_bytes_per_token
usable             = capacity * gpu_memory_utilization
kv_pool            = usable minus weight_bytes minus activation_peak minus non_torch_overhead
num_gpu_blocks     = floor(kv_pool / kv_bytes_per_block)
max_concurrency    = num_gpu_blocks * block_size / max_model_len
```

* R10.3. `determine_available_memory` runs a simulated profiling forward pass at
  max_num_batched_tokens to establish the activation peak, at the same point upstream does it.
* R10.4. The startup timeline is simulated and observable: weight load time from a configurable load
  bandwidth, profiling run, KV allocation, graph capture. The total is reported in the same log line
  upstream emits.
* R10.5. A non positive kv_pool fails at startup, not at request time, with the actionable message
  shape (lower gpu_memory_utilization or max_model_len).
* R10.6. A max_model_len that cannot fit one request in the KV pool is a startup error.

### R11. Model, sampler, output

* R11.1. `SimModel` produces one token id per sampled position. It never allocates a vocab sized
  array unless logprobs are requested.
* R11.2. Output length policy, seeded per request: fixed, uniform(a, b), lognormal(mu, sigma),
  from_request (respect max_tokens and ignore_eos), from_fixture (prompt hash to output map). This
  knob is what makes workload experiments meaningful.
* R11.3. Content policy: synthetic deterministic pseudo words, echo, or fixture. Content must
  detokenize to stable text so HTTP level golden tests are possible.
* R11.4. The sampler exercises the full parameter surface: temperature, top_p, top_k, min_p,
  repetition and presence and frequency penalties, seed, n, stop token ids, min_tokens, logit_bias,
  logprobs, prompt_logprobs. Effects reach only as far as changing the PRNG draw. Values are
  synthetic but schema correct.
* R11.5. Stop condition parity: EOS, stop token ids, stop strings with incremental detokenization
  and correct truncation, max_tokens, min_tokens suppression.
* R11.6. Incremental detokenization is real, including the partial UTF8 and leading space rules,
  because that is a genuine source of bugs in stream consuming products.
* R11.7. `n > 1` is handled by the frontend parallel sampling layer, as upstream does.

### R12. Metrics, logging, tracing

* R12.1 **[F5, verified against the pin]**. Prometheus metrics with upstream names and label sets.
  The draft's list was wrong in a way that would have broken every dashboard this project exists to
  serve, so it is replaced by the names actually declared in `vllm/v1/metrics/loggers.py` at v0.27.1.

  **Counters are declared without a `_total` suffix.** `prometheus_client` appends `_total` on
  export. Declaring `vllm:prefix_cache_queries_total` therefore exports
  `vllm:prefix_cache_queries_total_total`. Histograms carry no such suffix and are declared as-is,
  which is why `vllm:iteration_tokens_total` genuinely does end in `_total`.

  Gauges: `vllm:num_requests_running`, `vllm:num_requests_waiting`,
  `vllm:num_requests_waiting_by_reason`, `vllm:kv_cache_usage_perc`, `vllm:engine_sleep_state`,
  `vllm:lora_requests_info`, `vllm:cache_config_info`.

  Counters (exported with `_total` appended): `vllm:prefix_cache_queries`,
  `vllm:prefix_cache_hits`, `vllm:external_prefix_cache_queries`,
  `vllm:external_prefix_cache_hits`, `vllm:num_preemptions`, `vllm:prompt_tokens`,
  `vllm:prompt_tokens_cached`, `vllm:prompt_tokens_by_source`, `vllm:generation_tokens`,
  `vllm:request_success`, `vllm:corrupted_requests`, `vllm:mm_cache_queries`,
  `vllm:mm_cache_hits`.

  Histograms: `vllm:iteration_tokens_total`, `vllm:time_to_first_token_seconds`,
  `vllm:inter_token_latency_seconds`, `vllm:request_time_per_output_token_seconds`,
  `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`,
  `vllm:request_inference_time_seconds`, `vllm:request_prefill_time_seconds`,
  `vllm:request_decode_time_seconds`, `vllm:request_prompt_tokens`,
  `vllm:request_generation_tokens`, `vllm:request_max_num_generation_tokens`,
  `vllm:request_prefill_kv_computed_tokens`, `vllm:request_params_n`,
  `vllm:request_params_max_tokens`, `vllm:kv_block_lifetime_seconds`,
  `vllm:kv_block_reuse_gap_seconds`, `vllm:kv_block_idle_before_evict_seconds`.

  Note `vllm:time_per_output_token_seconds` from the draft **no longer exists**; it split into
  `vllm:request_time_per_output_token_seconds` and `vllm:inter_token_latency_seconds`. Speculative
  decoding metrics are declared in `vllm/v1/spec_decode/metrics.py`, not in `loggers.py`.

  A test asserts the exported name set against this list, so a future pin bump that renames a metric
  fails loudly rather than silently emptying a dashboard panel.
* R12.2. Histogram bucket edges match upstream so dashboards built against real vLLM render
  correctly against pretending-vllm.
* R12.3. The periodic stats log line matches the upstream format closely enough that log parsers
  work.
* R12.4. Under a virtual clock all durations are virtual, and that fact is discoverable from a label
  or help string.
* R12.5. KV cache events (block stored, block removed, all cleared) publishable on a ZeroMQ topic in
  the upstream event schema, so external prefix aware routers can be tested against it.

### R13. Parallelism

* R13.1. Tensor parallel shards weights and KV heads, and therefore memory and cost model inputs.
  Simulated as N worker objects with a barrier and a modeled allreduce cost. The requirement is that
  per device memory and step time change correctly, not that data moves.
* R13.2. Pipeline parallel shards layers, introduces microbatching and bubbles in the cost model, and
  requires more than one batch in flight.
* R13.3. Data parallel: multiple engine core replicas with a coordinator, per replica queues, and the
  lockstep rule when expert parallelism is enabled (P3).
* R13.4. Process count reporting matches the upstream formula so that operational docs transfer.

### R14. Speculative decoding (P3)

* R14.1. Proposers: ngram (fully real, pure token manipulation) and draft_model (simulated).
* R14.2. Acceptance modeled as a per position Bernoulli chain with a configurable base rate and
  decay, giving a realistic accepted length distribution.
* R14.3. The scheduler and KV manager handle speculative slots, rejection, and rollback of
  unaccepted positions. This is where block accounting gets hard, which is exactly why it is worth
  having.
* R14.4. The cost model accounts for draft passes and the widened verify batch.

### R15. Structured output (P3)

* R15.1. `StructuredOutputManager` with a grammar compile step and a per step bitmask hook.
  Compilation may be real (a small JSON schema to FSM) or stubbed with a modeled compile latency,
  but the scheduler side interaction is real.

### R16. LoRA (P3)

* R16.1. LoRARequest, adapter registry, per request adapter id, adapter slot capacity, max_loras and
  max_lora_rank affecting memory accounting, and adapter id participating in prefix cache extra keys.

### R17. KV transfer and disaggregation (P3)

* R17.1. Connector interface with scheduler side and worker side halves:
  get_num_new_matched_tokens, update_state_after_alloc, build_connector_meta, start_load_kv,
  wait_for_save.
* R17.2. A SimSharedStoreConnector implementing a fake external KV store with configurable bandwidth
  and latency, so prefill and decode disaggregation can be exercised end to end between two
  pretending-vllm instances.

### R18. Multimodal (P4)

* R18.1. Placeholder based inputs with encoder cache, a separate encoder budget, and encoder cost in
  the cost model.

### R19. Determinism, clock, tracing

* R19.1. Three clock modes: virtual (advances only by modeled durations, no sleeping), real (sleeps
  the modeled duration), scaled (sleeps duration over time_scale). The engine core owns the clock;
  nothing else reads it. Enforced from the first commit.
* R19.2. One global seed reproduces an entire run: arrival times, output lengths, sampled tokens,
  jitter. Per request RNG is derived from (seed, request_id) so request level results are independent
  of arrival interleaving.
* R19.3. Event trace as JSONL, one record per engine step plus one per request lifecycle transition.
  Fields: step index, virtual time, scheduled tokens per request, new and cached block counts,
  preemption events, KV usage, queue depths. Primary artifact for both understanding and conformance.
* R19.4. A trace viewer rendering a step timeline to text or SVG (P2).

### R20. Benchmarks and workloads

* R20.1. bench latency, throughput, and serve mirroring the upstream CLI, including Poisson and
  Gamma arrival processes and a request rate parameter.
* R20.2. Datasets: synthetic length distributions, shared prefix workloads to exercise prefix
  caching, and replay of recorded (arrival_time, prompt_len, output_len) traces.
* R20.3. A sweep runner varying config over a grid and emitting tidy CSV. Running thousands of
  configurations cheaply is the point of the project.

### R21. Testing and conformance

* R21.1. Invariants asserted in debug mode: total blocks equals free plus allocated; no negative
  ref_cnt; no slot written twice in one step; scheduled tokens never exceed the budget; running
  requests never exceed max_num_seqs; KV usage never exceeds one; a preempted request produces the
  same output as it would have without preemption; every admitted request terminates.
* R21.2. Property based tests over random workloads and configs for the scheduler and KV manager.
* R21.3. Conformance suite C1 to C7. While the contract is in `asserted` state (D4), C1 to C4 run as
  regression tests against recorded pretending-vllm traces. Once golden traces from a real vLLM run
  at the pinned version are checked in, the same tests compare against those instead, and the
  contract is promoted to `verified`.
* R21.4. HTTP contract tests against real vLLM where a CPU build is feasible, otherwise against
  recorded responses.
* R21.5. Whole suite under 30 seconds on a laptop with the constant cost model.
* R21.6. A mutation catalogue (`tests/mutations.toml`, run by `tools/mutate.py`). Each entry names a guarantee, the minimal edit that breaks it, and the test that should notice; the tool asserts that test FAILS under the edit. A green suite says the tests pass, not that they would fail if the code were wrong — and on this project the gap has been real: mutation testing has found a non-discriminating test nearly every time it was run. Enforced in CI so it is no longer something someone remembers to do.
* R21.7. An inert-mechanism lint (`tests/unit/test_inert.py`). Every `self.X = ...` in the package is checked against every `.X` read in the package, its tests and its tools; a name written and never read is reported. Defect class two in this project's taxonomy — a mechanism present, commented, sometimes tested, and changing nothing — had produced six findings by hand and four more on this lint's first run, two of them carrying comments claiming they fed metrics that were never built. Exemptions live in an `ALLOWED` map and must state a reason.

## 7. Core data structures

Named to match upstream. Field lists are the minimum.

* `Request` **[F8]**: request_id, prompt_token_ids, sampling_params, pooling_params, client_index,
  arrival_time, status, num_computed_tokens, output_token_ids, block_hashes, num_preemptions,
  lora_request, structured_output_request, cache_salt, priority, mm_features, trace_headers,
  resumable, abort_immediately, and **`block_hasher`** — a callable injected at construction. Block
  hashing is dependency-injected upstream rather than inlined, which R6.3 must mirror or prefix cache
  hashing ends up in the wrong place.
* `RequestStatus` **[F6]**: WAITING, **WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR** (not
  `WAITING_FOR_FSM`), WAITING_FOR_REMOTE_KVS, **WAITING_FOR_STREAMING_REQ**, RUNNING, PREEMPTED,
  FINISHED_STOPPED, FINISHED_LENGTH_CAPPED, FINISHED_ABORTED, FINISHED_IGNORED, **FINISHED_ERROR**,
  **FINISHED_REPETITION**.

  It is an `IntEnum` and **the member order is load-bearing**: `is_finished(s)` is implemented as
  `s > PREEMPTED`, so every finished state must sort after `PREEMPTED` and no non-finished state may.
  Reordering the enum silently breaks finished-request detection everywhere, with no type error and
  no failing assertion until a request never completes. A test pins the ordering.
* `SchedulerOutput` **[F7]**: scheduled_new_reqs, **scheduled_cached_reqs** (a `CachedRequestData`
  object, not a list — this is the incremental diff R7.3 depends on), num_scheduled_tokens,
  total_num_scheduled_tokens, scheduled_spec_decode_tokens, scheduled_encoder_inputs,
  num_common_prefix_blocks, finished_req_ids, **free_encoder_mm_hashes** (not
  `free_encoder_input_ids`), preempted_req_ids, has_structured_output_requests,
  pending_structured_output_tokens, num_invalid_spec_tokens, structured_output_request_ids,
  grammar_bitmask, kv_connector_metadata, ec_connector_metadata, new_block_ids_to_zero,
  kv_cache_block_copies, num_spec_tokens_to_schedule.
* `Scheduler.schedule()` takes `throttle_prefills: bool = False`. `EngineCore.step()` returns
  `tuple[dict[int, EngineCoreOutputs], bool]` — keyed by client index for data parallelism, plus a
  `model_executed` flag.
* `ModelRunnerOutput`: req_ids, req_id_to_index, sampled_token_ids, logprobs, prompt_logprobs_dict,
  spec_token_ids, finished_sending, finished_recving.
* `EngineCoreOutput(s)`: request_id, new_token_ids, finish_reason, stop_reason, events,
  scheduler_stats.
* `KVCacheBlock`: block_id, ref_cnt, block_hash, prev_free_block, next_free_block.
* `KVCacheSpec` and `KVCacheConfig`: block_size, num_blocks, kv_cache_groups, page size in bytes.

## 8. Configuration data

* Model cards, `sim/models/*.json`: num_hidden_layers, hidden_size, num_attention_heads,
  num_key_value_heads, head_dim, intermediate_size, vocab_size, max_position_embeddings,
  num_parameters, tie_word_embeddings, dtype, architecture family, and MoE fields (num_experts,
  num_experts_per_token) so the active parameter count is correct.
* Device cards, `sim/hardware/*.json`: memory_bytes, memory_bandwidth, peak_flops per dtype,
  interconnect bandwidth, launch overhead, default efficiency factors. Ship a datacenter class card,
  a workstation class card, and a deliberately tiny card for forcing preemption in tests.
* Cost model profiles: efficiency factors, overhead constants, provenance notes.

## 9. Milestones

**Reordered per D8.** The draft ordering (P0 offline → P1 depth → P2 product surface) was
fidelity-first, deferring the HTTP server to third. But the consuming projects speak all four
surfaces, and C5 (HTTP schema) is independent of C1–C4 (scheduler fidelity) — so the surface can
stabilize in M1 while depth is added underneath it, rather than depth being built against nothing.

* **M0, foundations** — the things that cannot be retrofitted. Vendored upstream tree plus
  `spec_sync`; clock, RNG, and trace; the platform seam; the purity lint enforcing clock/RNG
  ownership from commit one (D2); CI. *Acceptance: `current_platform` resolves to `SimPlatform`
  through the real plugin mechanism; purity lint and `spec_sync` pass.* **— complete**
* **M1, vertical slice to HTTP.** Config and `EngineArgs`; core types with the corrected
  `RequestStatus` and `SchedulerOutput`; scheduler with token budget and continuous batching (prefix
  caching and chunked prefill present as inert call sites); `BlockPool` and `KVCacheManager` without
  hashing; `UniProcExecutor`, `SimWorker`, `SimModelRunner` mirroring V2's decomposition; `SimDevice`,
  analytic memory ledger, constant cost model, `SimModel`, `MockTokenizer`; `EngineCore`, `LLMEngine`,
  `AsyncLLM`, input/output processors, incremental detokenizer; offline `LLM.generate`; the FastAPI
  server with completions, chat, models, tokenize, health, and SSE streaming with disconnect-abort;
  `/metrics` with the corrected name set; JSONL trace. *Acceptance: a consuming project points at it
  unmodified and streams, cancels, and scrapes metrics. 100 mixed-length requests drain, invariants
  hold, the trace is readable.*
* **M2, fidelity depth** — makes C1–C4 mean something. Prefix caching with hashing and eviction,
  chunked prefill, preemption by recompute, memory model with profiling and derived `num_gpu_blocks`,
  roofline cost model, slot-mapping validation as a hard assert, full metrics, invariant and property
  tests. *Acceptance: C2, C3, C4 stable in regression mode; a shared prefix workload shows the
  expected hit rate.*
* **M3, transparency and conformance.** HTTP introspection endpoints, trace viewer, conformance suite
  C1–C7, `capture_golden_trace.py`, multiprocess engine core over ZeroMQ, real and scaled clocks,
  benchmarks CLI and sweep runner. *Acceptance: C5, C6, C7 pass; dashboards built against real vLLM
  render against this.*
* **M4, the depth.** Tensor and pipeline parallel, data parallel replicas, speculative decoding,
  LoRA, structured output, KV connector and disaggregation, hybrid KV groups, multimodal encoder
  cache, sleep mode.

## 10. Non functional requirements

* NF1 **[F10]**. Python 3.11 or newer. Dependencies: msgspec, pyzmq, fastapi, uvicorn, pydantic,
  prometheus_client, **numpy (required, not optional)**. Upstream's V2 input preparation computes
  `query_start_loc_np`, `idx_mapping_np`, `prefill_len_np`, and `is_prefilling_np` in numpy and only
  mirrors the result to device; that numpy half *is* the real logic this project keeps, so numpy is
  load-bearing rather than an optimization. No torch, no CUDA, no transformers at import time —
  enforced by `tests/unit/test_purity.py`. Optional extra `realtok` pulls in tokenizers.
* NF2. Cold start from install to first token under 2 seconds with the constant cost model.
* NF3. Runs on macOS, Linux, and Windows, because there is no device to detect.
* NF4. Fully type annotated, checked in CI.
* NF5. Every module header names its upstream counterpart path, so the repo doubles as an index into
  the real codebase.
* NF6. Docs mirror the upstream design docs structure, plus one page per simulated subsystem stating
  precisely what is fake and why.

## 11. Risks

* The cost model becomes trusted beyond its accuracy. Mitigated by R9.4 calibration, R9.5 labeling,
  and a rule that no latency metric ships without a modeled tag available.
* Upstream drifts and the port rots. Mitigated by the pin in `UPSTREAM.md` and in the golden traces,
  and by treating a version bump as a deliberate reviewed task.
* ~~Model Runner V2 lands as the upstream default and this port mirrors the previous execution
  core.~~ **Retired by D6 [F1]** — this already happened before the project started, and the port
  targets V2 from the first commit. The residual risk inverts: V2 is the newer code and will keep
  moving, so Tier B churn in `v1/worker/gpu/` should be expected at each pin bump. The boundary stays
  at `execute_model`, which both runner designs preserve, so the blast radius is
  `sim_model_runner.py` and `sim_input_batch.py`.
* The port rots as upstream drifts, and nobody notices. This is not hypothetical: draft v1 of this
  document had eight stale assumptions within months. Mitigated by `tools/spec_sync.py`, which
  resolves every module's declared upstream counterpart against the vendored tree in CI, so a rename
  or removal is a build failure rather than a discovery years later.
* Scope creep into a real reimplementation. Mitigated by the LOC budget and the milestone gates.
  Anything below the boundary growing past a few hundred lines is a smell.
* The fidelity contract is never verified. Mitigated by shipping `tools/capture_golden_trace.py` from
  P1 and by the README stating `asserted` until traces exist. Without them, section 3 is a design
  intent, not a guarantee.
