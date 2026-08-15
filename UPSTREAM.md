# Upstream pin

pretending-vllm is a structurally faithful reimplementation of the vLLM V1 engine. Module paths,
class names, and method signatures mirror upstream closely enough that a diff against the real repo
is a study exercise (G2). That property is only true against a *specific* upstream version.

| | |
|---|---|
| **Upstream** | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **Pinned version** | **v0.27.1** |
| **Released** | 2026-08-11 |
| **Vendored at** | `vendor/vllm-0.27.1/` (gitignored) |
| **Integrity** | `vendor/MANIFEST.sha256` (committed, 2736 files) |

Bumping the pin is a deliberate, reviewed task -- never a drive-by. See *Bumping the pin* below.

## Getting the reference tree

```bash
python tools/fetch_upstream.py
```

Stdlib only, so it runs before `pip install -e .`. It downloads the GitHub source tarball for the
pinned tag and extracts the text sources under `vllm/` (~36 MB, 2160 `.py` files). Kernels and test
data are skipped -- a port with no device code has no use for them.

The tree is **not committed**; `vendor/MANIFEST.sha256` is. To verify a vendored tree is
byte-identical to the one this port was written against:

```bash
python tools/fetch_upstream.py --check
```

CI runs `--check`, then `tools/spec_sync.py`.

## How the mirror is kept honest

Every pvllm module carries a header naming its upstream counterpart and its fidelity tier:

```python
"""...

Upstream: vllm/v1/core/sched/scheduler.py
Tier: A
"""
```

`tools/spec_sync.py` resolves every such header against `vendor/` and fails when a counterpart has
been renamed, moved, or deleted. This is the durable mitigation for the "port rots" risk in §11 of
the requirements: upstream drift becomes a failing CI check instead of a discovery years later.

### Fidelity tiers

| Tier | Meaning | Rule |
|---|---|---|
| **A** | Line-for-line | Same method names, same order of operations, same branch structure. Binds the C1–C4 contract. A behavioral divergence is a bug by definition. |
| **B** | Signature-faithful, body-thinned | Same public API and observable behavior. Internals may drop unsupported paths. |
| **C** | Shape-only | Field names, types, and validation *intent* match. Implementation is ours. |
| **D** | Invented | No upstream counterpart. The only place allowed randomness, wall-clock, or invented numbers (B3). |

A module with **no upstream counterpart** declares `Upstream: (none -- simulator)` and Tier D, or
`Upstream: (none -- pvllm addition)` at whatever tier fits. The second form exists for pvllm-only
*interfaces* that sit above the simulation boundary — the trace sink protocol, for instance. Those
are neither ports nor simulator internals, and mistiering them into D just to satisfy `spec_sync`
would wrongly mark them as places allowed to invent numbers.

**Unsupported-path discipline:** a dropped upstream code path raises `NotImplementedError` naming the
upstream feature. It never silently no-ops. For a test double, failing loudly beats behaving subtly
wrongly.

## Delta from the requirements draft

`pretending_vllm_requirements.md` (draft v1) was written before the tree at this pin was available,
and several load-bearing assumptions were stale. These were verified against `vendor/vllm-0.27.1/`
and the spec has been amended. Recorded here so the corrections are traceable.

| # | Draft assumed | v0.27.1 actually | Consequence |
|---|---|---|---|
| **F1** | Model Runner V2 is a *future* migration risk (D5, §11); mirror the classic runner | **V2 has landed and is the default.** `VllmConfig.use_v2_model_runner` returns True for any dense, non-MoE, non-hybrid generate model. V2 lives at `vllm/v1/worker/gpu/`; `model_runner.py` is 1,723 lines vs the legacy `gpu_model_runner.py` at **7,928** | D5 superseded by **D6**: mirror V2. Correct *and* 4.6× less work |
| **F2** | `transformers_utils/tokenizer.py`, `transformers_utils/tokenizers/mock.py` | New top-level **`vllm/tokenizers/`** package with `protocol.py` + `registry.py` | `MockTokenizer` implements `tokenizers/protocol.py`; cleaner seam than planned |
| **F3** | `v1/engine/processor.py` | **`v1/engine/input_processor.py`** | Path rename |
| **F4** | `entrypoints/cli/bench.py`, `benchmarks/datasets.py`, invent `benchmarks/sweep.py` | **`entrypoints/cli/benchmark/{main,latency,serve,throughput,sweep,startup}.py`**, **`benchmarks/datasets/datasets.py`**, and upstream already ships a **`benchmarks/sweep/`** package | Mirror the real layout; nothing to invent |
| **F5** | R12.1 metric names carry `_total` suffixes | Upstream *declares* `vllm:prefix_cache_queries`, `vllm:num_preemptions`, `vllm:prompt_tokens`, `vllm:generation_tokens`, `vllm:request_success` with **no suffix** -- `prometheus_client` appends `_total` on export. `vllm:time_per_output_token_seconds` no longer exists; it is `vllm:request_time_per_output_token_seconds` + `vllm:inter_token_latency_seconds`. 38 metrics total, including `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_reuse_gap_seconds`, `vllm:num_requests_waiting_by_reason`, `vllm:prompt_tokens_cached`, `vllm:external_prefix_cache_*` | Using the draft's names would double-suffix every counter and break the dashboards this project exists to serve |
| **F6** | `RequestStatus` includes `WAITING_FOR_FSM` | `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`, plus `WAITING_FOR_STREAMING_REQ`, `FINISHED_ERROR`, `FINISHED_REPETITION`. It is an **`IntEnum` whose member ordering is load-bearing**: `is_finished(s) == s > PREEMPTED` | Get the ordering wrong and finished-request detection breaks silently |
| **F7** | `SchedulerOutput` field list (§7) | `free_encoder_input_ids` → **`free_encoder_mm_hashes`**; `scheduled_cached_reqs` is a **`CachedRequestData` object**, not a list; ~11 further fields (`scheduled_spec_decode_tokens`, `preempted_req_ids`, `num_invalid_spec_tokens`, `new_block_ids_to_zero`, `kv_cache_block_copies`, …) | §7 amended |
| **F8** | `Request` field list (§7) | Also takes `pooling_params`, `client_index`, `mm_features`, `trace_headers`, **`block_hasher`** (hashing injected as a callable), `resumable`, `abort_immediately` | Block hashing is injected, not inlined -- affects R6.3 |
| **F9** | §5 budget: package under 6,000 lines | The mirrored upstream subset is **~148,000 lines**. `scheduler.py` alone is **2,915** -- larger than the draft's entire "engine + scheduler + KV manager under 2,500" budget | Global cap replaced by the per-subsystem tier table. G1 is preserved as: the *read path* for one request stays under 6,000 lines |
| **F10** | NF1: numpy optional | **numpy is required.** V2's real logic *is* the numpy path (`query_start_loc_np`, `idx_mapping_np`, `is_prefilling_np`); torch only mirrors it to device | Good news: `SimModelRunner` keeps upstream's numpy half near-verbatim and drops the device copies |
| **F11** | B2: the seam is the out-of-tree platform plugin mechanism | **Confirmed exactly right.** `PlatformEnum.OOT` exists, OOT entry-point plugins beat builtins, and `CudaPlatform.check_and_update_config` sets `parallel_config.worker_cls` from `"auto"`. The `entrypoints/` layout in §5 is also correct | No change needed |

## Bumping the pin

v0.27.2rc0 was already tagged when this pin was chosen. Upstream moves fast; the pin is what keeps
the port coherent.

1. Update `UPSTREAM_VERSION` in `tools/fetch_upstream.py`.
2. `python tools/fetch_upstream.py --force --write-manifest`
3. `python tools/spec_sync.py` -- triages every module whose counterpart moved or vanished.
4. Diff the Tier A files against their counterparts and port the behavioral changes. Tier A is where
   fidelity is contractual; a silent divergence here breaks C1–C4.
5. `python tools/capture_golden_trace.py --check` — every conformance workload whose behavior moved
   fails here, naming the class and the step. **Triage each one before re-recording.** A behavioral
   change is expected after a pin bump; the point of the check is that you decide which changes were
   the port and which were bugs, workload by workload.
6. `python tools/capture_golden_trace.py --force` and
   `python tools/capture_golden_trace.py --metrics --force` once each difference is accounted for.
   The goldens embed the pin, so `test_goldens_declare_their_source` fails until they are re-recorded
   — the suite will not let a stale golden pass under a new pin.
7. Update this file's delta table.
