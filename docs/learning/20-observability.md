# 20. Observability

> **Files:** [`pvllm/v1/metrics/loggers.py`](../../pvllm/v1/metrics/loggers.py), [`metrics/stats.py`](../../pvllm/v1/metrics/stats.py), [`pvllm/tracing.py`](../../pvllm/tracing.py), [`pvllm/sim/trace.py`](../../pvllm/sim/trace.py), [`pvllm/trace_viewer.py`](../../pvllm/trace_viewer.py), [`entrypoints/serve/dev/`](../../pvllm/entrypoints/serve/dev)
> **Upstream:** `vllm/v1/metrics/loggers.py` (Tier B); the trace, the viewer, and the introspector have **no upstream counterpart**
> **Prerequisites:** chapter [19](19-openai-server.md).
> **Contract:** C6 — metric names, types, labels, and histogram bucket edges are **exact**.

Four ways to find out what the engine is doing. There is deliberately **no verbose stdout
narration to grep** — everything is structured.

| Surface | Answers | When |
|---|---|---|
| `/metrics` | aggregate rates and distributions | production-shaped monitoring |
| a JSONL trace | every decision, exactly | after the fact, or in CI |
| `pvllm trace view` | *why did this run behave like that* | when you do not know what to query for |
| `/debug/*` | live scheduler, blocks, cache, cost model | while a product is driving the engine |

## `/metrics` — Prometheus, and the suffix trap

38 metrics, with upstream's names, types, labels, and bucket edges. A dashboard built against
real vLLM renders against this without modification. That is conformance class C6.

**The trap that the requirements draft got wrong** (delta F5 in
[UPSTREAM.md](../../UPSTREAM.md)), and that is worth internalising if you ever write Prometheus
code:

> Upstream declares counters **without** a `_total` suffix. `prometheus_client` appends it on
> export. Declaring `vllm:prompt_tokens_total` therefore exports
> `vllm:prompt_tokens_total_total`, and every counter panel on the dashboard goes empty.

Histograms carry no auto-suffix, which is why `vllm:iteration_tokens_total` genuinely does end in
`_total` — an inconsistency that looks like a typo and is not.

### What is exported

**Gauges** — instantaneous:

```
vllm:num_requests_running          vllm:num_requests_waiting
vllm:kv_cache_usage_perc
```

**Counters** (exported with `_total`):

```
vllm:num_preemptions               vllm:prompt_tokens           vllm:generation_tokens
vllm:prefix_cache_queries          vllm:prefix_cache_hits
vllm:external_prefix_cache_queries vllm:external_prefix_cache_hits
vllm:mm_cache_queries              vllm:mm_cache_hits
vllm:spec_decode_num_draft_tokens  vllm:spec_decode_num_accepted_tokens
vllm:request_success               (labelled by finished_reason)
```

**Histograms**:

```
vllm:iteration_tokens_total
vllm:time_to_first_token_seconds        vllm:inter_token_latency_seconds
vllm:request_time_per_output_token_seconds
vllm:e2e_request_latency_seconds        vllm:request_queue_time_seconds
vllm:request_inference_time_seconds     vllm:request_prefill_time_seconds
vllm:request_decode_time_seconds
vllm:request_prompt_tokens              vllm:request_generation_tokens
vllm:request_params_max_tokens          vllm:request_params_n
```

Note two upstream renames the draft spec had wrong: `vllm:time_per_output_token_seconds` **no
longer exists** — it is `vllm:request_time_per_output_token_seconds` plus
`vllm:inter_token_latency_seconds`.

Every latency metric's help text carries the label that matters here:

```
NOTE: this duration is MODELED by pretending-vllm's cost model, not measured.
See the fidelity contract in the README.
```

A consumer reading only `/metrics` must still be able to tell these were not measured.

### The four metrics a capacity question actually uses

- `vllm:num_requests_waiting` — if this is persistently non-zero, you are capacity-bound.
- `vllm:kv_cache_usage_perc` — approaching 1.0 means preemption is imminent.
- `vllm:num_preemptions_total` — non-zero means you are already over.
- `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` — in **tokens**, so the ratio
  is directly "what fraction of prefill did we avoid".

### One known gap

`vllm:prompt_tokens_total` currently exports **0** on any workload:
`IterationStats.num_prompt_tokens` is declared and read by the logger but never incremented in the
output processor, so the counter never moves. `vllm:iteration_tokens_total` is affected by the same
root cause (it sums prompt + generation tokens per step) and undercounts by the prompt half.

The per-request histogram is fine — `vllm:request_prompt_tokens_sum` reports real values, because
it is fed from `FinishedRequestStats` on a different path. If you need prefill token throughput
today, derive it from the histogram or from a trace. This is a real divergence from upstream's
behaviour, not a modelling choice.

```bash
curl -s localhost:8000/metrics | grep -E 'prompt_tokens'
```

```
vllm:prompt_tokens_total{engine="0",model_name="dense-0.6b"} 0.0        ← the gap
vllm:request_prompt_tokens_count{engine="0",model_name="dense-0.6b"} 1.0
vllm:request_prompt_tokens_sum{engine="0",model_name="dense-0.6b"} 12.0  ← correct
```

### Scrape semantics

Histogram observations are accumulated by the frontend and **taken and cleared** by `/metrics`, so
an observation is recorded exactly once however often you scrape. Counters are set from cumulative
totals rather than incremented, so a scrape cannot double count.

## The JSONL trace

```bash
pvllm serve --model dense-0.6b --trace-path run.jsonl
# or PVLLM_TRACE_PATH=run.jsonl, or LLM(..., trace_path="run.jsonl")
```

One record per engine step, plus one per request lifecycle transition. The header names the run:

```json
{"v":1,"seq":0,"type":"header","schema_version":1,"upstream_version":"0.27.1","seed":0,
 "clock_mode":"virtual","config":{"model":"dense-0.6b","model_card":"dense-0.6b",
 "device_card":"datacenter-80gb","block_size":16,"max_model_len":512,"cost_model":"constant"}}
```

A request record:

```json
{"v":1,"seq":1,"type":"request","t":1767225600.08293,"request_id":"0",
 "event":"arrived","num_prompt_tokens":193,"max_tokens":40}
```

A step record — this is the interesting one:

```json
{"v":1,"seq":9,"type":"step","t":1767225600.09155,
 "new_reqs":[],"cached_reqs":["0","1","2"],"resumed_reqs":[],
 "num_scheduled_tokens":{"0":1,"1":1,"2":1},"total_num_scheduled_tokens":3,
 "finished_req_ids":[],"preempted_req_ids":[],"num_common_prefix_blocks":[1],
 "step":3,"num_running":3,"num_waiting":3,"kv_usage":0.925,
 "num_preemptions_total":0,"prefix_cache_hits":80,"prefix_cache_queries":1158,
 "waiting_req_ids":["3","4","5"]}
```

Design properties, each with a reason:

- **`seq` is gap-free.** "A conformance diff that sees a gap knows records were dropped rather than
  that behavior changed."
- **Deterministic encoding.** No wall-clock timestamps, no set iteration order, no dict ordering
  that depends on insertion history — sets are *sorted* before serialization. Given the same seed
  and config, two runs produce byte-identical traces.
- **`waiting_req_ids` carries ids, not just a count.** "A request starving behind a long prefill is
  the thing a timeline is opened to find, and a count cannot show *which* request waited."
- **Tracing off is a real object, not a `None` check.** `NullTraceWriter` discards everything, so
  turning tracing off cannot change control flow.

Ordinary tools work:

```bash
grep '"event":"finished"' run.jsonl | wc -l
python -c "
from pvllm.tracing import read_trace
steps = [r for r in read_trace('run.jsonl') if r['type'] == 'step']
print('steps:', len(steps))
print('peak kv:', max(s['kv_usage'] for s in steps))
print('max waiting:', max(s['num_waiting'] for s in steps))
"
```

## `pvllm trace view` — the timeline

```bash
pvllm trace view run.jsonl              # text
pvllm trace view run.jsonl --format svg -o run.svg
pvllm trace view run.jsonl --width 60   # narrower: buckets several steps per column
```

```
pretending-vllm trace  (upstream 0.27.1, seed 0, clock virtual)
  model='dense-0.6b' card='dense-0.6b' device='datacenter-80gb' block_size=16 cost_model='constant'

  0  #===================                              length
  1  #===================                              length
  2  #===============!...#===                          length
  3  ....................#===================          length
  4  ....................#===================          length
  5  ........................#===========!...#=======  length

  steps=96  tokens=1368  preemptions=2  peak_kv=100.0%
  prefix cache: 3792/12414 tokens (30.5%)
  (each column is 2 steps)
  legend: # prefill  : small prefill  = decode  . waiting  ! preempted  ^ resumed
```

> A JSONL trace answers any question you can write a query for. A timeline answers the one you did
> not know to ask: *why did this run behave like that.*

Preemption thrash, a request starved behind a long prefill, a prefix cache that stopped hitting —
all obvious in a picture and tedious to find by grepping.

Each row is a request, each column a step, and each cell is **the scheduling decision**. It is
deliberately **not** a Gantt chart over wall time: under a virtual clock the steps are what matter
and their durations are modeled, so "laying rows out by duration would make the modeled cost model
look like the measured subject of the picture."

When several steps share a column, the *most significant* glyph wins — a preemption is never hidden
by the decodes around it.

## `/debug/*` — live introspection

```bash
pvllm serve --model dense-0.6b --enable-debug-endpoints
```

| Endpoint | Answers |
|---|---|
| `GET /debug/scheduler` | what is running, what is waiting, in what order |
| `GET /debug/requests` | every tracked request, counted by state |
| `GET /debug/requests/{id}` | one request's state machine and block table |
| `GET /debug/blocks` | the block pool, and which requests hold which blocks |
| `GET /debug/prefix_cache` | hit rate overall, and per live request |
| `GET /debug/cost_model` | the term-by-term breakdown of recent steps |
| `GET /debug/config` | the fully resolved config, **including what was derived** |

```bash
curl -s localhost:8000/debug/prefix_cache | python -m json.tool
```

```json
{"enabled": true, "hash_algorithm": "sha256",
 "prefix_cache_queries": 21, "prefix_cache_hits": 0, "prefix_cache_hit_rate": 0.0,
 "prefix_cache_evictions": 0, "prefix_cache_cached_blocks": 1, "by_request": []}
```

```bash
curl -s localhost:8000/debug/cost_model | python -m json.tool
```

```json
{"cost_model": "constant", "provenance": "modeled", "history_size": 64, "num_steps": 13,
 "steps": [{"step": 0, "num_tokens": 8192, "num_reqs": 1, "max_seq_len": 8192,
            "is_graph_hit": false, "duration": 0.08293, "compute_s": 0.08293,
            "memory_s": 0.0, "comm_s": 0.0, "encoder_s": 0.0, "overhead_s": 0.0,
            "jitter": 1.0, "flops": 0.0, "bytes": 0.0, "bound_by": "compute",
            "provenance": "modeled"}, ...]}
```

(That first step at 8,192 tokens is the startup profiling pass — chapter
[14](14-memory-model.md).)

Three properties of this surface:

- **Read-only.** Every method returns a dict and mutates nothing, so no engine behaviour can depend
  on what it reports.
- **Off by default**, because it exposes prompt token ids. The gate mirrors upstream's
  `VLLM_SERVER_DEV_MODE`.
- **Everything reported is real** — the block map is the actual block pool, the request states are
  the actual scheduler queues. The one exception is labelled: the cost-model breakdown is modeled,
  and says so in its own payload.

The introspector is also the single deliberate exception to the no-simulator-awareness rule
(chapter [03](03-simulation-boundary.md)): it reaches through the worker into the simulator, because
a cost-model breakdown *is* simulator state and showing it is what the transparency goal asked for.
Upstream has no counterpart, because on real hardware nobody can.

## Try it: find out why a run was slow

The workflow this chapter exists to enable:

```bash
# 1. run it, with a trace
pvllm bench serve --model dense-8b --device-card datacenter-80gb \
  --cost-model-profile roofline --request-rate 8 --num-prompts 40 \
  --trace-path serve.jsonl

# 2. look at the shape
pvllm trace view serve.jsonl

# 3. if TTFT is bad: was it queueing or prefill?  bench serve splits them.
# 4. if it was queueing: was the pool full?  ->  peak_kv in the summary
# 5. if the pool was full: was the prefix cache helping?  ->  hit rate in the summary
# 6. if you need per-step detail:  jq over the JSONL, or /debug/cost_model live
```

Steps 3 to 5 are the actual diagnostic tree for LLM serving, and the trace summary line answers all
three:

```
steps=96  tokens=1368  preemptions=2  peak_kv=100.0%
prefix cache: 3792/12414 tokens (30.5%)
```

## Check yourself

- Why is `vllm:prompt_tokens` declared without a `_total` suffix, and what happens if you add one?
- Which metric is currently broken, and which surface still reports the same quantity correctly?
- Why does a step record sort its sets before serialization?
- Why is the timeline not laid out over wall-clock time?
- Why are the `/debug/*` endpoints off by default?
- `vllm:num_requests_waiting` is persistently 12 and `kv_cache_usage_perc` is 0.4. What is your
  binding constraint likely to be?

## Next

[21. Structured output](21-structured-output.md) — the first of the advanced-feature chapters.
