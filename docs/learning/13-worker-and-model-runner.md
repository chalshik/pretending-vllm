# 13. Worker and model runner

> **Files:** [`pvllm/v1/executor/abstract.py`](../../pvllm/v1/executor/abstract.py), [`executor/uniproc_executor.py`](../../pvllm/v1/executor/uniproc_executor.py), [`pvllm/v1/worker/sim_worker.py`](../../pvllm/v1/worker/sim_worker.py), [`worker/gpu/model_runner.py`](../../pvllm/v1/worker/gpu/model_runner.py), [`worker/gpu/input_batch.py`](../../pvllm/v1/worker/gpu/input_batch.py), [`worker/gpu/block_table.py`](../../pvllm/v1/worker/gpu/block_table.py), [`worker/gpu/states.py`](../../pvllm/v1/worker/gpu/states.py), [`worker/gpu/attn_utils.py`](../../pvllm/v1/worker/gpu/attn_utils.py)
> **Upstream:** `vllm/v1/executor/*`, `vllm/v1/worker/gpu_worker.py`, `vllm/v1/worker/gpu/*` (Tier B) — the **V2** runner, upstream's default at the pin
> **Prerequisites:** chapter [12](12-scheduler.md).

The scheduler produced a `SchedulerOutput`. This chapter is everything that happens to it —
and it is where the boundary actually is, three layers down.

## Three layers, each hiding one thing

| Layer | Hides |
|---|---|
| `Executor` | how many workers there are and where they run |
| `Worker` | one device's lifecycle |
| `ModelRunner` | the persistent batch and the input preparation |

Only the last line of the last layer is fake.

### `Executor`

```python
def determine_available_memory(self) -> list[int]
def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]
def initialize_from_config(self, kv_cache_configs) -> None
def compile_or_warm_up_model(self) -> None
def execute_model(self, scheduler_output) -> ModelRunnerOutput
async def execute_model_async(self, scheduler_output) -> ModelRunnerOutput
def collective_rpc(self, method, args=(), kwargs=None) -> list[Any]
```

One implementation, `UniProcExecutor`. Asking for another **names what is missing**:

```
NotImplementedError: distributed_executor_backend='mp' is not implemented. The
multi-worker executors (tensor and pipeline parallelism) ... only the in-process
executor exists. Note that the multiprocess engine core is a separate axis and does
exist -- see PVLLM_ENABLE_V1_MULTIPROCESSING.
```

Note `execute_model_async` is **abstract**, not a default that calls `execute_model`. A
default would let an executor silently block the event loop under a real clock — "the exact
failure this exists to prevent, and it would look correct in every virtual-clock test."

`execute_dummy_batch` is concrete but *refusing*, for a related reason: only an executor that
models a device can charge for a dummy pass, and returning a plausible zero would report an
idle expert-parallel replica as free — which is the whole thing the dummy step exists to
price (chapter [24](24-parallelism.md)).

### `Worker` — the lifecycle upstream defines

[`sim_worker.py`](../../pvllm/v1/worker/sim_worker.py) mirrors `GPUWorker`'s lifecycle
exactly, and the order is upstream's:

```python
init_device()                   # build the SimDevice and its cost model; validate TP/PP/EP
load_model()                    # "load" weights, spend the modeled time, build SimModel
determine_available_memory()    # run a full-budget profiling step, then the memory model
initialize_cache(kv_config)     # claim the KV pool on the ledger; build block tables
compile_or_warm_up_model()      # simulate graph capture
execute_model(scheduler_output) # the boundary
```

Three things worth noticing:

**The clock is passed in, never created.**

```python
if clock is None:
    raise ValueError("Worker requires the engine core's clock; it must not create one")
```

**Configuration validation that needs the model card lives here.** `tensor_parallel_size`
must divide the attention and KV head counts — sharding 16 heads over 3 ranks drops a head,
and "a capacity answer for a configuration that cannot start is worse than no answer". Same
for expert parallelism on a dense model, and `pipeline_parallel_size` above the layer count.
These checks are in the worker rather than in `ParallelConfig` because only the model card
knows the head counts.

**`determine_available_memory` runs a real profiling pass first.** Upstream runs a profiling
forward pass at `max_num_batched_tokens` and reads the allocator's high-water mark; here the
same step is executed through the cost model so its modeled duration lands on the startup
timeline, and then the analytic memory model runs. Chapter [14](14-memory-model.md).

## The V2 model runner

[`worker/gpu/model_runner.py`](../../pvllm/v1/worker/gpu/model_runner.py) mirrors upstream's
**V2** runner. Its method decomposition is kept because *it is the interface*:

```python
add_requests(scheduler_output)      # requests the worker has never seen
update_requests(scheduler_output)   # incremental patch for requests it has
finish_requests(scheduler_output)   # requests that ended
free_states(scheduler_output)       # drop their slots
prepare_inputs(scheduler_output)    # → InputBatch  (the numpy half)
prepare_attn(input_batch, ...)      # → attention metadata, per KV group
execute_model(scheduler_output)     # → ModelRunnerOutput
sample_tokens(input_batch, ...)     # the fake part
```

A monolithic `execute_model` would work and would make a diff against upstream useless.

### The order inside `execute_model`

```python
def execute_model(self, scheduler_output) -> ModelRunnerOutput:
    plan = self._plan_step(scheduler_output)      # everything before the forward pass
    if plan is None:
        return ModelRunnerOutput.make_empty()
    input_batch, profile = plan
    return self._finish_step(input_batch, self.device.execute(profile), scheduler_output)
```

and `_plan_step`, in order:

```
1. finish_requests / free_states   — requests that left, before slots are needed
2. add_requests / update_requests  — the persistent batch diff
3. prepare_inputs                  — flatten to batch-ordered arrays
4. prepare_attn                    — attention metadata + slot mapping validation
5. decide graph-hit                — R8.4
6. count encoder embeddings        — chapter 23
7. build the StepProfile           — the cost model's input
```

**Everything through step 7 is real.** Only `self.device.execute(profile)` (ask the cost
model, advance the clock) and `sample_tokens` (draw ids) are not.

### The persistent batch (R7.3)

The worker keeps per-request state across steps in
[`states.py`](../../pvllm/v1/worker/gpu/states.py): token ids, computed counts, block table
rows, sampling metadata. Each step it applies an **incremental diff** rather than rebuilding
— which is exactly why `SchedulerOutput.scheduled_cached_reqs` is parallel arrays and why
only unseen tokens travel on the wire.

Getting the diff wrong is a class of bug that does not announce itself: a stale
`num_computed_tokens` in the worker means wrong `positions`, which means the next step reads
KV from the wrong slots.

### `prepare_inputs`: the numpy half, near-verbatim

This is the part of upstream that people are usually surprised to learn is *not* torch. From
[UPSTREAM.md](../../UPSTREAM.md)'s delta table (F10):

> numpy is **required**. V2's real logic *is* the numpy path (`query_start_loc_np`,
> `idx_mapping_np`, `is_prefilling_np`); torch only mirrors it to device.

So this port keeps upstream's numpy half almost line for line and drops the device copies:

```python
num_scheduled_tokens = np.fromiter(...)                 # per request, in batch order
query_start_loc_np   = np.zeros(num_reqs + 1)
np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
num_tokens           = int(query_start_loc_np[-1])
seq_lens_np          = num_computed + num_scheduled_tokens
positions[start:end] = np.arange(first_position, first_position + (end - start))
is_prefilling_np     = (num_computed_prefill_tokens + num_scheduled_tokens) < prefill_len
logits_indices       = (query_start_loc_np[1:] - 1)[~is_prefilling_np]
```

Two lines carry most of the meaning:

- **`query_start_loc`** is the flattened batch's index: request *i*'s tokens are
  `[query_start_loc[i], query_start_loc[i+1])`. Every attention kernel upstream takes this
  layout.
- **`logits_indices`** is *which positions sample a token* — the last position of each
  request that has finished prefilling. A request mid-prefill contributes tokens to the batch
  and **no logits**, which is why a chunked prefill produces no output token until its final
  chunk.

The batch is also **reordered** (`sort_batch_req_ids`), so decode-length requests group
together. Which is why structured-output bitmask rows are keyed by request id in sorted order
rather than by batch position — a row index derived from the scheduler's ordering would
address the wrong request's row about half the time.

### Slot mapping: the correctness oracle

[`block_table.py`](../../pvllm/v1/worker/gpu/block_table.py). A **slot** is where one token's
KV is written:

```
slot = block_id * block_size + offset_within_block
```

Upstream computes this in a Triton kernel. The arithmetic here is identical; only the
execution differs. And then this port does something upstream cannot:

> Validating that every written slot lies inside a block the request actually owns turns the
> simulator into a **correctness oracle**: a KV manager bug that would silently corrupt
> another request's cache on real hardware — and surface later as garbled output nobody can
> trace — raises here, at the step that caused it.

That validation runs when `PVLLM_DEBUG_INVARIANTS` is set, which the whole test suite does.
It is the single highest-value thing the simulator adds, and it is why a KV bug in this
repository is a failing test rather than a mystery.

`prepare_attn` builds metadata for **every** KV cache group, not just group 0 — because "the
oracle is only an oracle if it runs over all of them", and an off-by-one in block accounting
lands precisely in a hybrid model's windowed groups. Group 0's metadata is what the cost
model reads, since the step's shape is the same for every group.

One exception, documented in the source: a *windowed* group's block-table row keeps ids the
manager has already handed back (upstream never reads those positions), so scanning them for
ownership is not a valid check.

### Graph capture (R8.4)

```python
DEFAULT_CAPTURE_SIZES = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)

graph_hit = (input_batch.num_reqs in self.captured_sizes
             and not attn_metadata.is_mixed_batch
             and attn_metadata.num_prefills == 0)
```

A captured CUDA graph applies only to a uniform-decode batch of a captured size; a mixed
batch has a shape that was never captured. A hit pays a lower launch cost in the cost model
(8 launches per step instead of 12 per layer). `--enforce-eager` turns capture off entirely.
Chapter [15](15-cost-model.md).

## The fake part, in full

```python
def sample_tokens(self, input_batch, scheduler_output=None) -> ModelRunnerOutput:
    sampling_indices = np.flatnonzero(~input_batch.is_prefilling_np)
    for batch_idx in sampling_indices:
        ...
        num_drafts = len(scheduled_drafts.get(req_id, ()))
        accepted   = self.sim_model.accepted_draft_count(req_id, num_drafts)
        tokens     = [self.sim_model.sample_token(req_id, position + offset, max_tokens)
                      for offset in range(accepted + 1)]
        sampled[batch_idx] = tokens
        drafts[batch_idx]  = self.sim_model.propose_drafts(...)
```

Note what is still real even here: **which** positions sample (from `is_prefilling_np`), how
many tokens a verified step yields (`accepted + 1`), and the batch-index bookkeeping. What is
fake is the id each call returns.

`SimModel` ([`pvllm/sim/model.py`](../../pvllm/sim/model.py)) decides:

- **how long a request runs** — `planned_output_length`, decided once on first use so it
  cannot drift mid-generation, from the `output_length_policy` (`from_request`, `fixed`,
  `uniform`, `lognormal`);
- **which ids** — `sample_token(request_id, position, max_tokens)`, derived from
  `(seed, request_id, position)` so it is *idempotent*: the same position always gives the
  same token, however many times it is asked and whatever was asked before it (chapter
  [16](16-clock-and-determinism.md));
- **draft acceptance** — drawn per position, stopping at the first rejection (chapter
  [25](25-speculative-decoding.md));
- **embeddings** — `embed(prompt_token_ids, dimensions)` (chapter
  [27](27-pooling-and-embeddings.md));
- **constrained output** — the exact token sequence a grammar-constrained request will emit
  (chapter [21](21-structured-output.md)).

`R11.1` is worth calling out: **no vocab-sized array is allocated unless logprobs are
requested.** On a 128k vocabulary at batch 256 that array would be 128 MiB per step, which
would make the simulator slower than the thing it simulates.

## Try it

Drive the runner directly and look at the pieces:

```python
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams

llm = LLM(model="dense-0.6b", max_model_len=512)
llm.generate(["a somewhat longer prompt " * 8, "short"], SamplingParams(max_tokens=3))

runner = llm.llm_engine.engine_core.engine_core.executor.driver_worker.model_runner
meta = runner.last_attn_metadata
print("num_prefills   :", meta.num_prefills)
print("is_mixed_batch :", meta.is_mixed_batch)
print("cached requests:", runner.num_cached_requests)
print("last step cost :", runner.device.last_step_cost.as_dict())
```

```
num_prefills   : 0
is_mixed_batch : False
cached requests: 2
last step cost : {'duration': 0.00104, 'compute_s': 0.00104, 'memory_s': 0.0, 'comm_s': 0.0,
                  'encoder_s': 0.0, 'overhead_s': 0.0, 'jitter': 1.0, 'flops': 0.0,
                  'bytes': 0.0, 'bound_by': 'compute', 'provenance': 'modeled'}
```

`num_prefills: 0` because the *last* step was a pure decode; `flops: 0.0` because the default
`constant` cost model does no roofline arithmetic. Rerun with
`LLM(..., cost_model_profile="roofline")` and both fill in.

That reach-through is only for learning — nothing above the boundary is allowed to do it
(chapter [03](03-simulation-boundary.md)). The supported way to see the same information is
`/debug/cost_model`.

## Check yourself

- Which single call in `execute_model` is below the simulation boundary?
- What is `query_start_loc`, and what is `logits_indices`?
- Why does a request mid-chunked-prefill produce no output token?
- What is a slot, and what does validating the slot mapping catch?
- Why is `prepare_attn` built for every KV group rather than just group 0?
- Why is `execute_model_async` abstract rather than defaulting to `execute_model`?

## Next

[14. The memory model](14-memory-model.md) — where `num_gpu_blocks` comes from.
