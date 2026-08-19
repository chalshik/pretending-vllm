# 22. LoRA

> **Files:** [`pvllm/config/lora.py`](../../pvllm/config/lora.py), [`pvllm/lora/request.py`](../../pvllm/lora/request.py), [`pvllm/sim/memory.py`](../../pvllm/sim/memory.py) (`compute_lora_bytes`), [`pvllm/v1/core/sched/scheduler.py`](../../pvllm/v1/core/sched/scheduler.py) (the adapter-slot check), [`pvllm/entrypoints/openai/models/serving.py`](../../pvllm/entrypoints/openai/models/serving.py)
> **Upstream:** `vllm/config/lora.py`, `vllm/lora/request.py` (Tier C)
> **Prerequisites:** chapters [10](10-prefix-caching.md), [12](12-scheduler.md), [14](14-memory-model.md).

LoRA is usually described as a fine-tuning technique. From a *serving* engine's point of view it is
three things, none of which is about training: an admission constraint, a memory cost, and a cache
partition. All three are modeled here, and all three are invisible if you only think about the
weights.

## 1. An admission constraint

```python
if (self.lora_config is not None
        and request.lora_request is not None
        and len(scheduled_loras) >= self.lora_config.max_loras
        and request.lora_request.lora_int_id not in scheduled_loras):
    self.waiting.pop_request()
    self.skipped_waiting.add_request(request)
    continue
```

`max_loras` bounds how many **distinct** adapters may be resident at once.

> A request for a fifth adapter waits when four slots are full, even though there is KV capacity and
> a free sequence slot. That is a real source of queueing in a multi-tenant deployment and it is
> invisible unless it is modeled.

Note it is **set aside**, not blocked in place: a request behind it may want an adapter that *is*
resident, and stopping the loop there "would let one tenant's queue block every other tenant's."
Set-aside requests return to the head of the waiting queue before the step ends (chapter
[12](12-scheduler.md)).

And the invariant is asserted, because violating it is a memory-safety error rather than a scheduling
one:

```python
assert len(scheduled_loras) <= self.lora_config.max_loras
```

**The operational lesson:** if you serve eight tenants with `--max-loras 1`, your engine is
effectively serialised per tenant, and no amount of KV capacity fixes it. This engine will show you
that as queue time.

## 2. A memory cost that comes out of the KV pool

A LoRA layer replaces a `[d_in, d_out]` update with `A @ B`, where `A` is `[d_in, r]` and `B` is
`[r, d_out]`. For the attention projections both dimensions are the hidden size, so one adapted
projection costs `2 × r × d` parameters — for every targeted projection in every layer, times
`max_loras`.

Those bytes are resident on the device and come out of the same budget as everything else, so
**serving adapters shrinks the KV pool**:

```bash
python -c "
from pvllm.sim.model_db import load_model_card
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.memory import compute_lora_bytes, compute_memory_profile
m, d, g = load_model_card('dense-8b'), load_device_card('datacenter-80gb'), 2**30
for loras, rank in ((1,16), (8,16), (8,64), (32,64)):
    b = compute_lora_bytes(m, 'bfloat16', max_loras=loras, max_lora_rank=rank, num_target_modules=4)
    p = compute_memory_profile(m, d, dtype='bfloat16', kv_cache_dtype=None, block_size=16,
                               gpu_memory_utilization=0.92, max_model_len=8192,
                               max_num_batched_tokens=8192, max_num_seqs=256, lora_bytes=b)
    print(f'max_loras={loras:3d} rank={rank:3d}: adapters={b/g:5.2f} GiB  '
          f'blocks={p.num_gpu_blocks:6d}  concurrency={p.max_concurrency:6.2f}x')
"
```

```
max_loras=  1 rank= 16: adapters= 0.03 GiB  blocks= 29018  concurrency= 56.68x
max_loras=  8 rank= 16: adapters= 0.25 GiB  blocks= 28906  concurrency= 56.46x
max_loras=  8 rank= 64: adapters= 1.00 GiB  blocks= 28522  concurrency= 55.71x
max_loras= 32 rank= 64: adapters= 4.00 GiB  blocks= 26986  concurrency= 52.71x
```

32 adapters at rank 64 costs 4 GiB and about 7% of your concurrency. Modest — but modest in a
direction you should know about rather than discover.

Two corrections in `compute_lora_bytes` that the obvious arithmetic gets wrong, both documented at
the call site:

- **Only half of an adapter shards under tensor parallelism.** A column-parallel projection splits
  `B` and replicates `A`; a row-parallel one does the reverse. Dividing the whole thing by `tp_size`
  understated per-device memory by nearly a factor of two at high TP — "in the optimistic direction:
  the engine reported KV capacity that does not exist."
- **Layers divide across pipeline stages**, so a stage holds only its share of each adapter.
  Charging every stage the full set overstated the cost by `pp_size` and cost real KV blocks.

And one honest gap: **which projections are targeted is a config-wide assumption here**, defaulting
to the four attention projections. Upstream reads it from each adapter's own config, so an adapter
that also targets the MLP costs more than this reports — the MLP projections are several times wider.
The direction of the error is optimistic, which is worth knowing when the answer is a capacity
number.

`max_lora_rank` is validated against upstream's supported set:

```python
SUPPORTED_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
```

Not arbitrary — the kernels are specialised per rank, and a value outside the set is **rejected
rather than rounded**, because rounding would silently change the memory the adapter occupies.

## 3. A prefix cache partition

From chapter [10](10-prefix-caching.md):

```python
if request.lora_request is not None:
    keys.append(request.lora_request.lora_name)
```

The adapter joins the block hash's extra keys. **Two requests with byte-identical prompts and
different adapters must not share blocks**, because the same tokens under a different adapter produce
different KV.

The consequence for a multi-tenant deployment is worth stating: your shared system preamble is cached
**per adapter**, not once. Eight tenants with the same 800-token preamble occupy eight copies of it in
the KV pool, and each pays its own first-request prefill.

Two details:

- **Keyed by name, not id.** The id would be the more natural identity — two names for one adapter
  would then share cached prefixes — but upstream keys by name, and "a simulator that improves on the
  engine it stands in for is telling its user something the real engine will not do."
- **Ids are assigned by position** in `--lora-modules`, so a given line always produces the same id.
  That matters because an id that shifted between restarts would silently invalidate every cached
  prefix for that adapter.

## Serving adapters over HTTP

```bash
pvllm serve --model dense-8b --enable-lora --max-loras 4 --max-lora-rank 16 \
  --lora-modules support=/adapters/support sales=/adapters/sales
```

Each adapter is served **under its own model name**:

```bash
curl -s localhost:8000/v1/models          # lists dense-8b, support, sales
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"support","prompt":"hello","max_tokens":8}'
```

A request naming an adapter routes to it. No weights are read, of course — the path is a label. What
is real is the routing, the slot accounting, the memory, and the cache partitioning.

`--lora-modules` without `--enable-lora` is an error rather than an inference:

```
ValueError: --lora-modules was given without --enable-lora. Serving an adapter changes both the
memory budget and the admission constraint, so it is not inferred from the presence of a module.
```

## `LoRARequest`

A msgspec `Struct`, because it crosses the engine-core boundary inside `EngineCoreRequest` and has to
serialize. `lora_int_id` must be globally unique per adapter; upstream documents that it does not
enforce this, and neither does this — "the id is the caller's namespace, and two adapters sharing one
would be a caller error that shows up as cache sharing between them."

## Try it: watch the slot constraint bite

```python
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.lora.request import LoRARequest

llm = LLM(model="dense-0.6b", max_model_len=1024, enable_lora=True, max_loras=1,
          trace_path="lora.jsonl")
# four requests, four different adapters, one slot
for i in range(4):
    ...  # submit via llm_engine.add_request with lora_request=LoRARequest(...)
```

Then compare `pvllm trace view lora.jsonl` at `--max-loras 1` against `--max-loras 4`: the same
workload, and the difference is entirely queue time. `tests/v1/test_lora.py` asserts exactly this
behaviour if you would rather read it than write it.

## Check yourself

- Name the three serving-visible effects of LoRA, none of which is about weights.
- Why is a request whose adapter has no free slot *set aside* rather than left at the head of the
  queue?
- Why does only half of an adapter shard under tensor parallelism?
- Eight tenants share an 800-token preamble under eight adapters. How many copies are in the KV
  cache?
- Why is the cache keyed by adapter *name* rather than id, even though the id is a better identity?

## Next

[23. Multimodal](23-multimodal.md) — images as placeholders and a second budget.
