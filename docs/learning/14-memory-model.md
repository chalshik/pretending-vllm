# 14. The memory model

> **Files:** [`pvllm/sim/memory.py`](../../pvllm/sim/memory.py), [`pvllm/sim/device.py`](../../pvllm/sim/device.py), [`pvllm/sim/hardware_db.py`](../../pvllm/sim/hardware_db.py), [`pvllm/sim/model_db.py`](../../pvllm/sim/model_db.py), [`pvllm/sim/weights.py`](../../pvllm/sim/weights.py)
> **Upstream:** none — Tier **D** (upstream *measures* this with a real profiling pass)
> **Prerequisites:** chapters [09](09-kv-cache-blocks.md), [13](13-worker-and-model-runner.md).
> **Contract:** analytic — exact given the model card and the device card, with **one** modeled term.

This is the chapter that answers capacity questions. "Does a 70B model at 128k context fit on
eight 80 GB cards at `gpu_memory_utilization 0.92`, and how many concurrent requests does that
leave?" is arithmetic, and this is the arithmetic.

## The budget

```
capacity            = device_card.memory_bytes
usable              = capacity × gpu_memory_utilization
kv_pool             = usable
                      − weight_bytes
                      − activation_peak          ← the one modeled term
                      − non_torch_overhead
                      − graph_bytes
                      − lora_bytes
num_gpu_blocks      = kv_pool // kv_bytes_per_block
max_concurrency     = allocatable_blocks / blocks_for_one_request
```

Every term except `activation_peak` is arithmetic on declared quantities, and is **exact**.

## Term by term

### Weights

```python
def compute_weight_bytes(model, dtype, tp_size=1, ep_size=None) -> int:
    embedding = model.embedding_parameters
    experts   = model.num_hidden_layers * model.expert_parameters_per_layer
    dense     = model.num_parameters - embedding - experts
    ...
    return embedding * dtype_bytes + (dense * dtype_bytes) // tp_size + local_experts * dtype_bytes
```

Two subtleties that the obvious version gets wrong:

**Embeddings do not shard under tensor parallelism.** Dividing everything by TP is the common
shortcut, and it understates per-device memory on large-vocabulary models — a 128k-vocab model
puts over a gigabyte in embeddings alone. You can see the effect:

```bash
python -c "
from pvllm.sim.model_db import load_model_card
from pvllm.sim.memory import compute_weight_bytes
g = 2**30
c = load_model_card('dense-70b')
for tp in (1, 4, 8):
    print(f'tp={tp}: {compute_weight_bytes(c, \"bfloat16\", tp)/g:.2f} GiB per device')
"
```

```
tp=1: 131.42 GiB per device
tp=4: 35.79 GiB per device
tp=8: 19.85 GiB per device
```

131.42 / 8 = 16.4, but the real answer is 19.85 — the difference is the unsharded embedding
tables. On a tight fit, that gap is the whole decision.

**Experts divide differently under expert parallelism.** Each device owns *whole* experts
rather than a slice of every one, across `ep_size = data_parallel_size × tensor_parallel_size`
devices, while attention and norms keep sharding by `tp_size`. And the division is a
**ceiling, per expert**: 8 experts over 3 ranks is 3/3/2, and the device that has to fit the
model is the one holding 3. Chapter [24](24-parallelism.md).

### The activation peak — the one honest caveat

```python
ACTIVATION_HIDDEN_MULTIPLIER = 6
ACTIVATION_INTERMEDIATE_MULTIPLIER = 2

per_token = (hidden * 6 + intermediate_local * 2) * dtype_bytes
activations = max_num_batched_tokens * per_token
logits = max_num_seqs * vocab_size * 4          # fp32 regardless of model dtype
```

Upstream **measures** this: `determine_available_memory` runs a real profiling forward pass at
`max_num_batched_tokens` and reads the allocator's high-water mark. There is nothing to
measure here, so it is estimated from the architecture with those coefficients.

The consequence propagates, and the source states it plainly: `num_gpu_blocks` is exact
*given* an activation peak, and the activation peak is modeled. On a large model the term is
small next to the weights and the error is negligible; on a small model at a large batch it
can be a few percent of blocks. `MemoryProfile.activation_is_modeled` is `True` so anything
reporting these numbers can say so — and the startup line does:

```
activation_peak=1.30GiB (modeled)
```

The logits buffer is called out separately because it is frequently the largest single
activation and scales with **vocabulary** rather than hidden size: a 128k-vocab model sampling
256 sequences holds 128 MiB of fp32 logits, which dwarfs the per-token activations of a small
model.

### Non-torch overhead

```python
DEFAULT_NON_TORCH_OVERHEAD_BYTES = 1 << 30   # 1 GiB
```

Allocator fragmentation, CUDA context, NCCL buffers — everything real vLLM finds already
resident before it allocates anything. Upstream measures this too.

### KV bytes per block

```python
kv_bytes_per_block = kv_cache_groups[0].kv_cache_spec.page_size_bytes * layers_per_group
```

A block backs one page in each of a **group's** layers, and the groups *share* the pool. For a
dense model that is every layer; for a hybrid one it is the layers of one group, so a block is
smaller and the pool holds proportionally more of them. Chapter
[11](11-hybrid-kv-groups.md).

This is derived exactly the way `EngineCore._initialize_kv_caches` derives it, and the
docstring explains why that matters: an earlier version rescaled the model's per-token cost
instead, and two integer divisions drifted whenever `num_hidden_layers % pp_size` was
non-zero — so the profile printed one `num_gpu_blocks` in the startup line and the scheduler
was handed another. That is not cosmetic; the "no request could ever be served" guard ran on
the larger number, so a config that could not fit a single request passed startup and then
hung forever with no error and no log line.

## Two startup refusals

Both exist so a bad configuration fails at startup with an instruction, rather than at request
time as a mysterious hang.

**No room for KV at all:**

```
SimOutOfMemoryError: No memory left for the KV cache. The model's weights (131.42GiB),
modeled activation peak (2.11GiB), LoRA adapters (0.00GiB), and non-torch overhead
(1.00GiB) already exceed the 73.60GiB budget on a 80.00GiB device at
gpu_memory_utilization=0.92.
Try: raise gpu_memory_utilization, lower max_num_batched_tokens, max_num_seqs, or
max_loras, use a smaller model card, or pick a larger device card.
```

**Not enough room for one request:**

```
SimOutOfMemoryError: The KV cache holds 40 blocks (40 allocatable), but a single request at
max_model_len=131072 (window 131072) needs 8192 (block_size=16). No request could ever be
served.
Try: lower max_model_len, raise gpu_memory_utilization, or pick a larger device card.
```

(reproduce it with `num_gpu_blocks_override=40` against `dense-8b` at its default 131072
context)

The second guard is the more valuable one. Left to request time it would look like a request
that queues forever for capacity that will never exist — `allocate_slots` returning `None`
every step, with no error and no log line.

Note the peak-vs-steady-state subtlety in `windowed_blocks_for_one_request`: a windowed request
briefly holds *more* than its window, because eviction runs after the step has allocated slots
for everything it scheduled. Under-counting there turns the guard into exactly the silent hang
it exists to prevent. "A window of 64 against a 1024-token step budget needs 69 blocks and the
old arithmetic asked for 5."

## `max_concurrency` — the number to quote

```python
max_concurrency = allocatable_blocks / blocks_for_one_request
```

How many requests **at `max_model_len`** the pool can hold. Three refinements that each cost
real accuracy if skipped:

- the **null block** comes off the allocatable count when any group sheds (chapter
  [09](09-kv-cache-blocks.md));
- a **windowed** request is counted at its window, not at `max_model_len` — that *is* the
  capacity argument for windows;
- a **hybrid** request is counted per group and summed, because the groups do not cost the
  same.

Watch it move:

```bash
python -c "
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.memory import compute_memory_profile
from pvllm.sim.model_db import load_model_card
d = load_device_card('datacenter-80gb')
rows = [('dense-8b', 1, 8192, None), ('dense-8b', 1, 131072, None),
        ('dense-8b', 1, 131072, 4096), ('dense-70b', 4, 8192, None), ('dense-70b', 8, 8192, None)]
for m, tp, ctx, sw in rows:
    p = compute_memory_profile(load_model_card(m), d, dtype='bfloat16', kv_cache_dtype=None,
                               block_size=16, gpu_memory_utilization=0.92, max_model_len=ctx,
                               max_num_batched_tokens=8192, max_num_seqs=256,
                               tp_size=tp, sliding_window=sw)
    print(f'{m:10s} tp={tp} ctx={ctx:6d} sw={str(sw):5s} '
          f'weights={p.weight_bytes/2**30:6.2f}GiB blocks={p.num_gpu_blocks:6d} '
          f'concurrency={p.max_concurrency:7.2f}x')
"
```

```
dense-8b   tp=1 ctx=  8192 sw=None  weights= 14.96GiB blocks= 29034 concurrency=  56.71x
dense-8b   tp=1 ctx=131072 sw=None  weights= 14.96GiB blocks= 29034 concurrency=   3.54x
dense-8b   tp=1 ctx=131072 sw=4096  weights= 14.96GiB blocks= 29034 concurrency=  37.75x
dense-70b  tp=4 ctx=  8192 sw=None  weights= 35.79GiB blocks= 29261 concurrency=  57.15x
dense-70b  tp=8 ctx=  8192 sw=None  weights= 19.85GiB blocks= 84814 concurrency= 165.65x
```

Five capacity answers, no GPU. Read row 2 against row 3 (a window is worth 10.7×) and row 4
against row 5 (doubling TP nearly tripled concurrency, because weights came out of the pool's
way).

## The memory ledger

`MemoryProfile` is the *plan*. `MemoryLedger` ([`sim/memory.py`](../../pvllm/sim/memory.py)) is
the enforcement:

```python
class MemoryLedger:
    def allocate(self, pool: str, num_bytes: int) -> None:
        if num_bytes > self.free_bytes:
            raise SimOutOfMemoryError(...)
```

The profile's pools are applied as **real allocations** rather than bookkeeping —
`weights`, `activation_peak`, `non_torch_overhead`, `graph`, `kv_cache` — so a later allocation
that would not fit actually fails. "The ledger is the thing that makes a capacity answer
trustworthy": without it, the simulator could quietly pretend the device is bigger than its
card says.

The OOM message is shaped like upstream's so a product that pattern-matches on OOM text
behaves the same way against either engine.

## The startup timeline

[`sim/weights.py`](../../pvllm/sim/weights.py) reproduces the *shape* of startup, in the same
log line upstream emits:

```
init engine (profile, create kv cache, warmup model) took 8.87 seconds
(load=8.03s, profile=0.75s, kv_cache=0.00s [56.34GiB], graph_capture=0.09s) [modeled]
```

Why this matters more than it sounds: `/health` must report ready only after load and profiling
complete, and a product that polls readiness — or times out waiting for it — exercises real
behaviour only if startup takes plausible time. **A simulator that is ready instantly cannot
surface a readiness bug.**

Which brings the one gotcha in the whole chapter. Under the default `constant` cost model,
weight loading is **free** and the server is ready in about a tenth of a second. Under
`roofline`, an 8B model over `datacenter-80gb`'s declared load bandwidth takes about eight
seconds — because that is what the arithmetic says. So:

```bash
# will NOT exercise a client's readiness timeout
pvllm serve --model dense-8b --clock-mode real

# will
pvllm serve --model dense-8b --clock-mode real --cost-model-profile roofline
```

## What the device card declares

```python
@dataclass
class DeviceCard:
    memory_bytes: int              # exact, feeds this whole chapter
    memory_bandwidth: float        # the memory term of the roofline
    peak_flops: dict[str, float]   # DENSE peaks, not sparsity-doubled
    interconnect_bandwidth: float  # the tensor-parallel allreduce term
    launch_overhead: float
    load_bandwidth: float          # weights "read from disk" at startup
    num_devices: int = 1
    mfu: float = 0.45              # ┐
    bw_eff: float = 0.80           # ├ achievable fractions of peak — the calibration knobs
    link_eff: float = 0.75         # ┘
    provenance: str                # where these came from, and how much to trust them
```

Only the first field is used by *this* chapter; the rest belong to chapter
[15](15-cost-model.md). Note `peak_flops` is deliberately dense rather than sparsity-doubled —
no kernel here exploits sparsity, so a doubled figure would flatter every compute estimate.

A missing dtype entry fails clearly rather than defaulting:

```
KeyError: device card 'tiny-2gb' has no peak_flops entry for dtype 'float64';
it declares ['bfloat16', 'float16', 'float32']
```

## Try it: your own hardware

```bash
cat > my-card.json <<'EOF'
{
  "name": "my-card",
  "memory_bytes": 51539607552,
  "memory_bandwidth": 2000000000000,
  "peak_flops": {"bfloat16": 200000000000000, "float16": 200000000000000,
                 "float32": 25000000000000},
  "interconnect_bandwidth": 100000000000,
  "launch_overhead": 0.000005,
  "load_bandwidth": 1500000000,
  "num_devices": 4,
  "mfu": 0.4, "bw_eff": 0.75, "link_eff": 0.7,
  "provenance": "vendor spec sheet, uncalibrated"
}
EOF

pvllm bench latency --model dense-8b --device-card ./my-card.json \
  --max-model-len 8192 --input-len 512 --output-len 16
```

The memory arithmetic is now exact for your device. The *cost* arithmetic is still modeled —
next chapter.

## Check yourself

- Which single term in the budget is modeled rather than computed, and what does upstream do
  instead?
- Why is `dense-70b` at TP=8 19.85 GiB per device rather than 131.42/8 = 16.4?
- Why is `max_concurrency` computed against *allocatable* blocks rather than `num_gpu_blocks`?
- Why does the "no request could ever be served" check exist at startup rather than being left
  to `allocate_slots`?
- Why does a `--clock-mode real` demo need `--cost-model-profile roofline` to exercise a
  readiness timeout?

## Next

[15. The cost model](15-cost-model.md) — the one place this project can mislead you.
