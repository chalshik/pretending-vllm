# 11. Hybrid KV cache groups

> **Files:** [`pvllm/v1/kv_cache_interface.py`](../../pvllm/v1/kv_cache_interface.py), [`pvllm/v1/core/kv_cache_utils.py`](../../pvllm/v1/core/kv_cache_utils.py) (`get_kv_cache_groups`), [`pvllm/v1/core/kv_cache_coordinator.py`](../../pvllm/v1/core/kv_cache_coordinator.py), [`pvllm/v1/core/single_type_kv_cache_manager.py`](../../pvllm/v1/core/single_type_kv_cache_manager.py), [`pvllm/platforms/sim.py`](../../pvllm/platforms/sim.py)
> **Upstream:** same paths (Tier **A**), plus `vllm/platforms/cuda.py`'s block-size alignment
> **Prerequisites:** chapters [09](09-kv-cache-blocks.md), [10](10-prefix-caching.md).

Chapters 9 and 10 assumed every layer caches the same way. Modern models break that
assumption in four different directions, and each one changes a capacity answer by a lot.
This is the chapter that a 2023-era mental model of vLLM is missing entirely.

## Four ways a layer can cache

[`kv_cache_interface.py`](../../pvllm/v1/kv_cache_interface.py) has one spec class per
behaviour. The property that matters is `page_size_bytes` — how many bytes one block
occupies, for one layer.

| Spec | Caches | Page size | Grows with context? |
|---|---|---|---|
| `FullAttentionSpec` | key + value per token, forever | `2 × block_size × kv_heads × head_size × dtype` | yes, linearly |
| `SlidingWindowSpec` | key + value, but only the last *W* tokens | same | **no** — bounded at *W* |
| `MLAAttentionSpec` | one compressed latent per token | `block_size × 1 × (kv_lora_rank + qk_rope) × dtype` (no factor of 2) | yes, but ~4× smaller |
| `MambaSpec` | a fixed recurrent state per request | `state_bytes` — **independent of `block_size`** | **no** — constant |

See them for real:

```bash
python -c "
from pvllm.engine.arg_utils import EngineArgs
from pvllm.v1.worker.gpu.attn_utils import get_kv_cache_spec
from pvllm.v1.core.kv_cache_utils import get_kv_cache_groups
for m in ['dense-8b', 'hybrid-4b', 'hybrid-ssm-8b', 'mla-16b']:
    c = EngineArgs(model=m, max_model_len=8192).create_engine_config()
    groups = get_kv_cache_groups(get_kv_cache_spec(c))
    kinds = {}
    for g in groups:
        k = type(g.kv_cache_spec).__name__
        kinds[k] = kinds.get(k, 0) + 1
    print(f'{m:14s} block_size={c.cache_config.block_size:5d} groups={len(groups):2d} '
          f'{kinds} page={groups[0].kv_cache_spec.page_size_bytes}')
"
```

```
dense-8b       block_size=   16 groups= 1 {'FullAttentionSpec': 1} page=65536
hybrid-4b      block_size=   16 groups= 6 {'SlidingWindowSpec': 5, 'FullAttentionSpec': 1} page=65536
INFO ... State-space model: block_size 16 -> 1040 so an attention page (4096 B/token)
         covers one layer's 4255744 B recurrent state (R6.7). This changes block counts
         and every block hash value.
hybrid-ssm-8b  block_size= 1040 groups= 7 {'MambaSpec': 6, 'FullAttentionSpec': 1} page=4259840
mla-16b        block_size=   16 groups= 1 {'MLAAttentionSpec': 1} page=18432
```

## Why groups exist

One pool of fixed-size pages cannot serve layers whose pages differ in size — it would
fragment. So layers are partitioned into **KV cache groups**, where every group's page is
the same size, and the groups *share* the one pool.

The partitioning algorithm (`get_kv_cache_groups`) has a shape that is not obvious:

> A model with 25 windowed layers and 5 full ones is **not** two groups of 25 and 5. It is
> **six groups of five**.

Because: the layers repeat with a pattern, and the pool can only be divided evenly if every
group occupies the same bytes per block. Groups of unequal size would need pages of unequal
size. So the group size is the smallest bucket, and each larger bucket splits into
`ceil(len / group_size)` groups. That is exactly what `hybrid-4b` shows above: 5 windowed
layers per group × 5 groups, plus 1 full-attention group.

Two implementation details worth knowing:

- **Striping, not slicing.** Upstream uses `layers[i::num_groups]` rather than contiguous
  slices, because under pipeline parallelism a contiguous split puts whole groups on one
  stage and leaves empty ones on another — which then get padded to the same size and waste
  the memory the grouping exists to save.
- **A heuristic for near-equal buckets.** If `max(sizes) < min(sizes) * 1.5`, the group size
  becomes `max(sizes)` instead: padding one bucket up wastes less than splitting the larger
  one into two mostly-padding groups.

Group lengths need *not* match — only per-layer page sizes must. A bucket that does not
divide evenly leaves one short group, and the pool is sized from the longest, so those slots
are paid for and unused. The code logs that as waste rather than claiming to have "added
padding".

## Sliding windows: bounded KV

`SlidingWindowSpec` is the one whose consequence is easiest to state:

> A model with a 128k context and a 4k window holds 4k tokens of KV per request **however
> long the conversation gets.**

That is a different capacity planning problem — bounded by concurrency rather than by
conversation length. Watch it:

```bash
python -c "
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.memory import compute_memory_profile
from pvllm.sim.model_db import load_model_card
d, m = load_device_card('datacenter-80gb'), load_model_card('dense-8b')
for sw in (None, 4096):
    p = compute_memory_profile(m, d, dtype='bfloat16', kv_cache_dtype=None, block_size=16,
                               gpu_memory_utilization=0.92, max_model_len=131072,
                               max_num_batched_tokens=8192, max_num_seqs=256, sliding_window=sw)
    print(f'sliding_window={sw}: num_gpu_blocks={p.num_gpu_blocks} max_concurrency={p.max_concurrency:.2f}x')
"
```

```
sliding_window=None: num_gpu_blocks=29034 max_concurrency=3.54x
sliding_window=4096: num_gpu_blocks=29034 max_concurrency=37.75x
```

**Same pool, 10.7× the concurrency.** That ratio is the entire argument for windowed
attention, and it is why this is modeled rather than approximated.

The mechanism has two halves:

**`SlidingWindowManager.get_num_skipped_tokens`** — how much of the front has fallen out of
the window, and therefore which blocks can be released while the request is still running.
`remove_skipped_blocks` does the releasing.

**The null block** — those released slots in the block table are replaced with the shared
null block, not deleted. The table must keep its length because positions index into it.

And a timing subtlety with real consequences. Window eviction runs in
`update_from_output` — *after* the step's output has been folded back — not at schedule
time:

> Scheduling inflates `num_computed_tokens` by the drafts it is about to verify, and
> evicting on that inflated boundary freed blocks that were still inside the true window.
> Those blocks went back to the pool and were handed to other requests, which is
> cross-request KV corruption: the exact failure the window is not allowed to cause.

A windowed group's cache hit is also not a prefix. It is a contiguous run covering the
window *ending at the candidate length* — which is why the coordinator has to iterate
(below).

## Mamba: constant KV

A state-space layer holds one recurrent state per request. Position *N*'s state already
summarises every token before it, so there is nothing to keep per token.

Two properties are needed for that claim to be true, and for a while only one of them was:

1. **The page does not scale with `block_size`** — that is `MambaSpec.page_size_bytes`
   returning `state_bytes`.
2. **The *number* of pages a request holds does not scale with context** — that is
   `MambaManager`, following upstream's default `mamba_cache_mode="none"`: it keeps one live
   state and nulls the rest.

Without (2), a spec whose page is constant was held `ceil(context / block_size)` times and
the constant bought nothing. `state_blocks_for_one_request` in
[`sim/memory.py`](../../pvllm/sim/memory.py) has the fixed accounting, and its docstring says
what the old version got wrong.

### The block-size shock

One pool cannot hold two page sizes, and a recurrent state **cannot shrink**. So upstream
reconciles them the only way available: it **grows the attention block size** until an
attention page is at least as large as the state, then pads the state page up to match
exactly. That happens in the platform hook, before any spec is built
([`platforms/sim.py::_align_hybrid_block_size`](../../pvllm/platforms/sim.py)).

For a Nemotron-H-class model that moves the block size **from 16 tokens to about 1040**. The
consequence is much larger than the padding it saves:

- how many blocks a request holds changes (C2);
- prefix cache granularity becomes 1040 tokens instead of 16 (C3);
- **every block hash value changes**, because a hash is computed over `block_size` tokens.

A run that kept `block_size` at 16 would be wrong on all three even with the state bytes
right. This is the single most surprising number in the repository, and the engine logs it
at startup so nobody has to discover it.

## MLA: smaller KV that does not shard

Multi-head latent attention stores one compressed latent per token instead of a key and a
value per head. Two differences from full attention, and the second is the one that catches
capacity plans:

```python
def kv_bytes_per_token(self, kv_dtype=None, tp_size=1) -> int:
    ...
    if self.use_mla:
        return self.mla_head_size * dtype_bytes * layers  # no ×2, no ÷tp_size
    kv_heads_local = max(1, self.num_key_value_heads // tp_size)
    return 2 * kv_heads_local * self.head_dim * dtype_bytes * layers
```

- **No factor of two** — one latent, not a key and a value.
- **No division by `tp_size`** — upstream's `get_num_kv_heads` returns 1 for MLA *before*
  dividing, so the latent is **replicated on every rank**.

So scaling tensor parallelism on a DeepSeek-class model buys weights and compute and buys
**nothing at all** on the KV cache. That is exactly the sort of thing a plan gets wrong from
first principles, and it is why MLA gets its own spec class rather than a flag.

## The coordinator: reconciling a hit across groups

With several groups, "did this prefix hit" needs one answer, not six — the scheduler advances
a single `num_computed_tokens` for the request, and a group that cannot supply those tokens
would be read for KV that was never written.

`KVCacheCoordinator.find_longest_cache_hit` runs upstream's **fixed point**: each attention
type either accepts the candidate length or reduces it, and any reduction restarts the pass.
It converges because the length only ever decreases.

Why it has to iterate rather than take a minimum:

> A full-attention group needs the prefix from token zero, so its hit is *downward-closed* —
> shortening the candidate only trims it. A windowed group needs a contiguous run covering
> its window *ending at the candidate*, so moving the candidate can invalidate the run it
> just found and force a different one. Asking each type once and taking the smallest answer
> would report a hit the windowed group cannot actually serve.

The type test is written **positively** (`isinstance(spec, FullAttentionSpec)`) rather than as
"not a sliding window", because the negative form sweeps a state-space group into the
full-attention shortcut — looking it up once and then only min-ing, leaving a block list from
a longer candidate attached to a shorter reconciled length.

## Blended capacity

A hybrid request holds blocks in **every** group, and the groups do not cost the same. A 5:1
windowed-to-full model's KV per request is neither bounded nor unbounded — it is a blend, and
reporting either extreme answers a capacity question with the wrong model's number.
`compute_memory_profile` sums per group:

```python
for group in kv_cache_groups:
    if windowed:
        blocks += windowed_blocks_for_one_request(...)
    elif mamba:
        blocks += state_blocks_for_one_request(...)
    else:
        blocks += ceil(max_model_len / block_size)
```

Note `windowed_blocks_for_one_request` is **not** `ceil((W-1)/block) + 1`. That is the steady
state, and a request passes through a larger one on the way: eviction runs *after* the step
has allocated slots for everything it scheduled, so a prefill chunk of
`max_num_batched_tokens` is resident alongside the window before anything is given back.
Under-counting there is not a small reporting error — it is a **silent hang**: the startup
guard passes and then `allocate_slots` returns `None` every step, forever.

## The escape hatch

```python
from pvllm.entrypoints.llm import LLM

LLM(model="hybrid-4b", max_model_len=8192, disable_hybrid_kv_cache_manager=True)
```

Upstream's own flag: promote every sliding-window layer to full attention, giving up the
memory saving and keeping one KV cache group. Here it is worth more than compatibility —
**the two runs side by side *are* the capacity argument for hybrid attention**, on one model
rather than two.

> **Note:** this is an `EngineArgs` field, but it is **not exposed as a CLI flag** today —
> `pvllm serve --disable-hybrid-kv-cache-manager` is rejected as an unrecognized argument even
> though the README advertises it. Reach it through the Python API, as above.

## Try it

Price the same model both ways:

```python
from pvllm.engine.arg_utils import EngineArgs
from pvllm.v1.core.kv_cache_utils import get_kv_cache_groups
from pvllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

for disabled in (False, True):
    cfg = EngineArgs(
        model="hybrid-4b", max_model_len=8192, disable_hybrid_kv_cache_manager=disabled
    ).create_engine_config()
    groups = get_kv_cache_groups(get_kv_cache_spec(cfg))
    kinds = sorted({type(g.kv_cache_spec).__name__ for g in groups})
    print(f"hybrid manager disabled={disabled}: {len(groups)} group(s), {kinds}")
```

```
hybrid manager disabled=False: 6 group(s), ['FullAttentionSpec', 'SlidingWindowSpec']
hybrid manager disabled=True: 1 group(s), ['FullAttentionSpec']
```

Every windowed layer was promoted to full attention, and six groups collapsed to one. Then run
`pvllm bench latency --model hybrid-4b --max-model-len 8192 --input-len 1024 --output-len 32`
both ways and compare the startup line: `num_gpu_blocks` and `max_concurrency` are where the
promotion is paid for.

## Check yourself

- Why is a model with 25 windowed and 5 full layers six groups rather than two?
- Which two properties must both hold for a Mamba layer's KV to be constant in context?
- Why does a state-space model's block size jump from 16 to ~1040, and which two contract
  classes does that move?
- Why does tensor parallelism not reduce an MLA model's KV footprint?
- Why does the cache-hit reconciliation iterate instead of taking a minimum over groups?
- Why must window eviction run after `update_from_output` rather than during scheduling?

## Next

[12. The scheduler](12-scheduler.md) — the centerpiece, and the file everything else exists
to serve.
