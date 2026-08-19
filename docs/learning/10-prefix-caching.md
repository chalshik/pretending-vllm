# 10. Prefix caching

> **Files:** [`pvllm/v1/core/kv_cache_utils.py`](../../pvllm/v1/core/kv_cache_utils.py) (the hashing half), [`pvllm/v1/core/block_pool.py`](../../pvllm/v1/core/block_pool.py), [`pvllm/v1/core/kv_cache_manager.py`](../../pvllm/v1/core/kv_cache_manager.py), [`pvllm/v1/core/kv_cache_metrics.py`](../../pvllm/v1/core/kv_cache_metrics.py)
> **Upstream:** same paths (Tier **A**)
> **Prerequisites:** chapter [09](09-kv-cache-blocks.md).
> **Contract:** C3 — *prefix cache hit rate and block hash values* must match upstream.

Prefix caching is the highest-leverage feature in a real deployment. If every request in
your product carries the same 800-token system preamble, it decides whether you prefill it
once or once per request. On by default since V1.

## The mechanism in one paragraph

When a block becomes **full**, hash its contents *together with the hash of the block before
it* and publish it in a map from hash to block. When a new request arrives, hash its
prompt's blocks the same way and walk the map from the start, stopping at the first miss.
The blocks that hit are adopted — reference counted, pulled out of the free queue — and the
request is told those tokens are already computed.

Everything interesting is in the details of that paragraph.

## Detail 1: the hash chains

```python
def hash_block_tokens(
    hash_fn, parent_block_hash, curr_block_token_ids, none_hash, extra_keys=None
) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = none_hash
    return BlockHash(
        hash_fn((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )
```

**The chain through `parent_block_hash` is what makes this a *prefix* cache rather than a
block cache.** Block 3 of one sequence matches block 3 of another only if blocks 0–2
matched too.

Hash a block's tokens alone and two requests sharing a middle passage but not a beginning
would collide — and the second would read KV computed under a different preceding context.
That is not a cache miss; it is a wrong answer.

The chain also gives the lookup its shape: a miss ends the search, because a block that is
not cached guarantees every block after it is not either.

```python
for block_hash in block_hashes[: max_length // block_size]:
    cached = block_pool.get_cached_block(block_hash, group_id=group_id)
    if cached is None:
        break
    computed.append(cached)
```

## Detail 2: only *full* blocks are hashed

```
def request_block_hasher(request):
    start = len(request.block_hashes) * block_size
    ...
    while start + block_size <= num_tokens:      # ← full blocks only
```

A partial tail block is never hashed. A later token would change its contents, so caching it
would publish a block whose identity is about to change underneath any request that matched
it.

This is why **block granularity always rounds a shared prefix down**. Two prompts sharing
589 tokens at a block size of 16 share 36 blocks = 576 tokens; the block straddling the
divergence is recomputed. It is also why block size is a cache-granularity knob and not only
a memory knob: a larger block wastes less bookkeeping and captures less sharing.

The hasher is called incrementally, from `Request.append_output_token_ids`, so hashing costs
one pass over each block exactly once — and generated tokens get hashed too, which is how a
multi-turn conversation hits its own earlier turns.

## Detail 3: extra keys — the cache-poisoning guards

Two requests with identical tokens must **not** share blocks when anything else about them
differs:

```python
def generate_block_hash_extra_keys(request, start_token_idx=0, end_token_idx=None):
    keys = []
    if request.lora_request is not None:
        keys.append(request.lora_request.lora_name)  # different adapter ⇒ different KV
    if request.mm_features:
        keys.extend((f.identifier, f.position - start_token_idx) for f in ...)
    if start_token_idx == 0 and request.cache_salt:
        keys.append(request.cache_salt)  # tenant partitioning
    return tuple(keys) if keys else None
```

- **LoRA** — the same tokens under a different adapter produce different KV. Keyed by the
  adapter's *name*, as upstream keys it (chapter [22](22-lora.md)).
- **Multimodal** — only the images this *block* overlaps, plus each item's offset within the
  block. Folding every image into every block's key would partition the text *before* the
  first image too, and the reported hit rate would fall far below a real deployment's. The
  offset is what keeps two different tilings of the same images apart (chapter
  [23](23-multimodal.md)).
- **`cache_salt`** — a caller-supplied partition, applied only to block 0 because every later
  block chains through block 0's hash and already carries it.

Omitting any of these is a cache-poisoning bug: one request silently reading another's KV.
Which is why the unimplemented cases in this function raise rather than being skipped.

Upstream's *order* is preserved exactly — LoRA, then multimodal, then the salt — because the
tuple is hashed, so the order is part of the value, and C3 makes hash values themselves part
of the contract.

## Detail 4: the hash algorithm, and one honest caveat

```python
_HASH_FUNCTIONS = {"sha256": sha256_hash, "builtin": builtin_hash}
```

`sha256` is upstream's default at the pin and the default here. `builtin` (Python's `hash`,
8 bytes) is faster and far more collision-prone; it is only safe at all because a hit is
verified by the block still being *resident*, not by the digest alone. It is also **salted
per process** by `PYTHONHASHSEED`, so it is not reproducible across runs.

Now the caveat, which is a documented divergence from upstream:

```python
def compute_none_hash(hash_fn, seed) -> BlockHash:
    """The sentinel standing in for "no parent block"."""
    return BlockHash(hash_fn(("pvllm-none-hash", seed)))
```

Upstream uses `os.urandom(32)` unless `PYTHONHASHSEED` is set, so cache keys are not
predictable across processes. That would make block hashes differ on every run here, which
breaks reproducibility and makes a recorded conformance trace incomparable to the next one.
So the sentinel is derived from the run seed instead.

The consequence, stated precisely: **hit *rates* and block allocation order are reproducible
either way and can be compared to a real vLLM run directly. Hash *values* can only be
compared if that run had `PYTHONHASHSEED` set to a matching value.** Upstream's own warning
says as much. It is a real limit on what C3 can check, not a detail.

## Detail 5: at least one token is always recomputed

```python
per_group, num_computed_tokens = self.coordinator.find_longest_cache_hit(
    list(request.block_hashes), max(0, request.num_tokens - 1)
)
```

A request whose *every* block is cached would otherwise be scheduled with zero new tokens —
nothing to run, no logits, no sampled token, and it would never progress.

Note *how* it is enforced: by capping the search length, not by trimming the answer. For full
attention the two are arithmetically identical. For a group whose hit is not a prefix — a
state-space group's hit is `[null, ..., null, state]`, where the one meaningful block is the
*last* element — trimming afterwards would keep the placeholders and throw away the state
while still claiming those tokens were computed. Chapter [11](11-hybrid-kv-groups.md).

## Detail 6: eviction preserves prefixes, if the order is right

Chapter [09](09-kv-cache-blocks.md) covered the two ordering rules. This is where they pay
off:

- freeing a request's blocks **tail-first** means the head of the sequence — the part
  another request is most likely to share — survives longest in the free queue;
- **unhashed blocks first** means blocks that can never hit are spent before blocks that can.

`_maybe_evict_cached_block` is what makes eviction a real event rather than a stale entry
pointing at reused memory: the hash is cleared and the map entry dropped as the block is
reallocated.

## Measuring it

[`kv_cache_metrics.py`](../../pvllm/v1/core/kv_cache_metrics.py) tracks queries, hits,
evictions, and resident cached blocks. Note the unit: **queries and hits are counted in
tokens, not blocks or requests**, so the ratio is "what fraction of prompt tokens did we
avoid computing" — the number that actually predicts your prefill savings.

Three surfaces:

```bash
# Prometheus (chapter 20)
curl -s localhost:8000/metrics | grep prefix_cache
# vllm:prefix_cache_queries_total  ...
# vllm:prefix_cache_hits_total  ...

# live, per request
curl -s localhost:8000/debug/prefix_cache | python -m json.tool

# after the fact
pvllm trace view run.jsonl   #   prefix cache: 64/90 tokens (71.1%)
```

And `RequestOutput.num_cached_tokens` reports it per request, which is the one to assert on
in your own tests.

## Try it: the whole feature, end to end

```python
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams

system = "You are a helpful assistant. " * 20  # 581 tokens under the mock tokenizer
llm = LLM(model="dense-0.6b", max_model_len=2048)

for out in llm.generate(
    [system + "What is one?", system + "What is two?"], SamplingParams(max_tokens=4)
):
    print(
        "prompt_tokens =", len(out.prompt_token_ids), " cached =", out.num_cached_tokens
    )
```

```
prompt_tokens = 593  cached = 0
prompt_tokens = 593  cached = 576
```

Now the experiment that teaches the most: **turn it off.**

```python
llm = LLM(model="dense-0.6b", max_model_len=2048, enable_prefix_caching=False)
# → cached = 0 for both, and the trace shows two full prefills
```

And the one that teaches the second most: change `block_size` to 32 and watch the hit drop
to 576 → 576 (still 18 blocks of 32) but the *number of blocks* halve; change the shared
prefix length by a few tokens and watch the hit move in steps of `block_size`, never
smoothly.

## Where it does *not* help

Worth knowing before you tune for it:

- **Data parallelism partitions the cache.** Each replica has its own pool, so two requests
  sharing a preamble hit only if the router sent them to the same replica. A workload whose
  hit rate looks excellent on one engine can lose most of it at `--data-parallel-size 8`.
  Chapter [24](24-parallelism.md).
- **LoRA partitions the cache.** Two tenants with the same prompt and different adapters
  share nothing. Chapter [22](22-lora.md).
- **Cascade attention is not modeled here.** The common-prefix block count is computed and
  carried through `SchedulerOutput` exactly as upstream does, and you can read it at
  `/debug/cost_model`, but the cost model ignores it. So shared-prefix workloads are modeled
  **pessimistically** — a real backend taking that optimisation would be faster than this
  says. Prefix caching itself, which is the much larger effect, is fully modeled.

## Check yourself

- Why does a block's hash include its parent's hash? What breaks without it?
- Two prompts share 100 tokens exactly. At `block_size=16`, how many tokens hit?
- Why is a partial tail block never hashed?
- Two requests have byte-identical prompts but different LoRA adapters. What stops them
  sharing blocks, and what would happen if it did not?
- Why must at least one token always be recomputed, and why is that enforced by capping the
  search rather than trimming the result?
- Which is comparable to a real vLLM run: hit rates, hash values, or both?

## Next

[11. Hybrid KV cache groups](11-hybrid-kv-groups.md) — what happens when not every layer
caches the same way.
