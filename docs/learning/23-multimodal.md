# 23. Multimodal

> **Files:** [`pvllm/multimodal/inputs.py`](../../pvllm/multimodal/inputs.py), [`pvllm/v1/core/encoder_cache_manager.py`](../../pvllm/v1/core/encoder_cache_manager.py), [`pvllm/entrypoints/openai/multimodal.py`](../../pvllm/entrypoints/openai/multimodal.py), the encoder phase in [`sched/scheduler.py`](../../pvllm/v1/core/sched/scheduler.py), `ENCODER_PARAMS` in [`sim/cost_model.py`](../../pvllm/sim/cost_model.py)
> **Upstream:** `vllm/multimodal/inputs.py` (Tier C), `vllm/v1/core/encoder_cache_manager.py` (Tier **A**), `vllm/entrypoints/chat_utils.py` (the parsing half, Tier B)
> **Prerequisites:** chapters [10](10-prefix-caching.md), [12](12-scheduler.md).

**The engine never sees pixels.** It sees how many tokens an image occupies, where they sit in the
prompt, and a hash identifying the content — which is exactly the information scheduling and caching
need, and exactly what a simulator can carry faithfully.

## Placeholders

A multimodal request is a token sequence with a run of reserved ids standing in for where the image's
embeddings will go:

```
[BOS, "describe", " this", ":", <PH>, <PH>, ... <PH>, "\n"]
                                └── 256 placeholder tokens ──┘
```

```python
@dataclass(frozen=True)
class MultiModalFeatureSpec:
    identifier: str  # content hash — the identity every cache in the engine uses
    modality: str  # image | audio | video  (only image is modeled)
    position: int  # index of the first placeholder token
    length: int  # prompt tokens the item occupies
    num_embeds: int  # encoder output embeddings — what the encoder budget counts
```

`PLACEHOLDER_TOKEN_ID = 4` sits *below* the mock tokenizer's byte range, in the block of special ids
no text encodes to, so a placeholder can never be confused for content. In a real deployment the
placeholder id comes from the model's config; here it is fixed so a trace is readable.

`content_hash` is sha256 over the bytes **plus the modality**, so the same bytes offered as an image
and as a video frame are distinct entries — the encoder produces different embeddings for them, and a
cache that conflated the two would serve the wrong ones. And sha256 rather than `hash()` because the
identifier reaches the prefix cache, where a per-process salt would make cache behaviour
irreproducible.

`MAX_TOKENS_PER_MM_ITEM = 256` is **load-bearing, not informational**: an item larger than the encoder
cache can never be scheduled, and the scheduler's response to "cannot be scheduled" is to try again
next step, forever. So `SchedulerConfig` floors both encoder budgets with it.

## The separate budget

Encoder work and decoder work **do not trade against each other**: an image costs vision-encoder time
whatever the token budget is doing. So the scheduler spends a second budget:

```python
encoder_budget = self.max_num_encoder_input_tokens  # defaults to the token budget,
# floored at MAX_TOKENS_PER_MM_ITEM
```

`_schedule_encoder` runs for both running and waiting requests, and its return type tells you the
whole design:

```python
def _schedule_encoder(
    self, request, num_new_tokens, encoder_budget
) -> tuple[int, list[int], int]:
    """Returns (num_new_tokens, input_ids, encoder_budget).

    `num_new_tokens` may come back *smaller*: if an image cannot be encoded this step, the
    request is trimmed to stop just before its first placeholder rather than being blocked
    entirely."""
```

**Trimmed, not blocked.** A request whose image will not fit this step makes progress on the *text*
instead of stalling. Scheduling past a placeholder whose embeddings do not exist would mean attending
to KV that was never written.

Two placement details that were bugs before they were rules:

- The encoder trim happens **before** `allocate_slots` in the admission path, where upstream puts it.
  After it, the trim happens once blocks are already allocated and published — "the prefix cache would
  then hold blocks for KV this step never computes, and the next request with the same prefix would hit
  on them."
- A preempted request's encoder reservation is **released**, off both the budget and the step's encoder
  list. Left in place, "the step was charged vision-encoder time for a request it did not run, and the
  runner was told to encode an image for a request with no slot."

## The encoder cache

Vision encoders are expensive and their output is reusable: the same image in two requests produces
the same embeddings, and the same image in one request produces them once however many steps the
prompt takes to prefill. `EncoderCacheManager` (Tier **A**) caches them by content hash, with a budget
measured in embeddings.

Three behaviours, each a real source of observable scheduling:

**Eviction is deferred.** Freeing a request's reference moves the entry to `freeable`; the embeddings
stay resident until someone needs the space. "A request arriving with an image another request just
finished with gets a hit, which is the common case in a chat workload and would be lost by eager
eviction."

**Eviction is oldest-first among unreferenced entries**, so the cache behaves like an LRU over
completed work rather than discarding whatever is convenient.

**`can_allocate` mutates.** Upstream's does too, "and the name is a lie in both trees" — it evicts to
make room and reports whether it succeeded. Splitting it would mean walking the eviction candidates
twice per input per step.

References are freed when the placeholders are fully computed, not at request end:

```python
if feature.position + feature.length <= request.num_computed_tokens:
    self.encoder_cache_manager.free_encoder_input(request, input_id)
```

Once the placeholder run is behind `num_computed_tokens`, the embeddings are in the decoder KV cache
and nobody needs them. Holding them to the end of the request "turned the encoder cache into a
per-request reservation — a request whose images together exceeded it could never schedule the second
one, and retried forever."

And the eviction *notice* travels to the worker through `SchedulerOutput.free_encoder_mm_hashes`,
which the runner reads to drop what it holds. That field was carried and read by nobody once, so "the
worker's set only ever grew — the leak the eviction protocol exists to prevent."

## The prefix cache interaction

From chapter [10](10-prefix-caching.md), the multimodal extra keys:

```python
keys.extend(
    (feature.identifier, feature.position - start_token_idx)
    for feature in request.mm_features
    if feature.position < end and feature.position + feature.length > start_token_idx
)
```

Two subtleties, both about getting the hit rate *right* rather than merely safe:

- **Only the images this block overlaps.** Folding every one of a request's images into every block's
  key would partition the text *before* the first image too — so two prompts sharing a long system
  prompt and differing only in a later image would share nothing, and the reported hit rate would be
  far below a real deployment's. "C3 calls hit rate exact, so over-partitioning is as wrong as
  under-partitioning; it is just wrong in the safe direction."
- **Plus the item's offset within the block.** Blocks of pure placeholder tokens are byte-identical
  whatever produced them, so without the offset a block covering image A's tail and image B's head
  hashes the same as one covering a different split of the same pair — and the second request reads KV
  computed for a different layout.

## The cost

```python
ENCODER_PARAMS = 300_000_000  # ViT-L/14 scale

if profile.num_encoder_embeds:
    encoder_flops = 2.0 * ENCODER_PARAMS * profile.num_encoder_embeds
    t_encoder = encoder_flops / (device.mfu * device.peak_flops)
```

A separate term, not folded into compute — see chapter [15](15-cost-model.md) for why (it would flip
the `bound_by` verdict on any step carrying an image).

A **count** rather than a multiple of `hidden_size`, because "a vision tower does not grow with the
language model it is bolted to". The earlier form modeled a 256-patch image at a tenth of a
microsecond — free, against a 20 ms step:

> An image was documented as expensive and priced at nothing, which is the one direction a cost model
> must not be wrong in when the question is whether to cache encoder output at all.

Read it as "an image costs roughly one short prefill", not as a measurement.

## Over HTTP

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dense-0.6b",
  "messages": [{"role": "user", "content": [
      {"type": "text", "text": "what is in this image?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
  ]}],
  "max_tokens": 16
}'
```

[`openai/multimodal.py`](../../pvllm/entrypoints/openai/multimodal.py) turns the content parts into a
placeholder run plus a `MultiModalFeatureSpec`. The URL's *bytes* are hashed for the identifier;
nothing is fetched or decoded.

So what a product gets to exercise: **256 placeholder tokens, a separate encoder budget, an encoder
pass priced at ViT-L scale, and a cache the second request with the same image hits.** All of the
scheduling, none of the pixels.

## Try it

```python
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
from pvllm.multimodal.inputs import MultiModalFeatureSpec, content_hash

llm = LLM(model="dense-0.6b", max_model_len=2048, trace_path="mm.jsonl")
# submit two requests carrying the same image identifier via
# llm_engine.add_request(..., mm_features=[MultiModalFeatureSpec(...)])
```

Then read the two counters either side of the encoder cache:

```bash
curl -s localhost:8000/metrics | grep mm_cache      # queries vs hits
```

`tests/v1/test_multimodal.py` (412 lines) is the readable version: it asserts the trim behaviour, the
deferred eviction, the cache hit across requests, and the budget accounting.

## Check yourself

- What does the engine actually receive in place of an image?
- Why does the encoder get its own budget rather than sharing the token budget?
- A request's image cannot be encoded this step. What happens to the request?
- Why does the encoder trim have to happen before `allocate_slots`?
- Why do the block hash keys include the item's *offset within the block*?
- When is an encoder cache entry's reference released, and why not at request end?

## Next

[24. Parallelism](24-parallelism.md) — TP, PP, DP, EP, and what each one actually buys.
