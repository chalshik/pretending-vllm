# 01. LLM inference fundamentals

> **Files:** none yet — this chapter is concepts. The next one maps them onto code.
> **Prerequisites:** chapter [00](00-orientation.md).

Everything vLLM does is a response to four facts about transformer inference. If you
understand these four, the rest of the engine reads as inevitable rather than arbitrary.

## Fact 1: generation is sequential, one token at a time

A language model maps a sequence of tokens to a probability distribution over the next
token. To produce 100 tokens of output you run the model 100 times, each time feeding
back what it just produced. There is no way to compute token 50 without having computed
token 49 first.

```
prompt: [The, capital, of, France, is]
        → run model → " Paris"
        [The, capital, of, France, is, Paris]
        → run model → "."
        [The, capital, of, France, is, Paris, .]
        → run model → <eos>  → stop
```

Two consequences: **latency is unavoidably proportional to output length**, and the only
way to make a server efficient is to run many requests' steps *together*.

## Fact 2: attention would redo all previous work, so we cache it

Inside each transformer layer, every token attends to every token before it. Mechanically
each token produces a **key** and a **value** vector per layer, and computing attention
for the newest token requires the keys and values of every earlier token.

Recomputing those from scratch at each step would make generating token *n* cost O(n)
work, and the whole generation O(n²). So they are computed once and kept: the **KV
cache**.

The cache is large. For one token:

```
kv_bytes_per_token = 2 (key and value)
                   × num_key_value_heads
                   × head_dim
                   × bytes_per_element
                   × num_attention_layers
```

For a Llama-3.1-8B-class model (32 layers, 8 KV heads, head dim 128, bfloat16):

```
2 × 8 × 128 × 2 × 32 = 131,072 bytes = 128 KiB per token
```

You can read that number straight out of this repository's model cards:

```bash
python -c "
from pvllm.sim.model_db import load_model_card
c = load_model_card('dense-8b')
print(c.kv_bytes_per_token(), 'bytes/token')
print(c.kv_bytes_per_token() * 8192 / 2**30, 'GiB for an 8k context')
"
```

```
131072 bytes/token
1.0 GiB for an 8k context
```

**One request at 8k context costs a gigabyte of GPU memory just to remember its own
past.** That is the fact around which the entire engine is designed. Chapter
[09](09-kv-cache-blocks.md) is about managing it; chapter [14](14-memory-model.md) is
about how much of it fits.

## Fact 3: the two phases have opposite performance characters

A request's life has two shapes, and the engine's behaviour differs sharply between them.

**Prefill** processes the whole prompt at once. A 2,000-token prompt is 2,000 tokens
through the model in a single pass. Lots of arithmetic, weights read once — so it is
**compute-bound**, and its duration is roughly linear in prompt length.

**Decode** processes exactly one new token per request per step. Almost no arithmetic,
but the entire weight set still has to be read from memory to do it — so it is
**memory-bound**, and its duration is nearly flat regardless of how many requests are in
the batch, until KV traffic grows large enough to dominate.

| | prefill | decode |
|---|---|---|
| tokens per request per step | many (the whole prompt) | 1 |
| bottleneck | compute (FLOPs) | memory bandwidth (weight reads) |
| duration vs. tokens | roughly linear | roughly flat |
| what a bigger batch does | little (already saturated) | nearly free throughput |

Two practical consequences that drive the whole design:

- **Batching decode is nearly free.** If reading the weights costs the same whether one
  request or sixty-four are decoding, you should decode sixty-four. This is why
  throughput scales with batch size and why "how many requests fit in the KV cache" is
  the single most important capacity number.
- **A long prefill blocks everyone.** If a step is spent prefilling a 30,000-token
  prompt, every other request's next token waits for it. This is what **chunked
  prefill** exists to fix — chapter [12](12-scheduler.md).

> **In modern vLLM there is no prefill phase and no decode phase.** This is worth saying
> loudly because it is the biggest conceptual difference from older descriptions. V1's
> scheduler tracks only `num_computed_tokens` and `num_tokens` per request, and each step
> hands out tokens so the former catches up to the latter. A request with a 500-token
> prompt and nothing computed asks for 500; a request mid-generation asks for 1. Prefill
> and decode are *descriptions of what a step happened to do*, not modes the engine is
> in. Chunked prefill, prefix caching, and speculative decoding all fall out of that one
> idea instead of needing their own special cases.

## Fact 4: requests arrive and finish at different times

A naive server batches requests together and runs them to completion as a group. That
wastes almost everything: the group runs until its *longest* member finishes, and any
request that arrives one millisecond after the batch starts waits for the whole thing.

vLLM does **continuous batching** (also called iteration-level scheduling): before every
single model step, it re-decides which requests to run. A request that finishes leaves
the batch immediately and its memory is reclaimed; a request that arrives is admitted at
the next step boundary.

```
step:      1     2     3     4     5     6     7
req A:  prefill  dec   dec   dec   done
req B:           prefill dec  dec   dec   done
req C:                        prefill dec  dec ...
```

That is why the engine's core loop is `schedule → execute → update`, forever, and why
the scheduler is the most consequential file in the project.

## The four techniques that follow from these facts

Every one of these is real logic in this repository, and each gets its own chapter.

### PagedAttention: the KV cache as pages, not contiguous arrays

Reserving `max_model_len` tokens of KV per request means a request that generates 20
tokens against an 8k limit wastes 99.8% of its reservation. So the KV cache is a pool of
fixed-size **blocks** (16 tokens each, by default), and a request holds a *list* of block
ids — its **block table** — grown one block at a time as it generates.

This is the idea the original vLLM paper introduced, and it is the reason vLLM exists.
It converts internal fragmentation into an indirection. → chapter
[09](09-kv-cache-blocks.md)

### Prefix caching: two requests with the same prefix share blocks

Once KV lives in shareable blocks, two requests whose prompts start identically can
*point at the same physical blocks*. A 500-token system prompt shared by every request in
your product gets computed once, not once per request.

The mechanism is content hashing with a chain: block *n*'s hash includes block *n−1*'s
hash, so a hit means "everything before this point matched too". → chapter
[10](10-prefix-caching.md)

You can watch it happen right now:

```bash
python -c "
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams
system = 'You are a helpful assistant. ' * 20
llm = LLM(model='dense-0.6b', max_model_len=2048)
for out in llm.generate([system + 'What is one?', system + 'What is two?'],
                        SamplingParams(max_tokens=4)):
    print('prompt_tokens =', len(out.prompt_token_ids), ' cached =', out.num_cached_tokens)
"
```

```
prompt_tokens = 593  cached = 0
prompt_tokens = 593  cached = 576
```

The second request paid for 17 tokens of prefill instead of 593.

The exact number is instructive. The two prompts share 589 tokens (the system prompt
plus `"What is "`), but caching happens in whole 16-token blocks, and 589 does not divide
by 16. So the match is truncated down to 36 complete blocks — 576 tokens — and the block
that straddles the divergence has to be recomputed. **Block granularity always rounds a
shared prefix down.** Chapter [10](10-prefix-caching.md) covers this and the related rule
that at least one token is always recomputed.

### Chunked prefill: a long prompt is spread over several steps

If a 30,000-token prompt cannot be allowed to monopolise a step, split it: process 8,192
tokens this step, 8,192 the next, and let decodes ride along in the same batches.
`num_computed_tokens` makes this trivial — the request is simply given fewer tokens than
it wants. → chapter [12](12-scheduler.md)

### Preemption: when memory runs out, someone gives their blocks back

The KV pool is finite and requests grow. When a running request needs another block and
none is free, the scheduler picks a victim, frees *all* of its blocks, and puts it back
at the front of the waiting queue. Its generated tokens are kept, so it resumes
mid-generation — but everything it had computed is recomputed. This is **preemption by
recompute**, and it is the engine's back-pressure valve. → chapter
[12](12-scheduler.md)

## The numbers a serving engine is judged on

Vocabulary you will need in every later chapter:

| Metric | Meaning | Dominated by |
|---|---|---|
| **TTFT** (time to first token) | arrival → first token delivered | queue wait + prefill |
| **ITL** (inter-token latency) | gap between consecutive output tokens | decode step time, and batch size |
| **TPOT** | time per output token, averaged over a request | same as ITL |
| **E2E latency** | arrival → final token | TTFT + output_len × ITL |
| **Throughput** | output tokens/second across all requests | how many requests fit in KV |
| **Goodput** | throughput that met a latency target | both of the above |

The central tension of LLM serving: **throughput and latency trade against each other
through batch size.** A bigger batch decodes more tokens per step (throughput up) but
each step takes slightly longer and prefills queue behind each other (latency up). Every
knob in chapter [06](06-configuration.md) is somewhere on that curve, and chapter
[28](28-benchmarking.md) is how you find the knee.

`pvllm bench serve` splits TTFT into **queue wait** and **prefill** for exactly this
reason: a request that waited 200 ms and prefilled in 5 ms needs more concurrency, while
one that prefilled for 205 ms needs a smaller batch. The same total, opposite fixes.

## Check yourself

- Why is decode memory-bound while prefill is compute-bound?
- A request has a 1,000-token prompt and generates 200 tokens. Roughly how many times
  does the model run for it?
- Why does reserving `max_model_len` of KV per request waste memory, and what replaces it?
- If two requests share a 500-token prefix but differ at token 501, how many blocks can
  they share at a block size of 16?
- The engine has no prefill phase. What single pair of counters replaces it?

## Next

[02. The vLLM V1 architecture](02-vllm-v1-architecture.md) — the layers that implement
all of this, and who owns what.
