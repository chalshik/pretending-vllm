# 27. Pooling and embeddings

> **Files:** [`pvllm/pooling_params.py`](../../pvllm/pooling_params.py), [`pvllm/entrypoints/pooling/embed/`](../../pvllm/entrypoints/pooling/embed), `embed()` in [`pvllm/sim/model.py`](../../pvllm/sim/model.py), the pooling branches in [`sched/scheduler.py`](../../pvllm/v1/core/sched/scheduler.py) and [`worker/gpu/model_runner.py`](../../pvllm/v1/worker/gpu/model_runner.py)
> **Upstream:** `vllm/pooling_params.py`, `vllm/entrypoints/pooling/embed/*` (Tier B/C)
> **Prerequisites:** chapters [12](12-scheduler.md), [19](19-openai-server.md).

An embedding request runs the prompt through the model and returns a **vector** instead of generating
tokens. From the engine's point of view that is one difference — **it prefills and stops** — and
everything else about it is the ordinary path.

Which is exactly why it is worth modeling: an embedding workload's *capacity* behaviour is a real
question, and it is a different shape from a generation workload's.

## The one difference

```python
self.pooling_params = pooling_params
if (sampling_params is None) == (pooling_params is None):
    raise ValueError("exactly one of sampling_params and pooling_params must be set")
```

A pooling request has **no sampling params and no `max_tokens`**. It generates nothing, so:

- it occupies exactly as many steps as its prompt needs to prefill (one, unless chunked);
- it is never in `sampling_indices` in the runner, so the sampling path and the pooling path cannot
  interfere;
- it finishes the moment its whole prompt is computed.

```python
if request.use_pooling:
    index = model_runner_output.req_id_to_index.get(req_id)
    pooling_output = (model_runner_output.pooler_output or [])[index]
    if pooling_output is not None:
        request.status = RequestStatus.FINISHED_STOPPED
        stopped = True
```

Note the chunked-prefill subtlety in the runner:

```python
if int(input_batch.seq_lens_np[pool_idx]) < int(self.req_states.prompt_len[slot]):
    continue  # the prompt is not all in yet, so there is nothing to pool over
```

Under chunked prefill a long document takes several steps, and **the vector only exists on the last of
them**. A request that produced no vector this step is not finished.

## Everything else is the ordinary path

> Queueing, block allocation, prefix caching, chunked prefill — everything else about it is the
> ordinary path, which is the point of modelling it at all.

So a page of documents sent to `/v1/embeddings`:

- **batches** — many short prefills in one step, up to `max_num_batched_tokens`;
- **queues** — bounded by `max_num_seqs` and by the KV pool like anything else;
- **shares a preamble through the prefix cache** — if your documents carry a common instruction
  prefix ("Represent this document for retrieval: …"), it is cached exactly as a chat system prompt
  would be;
- **chunks** — a document longer than the step budget is split;
- **can be preempted** — though it rarely is, because it holds its blocks for so few steps.

The capacity shape is different from generation in one important way: embedding requests are almost
all prefill, so an embedding workload is **compute-bound** where a chat workload is memory-bound. That
means the knee is in a different place, and `max_num_batched_tokens` matters more than `max_num_seqs`.

## The vector is synthetic — and this warning is the important part

```python
def embed(self, prompt_token_ids: list[int], dimensions: int) -> list[float]:
    digest = hashlib.sha256(
        b"embed" + b"".join(t.to_bytes(4, "little") for t in prompt_token_ids)
    ).digest()
    # counter-mode expansion, then L2-normalise
```

What it guarantees:

- **the same text always embeds to the same vector**, in one process and across runs;
- **different text embeds differently**;
- the vector is **L2-normalised** and has the width a real pooler would emit (the model card's hidden
  size, or `dimensions` if requested).

Those three properties are what a product's plumbing, caching, and dedup logic actually depend on.

What it does **not** carry, in the source's words:

> It carries no semantic information. Cosine similarity between two of these says nothing about
> whether the texts are related, because there is no model here to say so. **Anything evaluating
> retrieval quality against these numbers is measuring a hash function.**

```bash
python -c "
import math
from pvllm.entrypoints.llm import LLM
llm = LLM(model='dense-0.6b', max_model_len=1024)
v = [o.outputs.data for o in llm.embed(['the first document', 'the second document', 'the first document'])]
print('dims                :', len(v[0]))
print('L2 norm             :', round(math.sqrt(sum(x*x for x in v[0])), 6))
print('same text, same vec :', v[0] == v[2])
print('diff text, diff vec :', v[0] != v[1])
print('cosine(doc0, doc1)  :', round(sum(a*b for a, b in zip(v[0], v[1])), 4), '<- meaningless')
"
```

```
dims                : 1024
L2 norm             : 1.0
same text, same vec : True
diff text, diff vec : True
cosine(doc0, doc1)  : 0.0216 <- meaningless
```

Note the vector is derived **from the content**, not drawn from an RNG stream: "a pooling model's
output depends on its input and nothing else. The same prompt embedded twice in one process, or across
two runs, must produce the same vector."

## The API

```bash
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"dense-0.6b","input":["first document","second document"]}'
```

Standard OpenAI shape: `object: "list"`, a `data` array of `{object, index, embedding}`, and `usage`
with `prompt_tokens`. Offline:

```python
llm.embed(["first document", "second document"])  # -> list[PoolingRequestOutput]
```

`PoolingRequestOutput` carries `outputs: PoolingOutput(data=[...])` plus `prompt_token_ids` — no
`CompletionOutput`, no `finish_reason` semantics beyond finished.

**Each document is its own engine request.** That is why a page of them batches and queues rather than
being one big request, and it is what makes the capacity behaviour realistic.

## Tasks: what is and is not supported

```python
PoolingTask = Literal["embed", "encode", "classify", "score"]
SUPPORTED_TASKS = ("embed", "encode")
```

`classify` and `score` are refused by name:

```
NotImplementedError: pooling task 'classify' needs a classification head, which has no simulated
counterpart: its output is a label distribution over labels this engine does not have.
['embed', 'encode'] are supported.
```

An embedding can be a normalised hash and still serve its purpose. A label distribution cannot —
inventing one would be inventing a model, and a product would read the labels as predictions.

## What this is good for

**Good:** testing that your embedding pipeline handles batching, backpressure, and partial failures;
measuring how many documents per second an engine configuration would sustain; checking that a shared
instruction prefix is being cached; verifying your client's handling of `usage.prompt_tokens` and of a
`dimensions` request.

**Not good for:** anything involving similarity. Do not evaluate retrieval, do not tune a threshold, do
not build a clustering test. The numbers are stable and distinct and that is all.

## Check yourself

- What single field distinguishes a pooling request from a generation request in the engine core?
- Under chunked prefill, when does a pooling request produce its vector?
- Why is an embedding workload compute-bound where a chat workload is memory-bound?
- Which three properties of the synthetic vector are real, and which property is absent?
- Why is `classify` refused when `embed` is not?

## Next

[28. Benchmarking and sweeps](28-benchmarking.md).
