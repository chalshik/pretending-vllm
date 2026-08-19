# 05. Your first run

> **Files:** [`pvllm/entrypoints/cli/main.py`](../../pvllm/entrypoints/cli/main.py), [`pvllm/entrypoints/llm.py`](../../pvllm/entrypoints/llm.py), [`pvllm/entrypoints/cli/openai.py`](../../pvllm/entrypoints/cli/openai.py)
> **Prerequisites:** chapter [04](04-repository-tour.md). A Python 3.11+ interpreter. No GPU.

Everything in this chapter was run to produce the output shown. If yours differs, that is
a bug worth reporting — the whole thing is deterministic from a seed.

## Install

```bash
python tools/fetch_upstream.py     # optional: vendors the reference tree (~36 MB, gitignored)
uv venv && uv pip install -e ".[dev]"
pytest -q
```

`fetch_upstream.py` is stdlib-only, so it runs before the package is installed. You only
need it for `tools/spec_sync.py` and for diffing against upstream; the engine and the
tests do not.

Base dependencies are deliberately small: msgspec, pyzmq, fastapi, uvicorn, pydantic,
prometheus-client, numpy. **No torch, no transformers** — chapter
[03](03-simulation-boundary.md) explains why, and `tests/unit/test_purity.py` enforces it.

Two optional extras:

```bash
pip install -e ".[realtok]"   # a real Hugging Face tokenizer (chapter 08)
pip install -e ".[dev]"       # pytest, hypothesis, httpx, ruff, mypy
```

## Run 1: offline generation

The smallest useful thing.

```python
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams

llm = LLM(model="dense-0.6b", max_model_len=2048, trace_path="run.jsonl")
outputs = llm.generate(
    ["hello there", "hello there, friend"],
    SamplingParams(max_tokens=8),
)
for out in outputs:
    print(out.request_id, repr(out.outputs[0].text), out.outputs[0].finish_reason)
```

```
INFO ... [sim_worker.py:169] Model weights loaded in 0.00 seconds [modeled]
INFO ... [sim_worker.py:221] Memory profile: capacity=80.00GiB, usable=73.60GiB,
         weights=1.11GiB, activation_peak=0.77GiB (modeled), non_torch=1.00GiB,
         kv_pool=70.72GiB, num_gpu_blocks=41382, max_concurrency=646.59x
INFO ... [sim_worker.py:262] init engine (profile, create kv cache, warmup model) took
         0.08 seconds (load=0.00s, profile=0.08s, kv_cache=0.00s [70.72GiB],
         graph_capture=0.00s) [modeled]
0 'vugapibasurenanofulobudivupasegi' length
1 'nadipororebikilesekumopakoromole' length
```

Read those startup lines carefully — they are the shape of a real vLLM startup and they
are the project's whole capacity story in three lines:

- **`weights=1.11GiB`** — from the model card's architecture, not from a file on disk.
- **`activation_peak=0.77GiB (modeled)`** — the one term upstream *measures* and this
  estimates. It is labelled everywhere it appears.
- **`num_gpu_blocks=41382`** — how many 16-token KV blocks fit in what is left.
- **`max_concurrency=646.59x`** — how many requests at `max_model_len` those blocks
  serve. This is the number a capacity plan wants.
- **`load=0.00s`** — under the default `constant` cost model, weight loading is free.
  Switch to `--cost-model-profile roofline` and an 8B model takes ~8 modeled seconds.

Note the import path: `pvllm.entrypoints.llm.LLM`. Note also that `finish_reason` is
`length` — the request hit `max_tokens=8` rather than emitting an end-of-sequence token.

## Run 2: read what the engine did

`trace_path="run.jsonl"` recorded every decision. Render it:

```bash
pvllm trace view run.jsonl
```

```
pretending-vllm trace  (upstream 0.27.1, seed 0, clock virtual)
  model='dense-0.6b' card='dense-0.6b' device='datacenter-80gb' block_size=16 cost_model='constant'

  0  #=======  length
  1  #=======  length

  steps=8  tokens=46  preemptions=0  peak_kv=0.0%
  prefix cache: 0/32 tokens (0.0%)
  legend: # prefill  : small prefill  = decode  . waiting  ! preempted  ^ resumed
```

Each row is a request, each column an engine step, and each cell is **the scheduling
decision** for that request in that step. Both requests were prefilled in step 1 (`#`)
and then decoded together for seven steps (`=`). Eight steps for eight tokens: the first
step produced the first token as a by-product of prefilling.

The raw records are JSONL, one per line, so ordinary tools work:

```bash
head -2 run.jsonl | python -m json.tool
grep '"event":"finished"' run.jsonl | wc -l
```

Chapter [20](20-observability.md) covers the schema, the SVG renderer
(`--format svg`), and what each summary line means.

## Run 3: serve the OpenAI API

```bash
pvllm serve --model dense-0.6b --max-model-len 2048 --enable-debug-endpoints
```

```
INFO ... Starting pretending-vllm server: model='dense-0.6b', card='dense-0.6b', ...
WARNING ... This is a SIMULATOR. Generated text is synthetic and latency figures are
            modeled, not measured. See the fidelity contract in the README.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

From another terminal, any OpenAI client works. Here is `curl`, so you can see the exact
bytes:

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"dense-0.6b","prompt":"hello there","max_tokens":6}'
```

```json
{
  "id": "cmpl-00000001",
  "object": "text_completion",
  "created": 1767225600,
  "model": "dense-0.6b",
  "choices": [{"index": 0, "text": "kakokeberunomelosevutefa",
               "logprobs": null, "finish_reason": "length", "stop_reason": null}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 0}}
}
```

Two details to notice, because they are contract, not accident:

- **`created: 1767225600`** is 2026-01-01T00:00:00Z, the fixed epoch of the virtual
  clock. A run is reproducible down to its timestamps. Chapter
  [16](16-clock-and-determinism.md).
- **`prompt_tokens: 12`** for `"hello there"` — eleven characters plus a BOS token. The
  default tokenizer is byte-level, so token counts are much higher than a real model's.
  Chapter [08](08-tokenizers.md) is about fixing that when it matters.

Streaming works the same way, including the `[DONE]` sentinel:

```bash
curl -sN localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"dense-0.6b","prompt":"stream me","max_tokens":3,"stream":true}'
```

```
data: {"id":"cmpl-00000006",...,"choices":[{"index":0,"text":"nuro","finish_reason":null,...}],"usage":null}

data: {"id":"cmpl-00000006",...,"choices":[{"index":0,"text":"luta","finish_reason":null,...}],"usage":null}

data: {"id":"cmpl-00000006",...,"choices":[{"index":0,"text":"pode","finish_reason":"length",...}],"usage":null}

data: [DONE]
```

And with the real OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
for chunk in client.chat.completions.create(
    model="dense-0.6b",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Run 4: poke a running server without writing a client

```bash
pvllm complete -q "hello there" --stats
pvllm chat --system-prompt "Be terse."
```

Both speak the OpenAI API over HTTP, exactly as `vllm complete` does, and use only the
standard library — so they work on a bare install. `--stats` prints TTFT and tokens/sec
as measured **by the client**, which under the default virtual clock is a measurement of
the simulator's own speed, not of the modeled deployment. The modeled numbers are the
ones on `/metrics`.

## Run 5: look inside while it runs

With `--enable-debug-endpoints`:

```bash
curl -s localhost:8000/debug/scheduler | python -m json.tool
```

```json
{
  "step": 7,
  "time": 1767225600.08916,
  "elapsed": 0.08916,
  "clock_mode": "virtual",
  "durations_are_modeled": true,
  "policy": "fcfs",
  "budget": {"max_num_batched_tokens": 8192, "max_num_seqs": 1024,
             "max_num_partial_prefills": 1},
  "running": [],
  "waiting": [],
  "num_preemptions_total": 0,
  "kv_cache_usage": 0.0
}
```

Seven endpoints, all read-only, all off by default because they expose prompt token ids:
`/debug/scheduler`, `/debug/requests`, `/debug/requests/{id}`, `/debug/blocks`,
`/debug/prefix_cache`, `/debug/cost_model`, `/debug/config`. Chapter
[20](20-observability.md).

## Run 6: compare two configurations

The reason the project exists. This costs a GPU reservation otherwise.

```bash
pvllm bench sweep --model meta-llama/Llama-3.1-8B-Instruct \
  --device-card datacenter-80gb \
  --sweep max-num-seqs=1,2,4,8,16 -o sweep.csv
```

One tidy CSV row per cell. And a single point measurement:

```bash
pvllm bench latency --model dense-8b --device-card datacenter-80gb \
  --cost-model-profile roofline --input-len 512 --output-len 32 --batch-size 8
```

```
==============================================================
                Latency: batch=8 in=512 out=32
==============================================================
Successful requests:                                         8
Benchmark duration (s, modeled):                         0.495
Total input tokens:                                       4096
Total generated tokens:                                    256
Request throughput (req/s):                              16.15
Output token throughput (tok/s):                        516.68
Total token throughput (tok/s):                        8783.58
---- Time to First Token -------------------------------------
Mean (ms):                                              302.17
---- Time per Output Token -----------------------------------
Mean (ms):                                                6.24
---- End-to-end Latency --------------------------------------
Mean (ms):                                              495.47
---- Queue Time ----------------------------------------------
Mean (ms):                                                0.00
==============================================================
Durations are MODELED by the simulated cost model, not measured. Treat them as shape, not truth.
Per-iteration modeled duration (s):        [0.4955, 0.4955]
Engine steps:                                               32
```

Look at the shape and ignore the values: TTFT of 302 ms for 4,096 prefill tokens
(compute-bound, linear in tokens), 6.24 ms per output token in a batch of eight
(memory-bound, nearly flat). Both regimes are right. Neither number is your hardware's.
Chapters [15](15-cost-model.md) and [28](28-benchmarking.md).

Note `Queue Time: 0.00` — with a batch of eight against a budget of 8,192 tokens,
nothing ever waited. Raise the batch size or lower the budget and this becomes the
interesting column.

## The knobs you will reach for first

| Flag | Does | Chapter |
|---|---|---|
| `--model` | model name, HF id, or a card path. Default `Qwen/Qwen3-0.6B` | [06](06-configuration.md) |
| `--device-card` | `datacenter-80gb`, `workstation-24gb`, `tiny-2gb`, or a JSON path | [14](14-memory-model.md) |
| `--max-model-len` | context cap. Defaults to the model card's `max_position_embeddings` | [06](06-configuration.md) |
| `--max-num-seqs` / `--max-num-batched-tokens` | the two scheduling budgets | [12](12-scheduler.md) |
| `--cost-model-profile roofline` | realistic-shaped latency instead of constant | [15](15-cost-model.md) |
| `--clock-mode real` | actually spend the modeled time | [16](16-clock-and-determinism.md) |
| `--trace-path run.jsonl` | record every decision | [20](20-observability.md) |
| `--seed` | reproduces an entire run | [16](16-clock-and-determinism.md) |
| `--enable-debug-endpoints` | attach `/debug/*` | [20](20-observability.md) |

Bundled cards, so you have something to point at:

```
models:    dense-0.6b  dense-8b  dense-70b  moe-8x7b
           mla-16b  hybrid-4b  hybrid-ssm-8b  tiny-test
hardware:  datacenter-80gb  workstation-24gb  tiny-2gb
```

Common Hugging Face ids are aliased onto them, so
`--model meta-llama/Llama-3.1-8B-Instruct` resolves to `dense-8b`. An **unknown** id is
an error rather than a guess — silently inventing an architecture would produce capacity
numbers that are fiction *and* unlabelled.

## When something refuses

You will hit this, and it is by design:

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"dense-0.6b","prompt":["a","b"],"max_tokens":2}'
```

```json
{"error": {"message": "Batched prompts are not supported; send one prompt per request.",
           "type": "NotImplementedError", "param": "prompt", "code": 501}}
```

**501 means "this build does not model that feature", by name.** A malformed request is a
400 in vLLM's envelope instead:

```json
{"error": {"message": "1 validation error:\n  {'type': 'missing', 'loc': ('body', 'prompt'), ...}",
           "type": "Bad Request", "param": "body.prompt", "code": 400}}
```

Note it is **not** FastAPI's default 422 with a `{"detail": [...]}` body — matching
vLLM's error shape is conformance class C5. Chapter [19](19-openai-server.md).

## Check yourself

- What are the four numbers in the startup memory line, and which one is modeled?
- Why is `created` always 1767225600 on a fresh run?
- `"hello there"` is 12 prompt tokens here and 2–3 in a real Llama tokenizer. Which three
  downstream numbers does that change?
- What is the difference between a 400, a 501, and a 200 with nonsense text?
- In `pvllm trace view` output, what does a column mean, and what does `#` mean?

## Next

[06. Configuration](06-configuration.md) — how a flag becomes a resolved config, and what
the engine derives for you.
