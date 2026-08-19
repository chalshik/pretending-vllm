# 08. Tokenizers and detokenization

> **Files:** [`pvllm/tokenizers/`](../../pvllm/tokenizers), [`pvllm/v1/engine/detokenizer.py`](../../pvllm/v1/engine/detokenizer.py), [`pvllm/entrypoints/serve/tokenize/serving.py`](../../pvllm/entrypoints/serve/tokenize/serving.py)
> **Upstream:** `vllm/tokenizers/*` (Tier C/D), `vllm/tokenizers/detokenizer_utils.py` (Tier **A**), `vllm/v1/engine/detokenizer.py` (Tier **A**)
> **Prerequisites:** chapter [07](07-requests-and-sampling.md).

Tokenization looks like a detail until you notice that **three numbers a product reads off
the engine depend on it exactly**: `usage.prompt_tokens`, the prompt length at which the
context-length error fires, and where the prefix cache's block boundaries fall.

## Two tokenizers, and when each is right

| Mode | Tokenizer | Needs | Use when |
|---|---|---|---|
| `auto` (default), `mock` | `MockTokenizer` — byte-level | nothing | exercising plumbing: streaming, cancellation, queueing, metrics |
| `slow` | `HFTokenizer` — the real thing | `pip install 'pretending-vllm[realtok]'` | any time a token *count* matters |

```bash
pvllm serve --model dense-0.6b --tokenizer Qwen/Qwen2.5-0.5B-Instruct --tokenizer-mode slow
```

That loads the tokenizer your model actually ships — a local `tokenizer.json`, a local
directory, or a Hub id — and renders its own chat template, so every token count downstream
becomes the one real vLLM would report. A Hub id fetches over the network on first use,
which is why it is opt-in.

The generated *text* is still synthetic either way: ids are drawn, not inferred. A real
tokenizer makes the output look like language without making it mean anything.

## The mock tokenizer, and why it is byte-level

[`pvllm/tokenizers/mock.py`](../../pvllm/tokenizers/mock.py), Tier D.

```
ids 0..3      special: BOS=0, EOS=1, PAD=2, UNK=3
ids 256..511  the 256 byte values: byte b → 256 + b
ids 512+      never produced by encode(); valid sampler outputs, rendered as pseudowords
```

**The design constraint is exact reversibility**: `decode(encode(text)) == text` for any
input, not approximately. Two things depend on it:

- generated content must detokenize to *stable* text so HTTP responses can be
  golden-tested;
- incremental detokenization must be genuinely exercised, and you cannot test that against
  a tokenizer whose round trip loses information.

Byte-level gets all of it for free: every possible string is representable, there is no
unknown token ever, and multi-byte characters split across token boundaries exactly the way
a real BPE tokenizer's do — so the partial-UTF-8 handling in the detokenizer is really
tested.

What it deliberately is *not* is realistic. Token counts per word are far higher than a
real BPE tokenizer's:

```bash
python -c "
from pvllm.tokenizers import get_tokenizer
tok = get_tokenizer('dense-0.6b', vocab_size=151936)
print(tok.encode('hello world'))
print(len(tok.encode('hello world')), 'tokens (a real Llama tokenizer: 2-3)')
print(repr(tok.decode(tok.encode('hello world'))))
"
```

```
[0, 360, 357, 364, 364, 367, 288, 375, 367, 370, 364, 356]
12 tokens (a real Llama tokenizer: 2-3)
'<s>hello world'
```

Eleven bytes plus a BOS. That is why the `realtok` extra is **mandatory for conformance
class C3**: prefix cache hit rates on real text depend on exact tokenization, and a
byte-level mock will not reproduce them.

`vocab_size` comes from the model card, so `max_token_id` and the logprobs schema report
what the model would — a client validating that a sampled id is in range gets the right
answer even though the id itself is meaningless.

## Incremental detokenization

A streaming server cannot detokenize the whole sequence on every token — that is O(n²) —
and it cannot detokenize each token in isolation either, because tokenizers are contextual
and a multi-byte character can span two tokens. So detokenization is **incremental**:
state is kept per request, and each new token yields a delta of text.

[`pvllm/tokenizers/detokenizer_utils.py`](../../pvllm/tokenizers/detokenizer_utils.py) is
Tier **A** — a line-for-line port, because getting this subtly wrong produces mojibake in a
client's stream and nowhere else.

## Stop strings: the part that surprises people

Token-level stops (`eos_token_id`, `stop_token_ids`) are checked in the scheduler
(chapter [07](07-requests-and-sampling.md)). Stop *strings* need text, so they live in
[`pvllm/v1/engine/detokenizer.py`](../../pvllm/v1/engine/detokenizer.py) — and they have
three behaviours a stream-consuming product will notice if they are wrong.

### 1. Text is held back from the stream

```python
self.stop_buffer_length = (
    max(len(s) for s in self.stop) - 1
    if self.stop and not self.include_stop_str_in_output
    else 0
)
```

With a stop string configured and `include_stop_str_in_output` off, the last
`max(len(stop)) - 1` characters are **withheld** from the stream. They might turn out to be
the beginning of a stop string, and streaming them and retracting later is impossible over
SSE.

So a request with `stop=["\n\nHuman:"]` streams eight characters behind the model. Once the
request finishes, nothing is held back — no further token can extend a partial match:

```python
buffer_length = 0 if finished else self.stop_buffer_length
```

This is real vLLM behaviour, and it is the sort of thing you only find out by testing
against an engine that reproduces it.

### 2. `min_tokens` suppresses matching

Text produced below `min_tokens` is excluded from stop matching by moving the search window
past it. A stop string appearing early cannot end the request — consistent with
`check_stop`'s early return.

### 3. The earliest match wins

When several tokens arrive at once (routine under speculative decoding) and more than one
stop string matches, the one **completing earliest** is chosen:

```python
end = stop_index + stop_string_len
if end < best_end:
    best_stop_str, best_stop_index, best_end = stop_str, stop_index, end
```

so the result matches what appending one token at a time would have produced. Ties break on
the order of the stop list. And the search starts far enough back
(`1 - new_char_count - stop_string_len`) that a stop string straddling the boundary between
old and new text is still found, without rescanning text already checked.

## What each layer knows

| Layer | Sees | Decides |
|---|---|---|
| `InputProcessor` | text or token ids | tokenize, validate length, resolve `max_tokens` |
| `Scheduler` | token ids only | EOS, stop token ids, length caps |
| `Detokenizer` | tokens → text, incrementally | stop *strings*, held-back text |
| `OutputProcessor` | the text deltas | what the client actually receives |

The split is upstream's, and it exists because the scheduler must not need text. Text
requires a tokenizer, a tokenizer requires state per request, and the scheduler's job is
already the hottest loop in the engine.

## `/tokenize` and `/detokenize`

The server exposes vLLM's own two utility endpoints
([`serve/tokenize/serving.py`](../../pvllm/entrypoints/serve/tokenize/serving.py)):

```bash
curl -s localhost:8000/tokenize   -H 'Content-Type: application/json' \
  -d '{"model":"dense-0.6b","prompt":"hello"}'
curl -s localhost:8000/detokenize -H 'Content-Type: application/json' \
  -d '{"model":"dense-0.6b","tokens":[0,360,357,364,364,367]}'
```

Useful for exactly one thing: checking that your client's idea of a prompt's token count
matches the engine's before you trust a `max_model_len` calculation.

## Try it

Watch text get held back, then released:

```python
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams

llm = LLM(model="dense-0.6b", max_model_len=1024)
out = llm.generate(["x"], SamplingParams(max_tokens=20, stop=["zzz"]))[0]
print(repr(out.outputs[0].text), out.outputs[0].finish_reason, out.outputs[0].stop_reason)
```

And compare token counts across tokenizer modes, if you have the extra installed:

```bash
pip install -e '.[realtok]'
python -c "
from pvllm.tokenizers import get_tokenizer
mock = get_tokenizer('dense-0.6b', tokenizer_mode='mock', vocab_size=151936)
print('mock:', len(mock.encode('The capital of France is Paris.')))
"
```

## Check yourself

- Why is the mock tokenizer byte-level rather than word-level?
- Which three product-visible numbers change when you switch to a real tokenizer?
- A request has `stop=["\n\nHuman:"]`. How many characters lag the stream, and why?
- Why are stop strings handled in a different layer from `stop_token_ids`?
- Two stop strings match in the same multi-token step. Which one wins, and why that rule?

## Next

[09. KV cache blocks](09-kv-cache-blocks.md) — the bookkeeping that PagedAttention is
actually made of.
