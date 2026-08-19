# 19. The OpenAI server

> **Files:** [`pvllm/entrypoints/openai/api_server.py`](../../pvllm/entrypoints/openai/api_server.py), [`openai/completion/`](../../pvllm/entrypoints/openai/completion), [`openai/chat_completion/`](../../pvllm/entrypoints/openai/chat_completion), [`openai/responses/`](../../pvllm/entrypoints/openai/responses), [`serve/utils/`](../../pvllm/entrypoints/serve/utils)
> **Upstream:** the same paths under `vllm/entrypoints/` (Tier B/C)
> **Prerequisites:** chapter [17](17-engine-core-and-frontends.md).
> **Contract:** C5 (HTTP schemas and errors) and C7 (failure modes at capacity) — both **exact**.

This is the surface a product actually points at, so the routes, their shapes, and **their
errors** are contract rather than convenience.

## The routes

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/completions` | streaming and non-streaming |
| POST | `/v1/chat/completions` | applies a chat template; multimodal content parts |
| POST | `/v1/responses` | the stateful surface — named SSE events, no `[DONE]` |
| GET | `/v1/responses/{id}` | **404 unless the store is enabled** (see below) |
| POST | `/v1/responses/{id}/cancel` | same |
| POST | `/v1/embeddings` | chapter [27](27-pooling-and-embeddings.md) |
| GET | `/v1/models` | includes LoRA adapters as their own models |
| POST | `/tokenize`, `/detokenize` | chapter [08](08-tokenizers.md) |
| GET | `/health`, `/ping` | readiness |
| GET | `/version` | |
| GET | `/metrics` | Prometheus, chapter [20](20-observability.md) |
| POST | `/reset_prefix_cache` | |
| GET | `/debug/*` | off by default, chapter [20](20-observability.md) |

## Requests, in the exact bytes

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"dense-0.6b","prompt":"hello there","max_tokens":6}'
```

```json
{
  "id": "cmpl-00000001", "object": "text_completion", "created": 1767225600,
  "model": "dense-0.6b",
  "choices": [{"index": 0, "text": "kakokeberunomelosevutefa",
               "logprobs": null, "finish_reason": "length", "stop_reason": null}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 0}}
}
```

Three fields worth pointing at:

- **`stop_reason`** — a vLLM extension, not OpenAI's. It carries the token id or stop string
  that ended the request. Products targeting vLLM read it.
- **`prompt_tokens_details.cached_tokens`** — how much the prefix cache saved. This is the
  field to assert on in a prompt-engineering test.
- **`created: 1767225600`** — the virtual clock's fixed epoch (chapter
  [16](16-clock-and-determinism.md)).

Chat is the same engine with a template in front:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dense-0.6b","messages":[{"role":"user","content":"hi"}],"max_tokens":3}'
```

```json
{"id": "chatcmpl-00000001", "object": "chat.completion", "created": 1767225600,
 "model": "dense-0.6b",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "nigiborumazo"},
              "logprobs": null, "finish_reason": "length", "stop_reason": null}],
 "usage": {"prompt_tokens": 27, "completion_tokens": 3, "total_tokens": 30,
           "prompt_tokens_details": {"cached_tokens": 0}}}
```

27 prompt tokens for `"hi"` — the chat template's role markers, tokenized byte by byte.

## Streaming

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

Deltas, `finish_reason` on the last chunk only, then the `[DONE]` sentinel. `stream_options`
controls whether a final usage chunk is included.

**Streaming is where this project earns its keep**, because the streaming behaviours a product
must handle are all real here:

- deltas arrive as the engine steps, not all at once at the end (the output handler yields to the
  event loop each step — chapter [17](17-engine-core-and-frontends.md));
- **cancellation** frees KV blocks within one step: close the connection mid-stream and the
  scheduler drops the request;
- with a stop string configured, text lags the stream by `max(len(stop)) - 1` characters
  (chapter [08](08-tokenizers.md));
- under `--clock-mode real`, chunks arrive at modeled intervals, so your client's read timeouts
  are genuinely exercised.

## Errors, which are the contract's sharp edge

Real vLLM installs exception handlers that convert failures into **400** with
`{"error": {message, type, param, code}}`. FastAPI's default is **422** with
`{"detail": [...]}`. If this port left the default, "a client that sends a bad request is the one
case where an unmodified product could tell pvllm from the real thing, on *every* endpoint at
once."

So a malformed body:

```json
{"error": {"message": "1 validation error:\n  {'type': 'missing', 'loc': ('body', 'prompt'), 'msg': 'Field required', 'input': {'model': 'dense-0.6b'}}",
           "type": "Bad Request", "param": "body.prompt", "code": 400}}
```

A rejected parameter:

```json
{"error": {"message": "best_of (3) must equal n (1); vLLM removed support for best_of > n",
           "type": "BadRequestError", "param": null, "code": 400}}
```

An unmodelled feature — **501, naming it**:

```json
{"error": {"message": "Batched prompts are not supported; send one prompt per request.",
           "type": "NotImplementedError", "param": "prompt", "code": 501}}
```

Note the two `type` strings, which are easy to conflate and both correct:

- a pydantic **schema** failure becomes `"Bad Request"` (the `HTTPStatus` phrase);
- an error raised inside handler code becomes `"BadRequestError"` (`create_error_response`'s
  default).

Both appear on the wire, on different paths, because both do upstream.

Handlers are registered for `HTTPException`, `RequestValidationError`, `ValueError`, `TypeError`,
`OverflowError`, `NotImplementedError`, and — the one that actually changes outcomes — bare
`Exception`. The first few are near-inert because every route already wraps its body; the catch-all
is what stops a `KeyError` out of `/tokenize` leaving as Starlette's bare `text/plain` "Internal
Server Error", which is the one shape the module exists to abolish.

There is also a message sanitiser: pydantic-core's internal `loc` vocabulary
(`function-wrap`, `tagged-union`, `lax-or-strict`, …) is stripped so `param` names a field a
client can act on rather than a pydantic construct.

## `/v1/responses` — the stateful surface

Different enough from chat completions to be worth its own paragraph, because a client written
against one will not work against the other:

| | chat completions | responses |
|---|---|---|
| stream frames | `data: {...}` deltas | **named events**: `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, `response.completed` |
| terminator | `data: [DONE]` | **no `[DONE]`** — a client that waits for one hangs |
| usage fields | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` |
| retrieval | none | `GET /v1/responses/{id}`, `previous_response_id` |

And the behaviour that surprises everybody: **the response store is off by default**, exactly as
upstream. Without `VLLM_ENABLE_RESPONSES_API_STORE=1`, `GET /v1/responses/{id}` and
`previous_response_id` 404 — precisely as they do against stock vLLM.

> Note the `VLLM_` prefix: this is the one deliberate exception to the `PVLLM_*` env convention
> (chapter [06](06-configuration.md)), so "one runbook flag flips both" engines. Its reasoning is worth
> quoting, because it is the project's philosophy applied to a default: "a pvllm that stored by default
> would *succeed* where the real thing fails, and the divergence would only surface when the user
> swapped the real engine back in."

Two refinements that mirror upstream's own choices:

- `store=True` is **accepted and quietly dropped** when the store is off, rather than erroring,
  because the OpenAI SDK sends `store=True` by default and erroring would reject every SDK
  request;
- `background=True` **is** an error when the store is off, with upstream's own message.

Not implemented, and refusing by name: tool calling, harmony/gpt-oss reasoning, background mode.

## The chat template and multimodal content

`/v1/chat/completions` accepts OpenAI content parts, including `image_url`. An image becomes 256
placeholder tokens plus an encoder pass — chapter [23](23-multimodal.md). Under the default mock
tokenizer, the chat template is pvllm's own; with `--tokenizer-mode slow` it is **the model's own
template**, rendered from the tokenizer's config, so prompt token counts match real vLLM exactly.

## Readiness

```
@app.get("/health")
```

Reports ready only once weight load and profiling are complete. That is true by construction —
`EngineCore` runs both in its constructor — but the endpoint exists because a product that polls
readiness needs something to poll. See chapter [14](14-memory-model.md) for why a meaningful
readiness test needs `--cost-model-profile roofline`: under the default `constant` model, startup
is essentially instant and no timeout is exercised.

## Trying it with a real client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

print(client.models.list())

for chunk in client.chat.completions.create(
    model="dense-0.6b",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The OpenAI SDK works unmodified. That is the promise: **compatibility is at the HTTP layer**, not
at the Python import layer.

## Known gaps worth naming

Because this chapter is about a contract, it should be explicit about what is not honoured today:

- **Batched prompts** — refused with 501 (above).
- **`best_of > n`** — refused with 400, matching upstream's removal of the feature.
- **`echo` and `suffix`** are declared in the completion schema but **not implemented**: they are
  accepted and ignored rather than refused. Real vLLM honours `echo` by prepending the prompt to
  the completion text, so a client relying on it gets different text here. If you depend on either,
  treat this as an open gap rather than a modelled behaviour.
- **`logit_bias`** is accepted and reaches `SamplingParams`, but there is no distribution to bias —
  nothing observable changes.
- **Responses API tool calling, harmony reasoning, background mode** — refused by name.

## Check yourself

- What status code does a malformed body get, and what would FastAPI have returned?
- What status code does an unmodelled feature get, and what must the message contain?
- Name three differences between the chat-completions stream and the responses stream.
- Why does `GET /v1/responses/{id}` 404 by default, and is that a bug?
- What ends a stream in `/v1/completions`, and what ends one in `/v1/responses`?
- Which field tells a client how much the prefix cache saved?

## Next

[20. Observability](20-observability.md) — metrics, traces, the timeline, and `/debug/*`.
