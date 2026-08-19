# 21. Structured output

> **Files:** [`pvllm/v1/structured_output/`](../../pvllm/v1/structured_output), [`pvllm/sim/structured_output.py`](../../pvllm/sim/structured_output.py), [`pvllm/sim/grammar.py`](../../pvllm/sim/grammar.py), [`pvllm/entrypoints/openai/structured_outputs.py`](../../pvllm/entrypoints/openai/structured_outputs.py)
> **Upstream:** `vllm/v1/structured_output/*` (Tier B); the grammar generator is Tier **D**
> **Prerequisites:** chapters [07](07-requests-and-sampling.md), [12](12-scheduler.md).

This is the chapter where the project departs from upstream's *mechanism* in order to keep its
*contract* — and the reasoning is worth following, because the shortcut is tempting in both
directions.

## What upstream does

Upstream constrains structured output by **masking the sampler**. At every step the compiled
grammar produces a bitmask over the vocabulary marking which tokens are legal, and the model picks
among them. Backends: `xgrammar`, `guidance`, `outlines`, `lm-format-enforcer`.

## Why this port does not

Reproducing the mask here would be easy and **useless**:

> The model is a token generator with no language model behind it and the tokenizer is a mock whose
> vocabulary is synthetic pseudowords, so a JSON grammar over that vocabulary would admit sequences
> that are "legal" in the emulated bitmask and detokenize to nothing resembling JSON. A product that
> called `json.loads()` on the result would fail — against the one feature it uses structured
> output *for*.

So the constraint is satisfied **at the level a consumer can observe**:
[`pvllm/sim/grammar.py`](../../pvllm/sim/grammar.py) generates a string that really does conform,
and `SimModel` emits that string's tokens.

```bash
python -c "
import json
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams, StructuredOutputsParams

llm = LLM(model='dense-0.6b', max_model_len=1024)
schema = {'type': 'object',
          'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
          'required': ['name', 'age']}

out = llm.generate(['give me a person'], SamplingParams(
    max_tokens=64, structured_outputs=StructuredOutputsParams(json=json.dumps(schema))))[0]
print('json  :', out.outputs[0].text, '->', json.loads(out.outputs[0].text))

print('choice:', repr(llm.generate(['pick'], SamplingParams(
    max_tokens=32, structured_outputs=StructuredOutputsParams(choice=['yes','no','maybe'])))[0].outputs[0].text))

print('regex :', repr(llm.generate(['number'], SamplingParams(
    max_tokens=32, structured_outputs=StructuredOutputsParams(regex=r'[0-9]{3}-[0-9]{4}')))[0].outputs[0].text))
"
```

```
json  : {"name": "delta", "age": 96} -> {'name': 'delta', 'age': 96}
choice: 'yes'
regex : '526-5465'
```

`json.loads()` works. That is the observable behaviour a product depends on, and it is the one this
engine keeps.

The generator is **self-checking where it can be**: a generated JSON value is validated against the
schema it came from, and a generated regex match is checked with `re.fullmatch`, before either is
returned. A generator that drifts from its own specification fails loudly.

## What *is* real: the scheduler-side interaction

This is the half the feature actually stresses in a deployment, and it is fully ported.

### Compilation is asynchronous

```python
def grammar_init(self, request: Request) -> None:
    if self.backend is None:
        self.backend = self._build_backend(request)
    request.structured_output_request.grammar = self.executor.submit(
        self._create_grammar, request)
```

A `ThreadPoolExecutor` sized at **half the CPUs**, as upstream: grammar compilation is CPU-bound,
and the default pool size (`CPUs × 5`) would oversubscribe a machine that is also running the
engine.

Compilation is started in `EngineCore.add_request` — *before* the request joins the queue — so it
overlaps the wait rather than beginning when the scheduler first looks at the request.

### A request whose grammar is not ready is not admissible

```python
class RequestStatus(enum.IntEnum):
    WAITING = auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = auto()     # ← set at construction
    ...
```

Set in `Request.__init__`, not by the scheduler, because "a request briefly `WAITING` before anyone
looked would be admissible in that window."

The first sampled token has to be constrained, and until the grammar compiles there is nothing to
constrain it with.

### And it is set aside, not left at the head of the queue

```python
blocked = request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
if blocked and not self._promote_grammar_request(request):
    self.waiting.pop_request()
    self.skipped_waiting.add_request(request)
    continue
```

One slow schema must not block every request behind it — which is the whole reason compilation is
asynchronous in the first place. Set-aside requests are returned to the **head** of the waiting
queue before the step ends ("they arrived before everything now queued behind them, and sending them
to the back would let a steady stream of unconstrained requests starve every constrained one
indefinitely").

**So a product that submits a hundred requests against a slow-to-compile schema sees the same
admission behaviour it would see against real vLLM.** That is what this feature is for here.

### A compile failure fails one request

```python
if isinstance(grammar, Exception):
    self.grammar_compile_error_reqs.add(request.request_id)
    return False
```

It does **not** raise: "a malformed schema is a client error belonging to one request, and taking
down the engine step that noticed it would let one bad request deny service to every other." The
engine core finishes those requests with `FINISHED_ERROR` and — importantly — delivers the failure
along the same path a completion takes, because the frontend is still holding their queues.

### Non-final prefill chunks are excluded

```python
constrained = sorted(
    request_id for request_id in num_scheduled_tokens
    if self.requests[request_id].use_structured_output
    and not self.requests[request_id].is_prefill_chunk
)
```

A request on a non-final prefill chunk samples no token this step, so constraining it would consume
a grammar position for a token that never exists. Upstream excludes it for the same reason.

Note the rows are keyed by request id in **sorted** order rather than by batch position: the worker
reorders the batch for its own reasons (chapter [13](13-worker-and-model-runner.md)), so a row index
derived from the scheduler's ordering would address the wrong request's row about half the time.

## The absent bitmask, and why absence is the honest choice

```python
def grammar_bitmask(self, requests, structured_output_request_ids) -> np.ndarray | None:
    """Always `None` here, and deliberately."""
    return None
```

The method is kept because it *is* the shape of the interaction — a consumer reading
`SchedulerOutput` sees the same field, and a future backend with a real distribution behind it would
fill it here without anything above changing.

> Returning a mask nothing consumes would be worse than returning nothing — it would read as
> fidelity that is not there.

That sentence is the project's whole philosophy in one line, and this is the cleanest example of it.

## Backends: refused by name

```python
UPSTREAM_BACKENDS = ("xgrammar", "guidance", "outlines", "lm-format-enforcer")
```

Asking for one of these raises, naming what is missing:

```
NotImplementedError: structured output backend 'xgrammar' is a compiled grammar engine over a
real vocabulary and is not available here. pretending-vllm provides the 'sim' backend, which
generates output conforming to the constraint ... Pass --structured-outputs-backend auto or sim.
```

(That message names a flag the CLI does not currently expose either; the backend is selectable through
`StructuredOutputsConfig` and per-request `guided_decoding_backend`.)

Rather than silently substituting a different grammar engine — "two backends disagree about edge
cases in JSON Schema, and a product that pinned one did so for a reason."

## The API surface

Both OpenAI's field and vLLM's extensions are accepted, because products targeting vLLM use the
latter:

```python
response_format = {"type": "json_schema", "json_schema": {...}}   # OpenAI
guided_json / guided_regex / guided_choice / guided_grammar        # vLLM extensions
structural_tag, guided_whitespace_pattern, guided_decoding_backend
```

`StructuredOutputsParams.__post_init__` enforces upstream's **mutual exclusion**: exactly one
constraint, because they compile to different grammars and there is no defined meaning for two at
once.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dense-0.6b",
  "messages": [{"role": "user", "content": "a person"}],
  "max_tokens": 64,
  "response_format": {"type": "json_schema", "json_schema": {"schema":
    {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}
}'
```

## What this can and cannot tell you

**Can:** whether your product handles the admission delay while a schema compiles; whether it
handles a malformed schema failing one request; whether your parser handles the shape you asked for;
what the queueing looks like when many requests share one expensive grammar.

**Cannot:** whether *your* backend's grammar accepts your schema. Backend conformance is one of the
two numbers a simulator cannot derive (the other is speculative-decoding acceptance, chapter
[25](25-speculative-decoding.md)) — there is no real grammar engine here to disagree with. Test your
schemas against the backend you actually deploy.

Deterministic like everything else: a constrained request's plan is decided once and cached, so a
request that is preempted and recomputed produces the same string. Otherwise preemption would change
the answer, which a product would experience as a nondeterministic API.

## Check yourself

- Why would emulating a token bitmask here be useless?
- Where does grammar compilation start, and why there rather than at first schedule?
- Why is a grammar-blocked request moved to a separate queue instead of left at the head?
- What happens to the *other* requests when one schema fails to compile?
- Why does `grammar_bitmask` return `None` instead of a plausible mask?
- Which structured-output property must you test against your real backend?

## Next

[22. LoRA](22-lora.md) — adapters as a queueing constraint.
