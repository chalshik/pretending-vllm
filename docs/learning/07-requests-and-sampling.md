# 07. Requests and sampling

> **Files:** [`pvllm/v1/request.py`](../../pvllm/v1/request.py), [`pvllm/sampling_params.py`](../../pvllm/sampling_params.py), [`pvllm/outputs.py`](../../pvllm/outputs.py), [`pvllm/v1/engine/__init__.py`](../../pvllm/v1/engine/__init__.py), [`pvllm/v1/engine/input_processor.py`](../../pvllm/v1/engine/input_processor.py), [`pvllm/v1/core/sched/utils.py`](../../pvllm/v1/core/sched/utils.py), [`pvllm/v1/engine/parallel_sampling.py`](../../pvllm/v1/engine/parallel_sampling.py)
> **Upstream:** `vllm/v1/request.py` (Tier **A**), `vllm/sampling_params.py` (Tier C), `vllm/v1/core/sched/utils.py` (Tier A)
> **Prerequisites:** chapter [06](06-configuration.md).

`Request` is the object the scheduler reads and mutates on every step. Its shape is part
of the fidelity contract, and two details in it are load-bearing in a way that is easy to
miss.

## The journey of one request

```
HTTP body / LLM.generate()
  └─► protocol model            validate the OpenAI schema        (entrypoints)
      └─► SamplingParams        validate the sampling surface
          └─► InputProcessor    tokenize, check lengths, resolve max_tokens
              └─► EngineCoreRequest   the msgspec wire type
                  └─► Request         engine-core state, arrival time stamped here
```

Each hop drops something. The protocol model knows about `"model"` and `"stream"`; the
engine core has never heard of either. `Request` knows nothing about HTTP.

## `SamplingParams`: what a client asks for

Tier C — the full surface is accepted and validated exactly as upstream validates it,
because a client sending an out-of-range `top_p` must get the same error here as from real
vLLM.

```python
n: int = 1                      # → fanned out in the frontend, see below
temperature: float = 1.0
top_p: float = 1.0
top_k: int = 0
min_p: float = 0.0
seed: int | None = None         # honoured: same seed → same completion
stop: str | list[str] | None    # stop *strings* → output processor
stop_token_ids: list[int] | None
ignore_eos: bool = False
max_tokens: int | None = 16
min_tokens: int = 0
logprobs: int | None = None     # shape is real, values are synthetic
detokenize: bool = True
output_kind: RequestOutputKind  # CUMULATIVE | DELTA | FINAL_ONLY
structured_outputs: StructuredOutputsParams | None
```

**What these do here, precisely.** Values reach as far as changing the PRNG draw. A
parameter being *accepted* is a statement about the API surface, not a claim that it steers
text — there is no distribution to steer. So `temperature=0.1` and `temperature=1.9` are
both valid, both validated, and both produce pseudowords. What *is* real:

- `max_tokens`, `min_tokens`, `stop`, `stop_token_ids`, `ignore_eos` — these change *when a
  request finishes*, which changes scheduling, which changes everything.
- `seed` — same seed and parameters reproduce the same completion.
- `n` — real KV pressure, real queueing.
- `logprobs` — the schema and the shape are contractual: `len(logprob_token_ids) ==
  len(logprobs) == len(sampled_token_ranks)`, each inner list of length *k*. The values are
  not.

`RequestOutputKind` is worth knowing because streaming depends on it: `DELTA` returns only
what is new (what SSE needs), `CUMULATIVE` returns everything so far, `FINAL_ONLY` returns
one result at the end.

## `Request`: engine-core state

```python
request_id: str
prompt_token_ids: list[int]
sampling_params: SamplingParams | None      # exactly one of these two
pooling_params: PoolingParams | None        # ← chapter 27
arrival_time: float                         # stamped by the engine core
status: RequestStatus
num_computed_tokens: int                    # ← the whole scheduling model
spec_token_ids: list[int]                   # ← chapter 25
block_hashes: list[BlockHash]               # ← chapter 10
num_cached_tokens: int
num_preemptions: int
is_prefill_chunk: bool
mm_features: list[...]                      # ← chapter 23
structured_output_request: ... | None       # ← chapter 21
events: list[EngineCoreEvent]
```

The counters that replace the prefill/decode distinction:

```python
@property
def num_tokens(self) -> int:          # prompt + everything generated
    return len(self._all_token_ids)

@property
def num_tokens_with_spec(self) -> int:  # ...plus drafts awaiting verification
    return len(self._all_token_ids) + len(self.spec_token_ids)
```

Every step, for every request, the scheduler computes
`num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens` and hands out
as much of that as the budget allows. A brand-new request with a 500-token prompt asks for
500; a decoding request asks for 1; a decoding request with three drafts in hand asks for
4. **One expression, three behaviours.**

### Load-bearing detail 1: the status enum's *order*

```python
class RequestStatus(enum.IntEnum):
    WAITING = auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = auto()
    WAITING_FOR_REMOTE_KVS = auto()
    WAITING_FOR_STREAMING_REQ = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    # Anything after PREEMPTED is considered finished.
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    FINISHED_ABORTED = auto()
    FINISHED_IGNORED = auto()
    FINISHED_ERROR = auto()
    FINISHED_REPETITION = auto()
```

```python
@staticmethod
def is_finished(status) -> bool:
    return status > RequestStatus.PREEMPTED
```

`is_finished` is a **comparison**, not a set membership test. Reorder the members and
finished-request detection breaks silently: no type error, no failing assertion, just
requests that never complete. `tests/v1/test_request.py` pins the ordering, and it is one
of the entries in the mutation catalogue (chapter [30](30-testing-and-tooling.md)).

Note also `FINISHED_IGNORED → FinishReason.LENGTH`: a prompt that exceeds the model's
length cap is *ignored*, and the OpenAI API reports that as `"length"`.

### Load-bearing detail 2: block hashing is injected

```python
def attach_block_hasher(self, block_hasher: Callable[[Request], list[BlockHash]]) -> None:
    self._block_hasher = block_hasher
    self.update_block_hashes()
```

The hasher arrives as a callable, attached by the scheduler when the request is admitted.
Two reasons, both real:

- **Ownership.** Hashing policy — the algorithm, the salt, the extra keys — belongs to the
  KV cache manager, which knows the block size. The frontend that built the request has no
  business knowing either.
- **No reference cycles.** It is stored unbound, so `Request → partial → Request` never
  forms a cycle. A server holding thousands of finished requests until the cycle collector
  runs is a real leak, and upstream avoids it the same way.

### And one deliberate divergence

`arrival_time` is a **required** constructor argument here, where upstream defaults it to
`time.time()`. The engine core owns the clock and stamps it (chapters
[03](03-simulation-boundary.md), [16](16-clock-and-determinism.md)).

## The wire types

`EngineCoreRequest`, `EngineCoreOutput`, `EngineCoreOutputs` are **msgspec Structs**
([`v1/engine/__init__.py`](../../pvllm/v1/engine/__init__.py)), so the multiprocess client
pays real serialization cost rather than a modeled one. The same types are used in-process,
so both paths exercise the same shapes.

```python
class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    request_id: str
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams | None
    arrival_time: float | None = None      # None from the frontend is normal
    client_index: int = 0
    lora_request: Any = None
    cache_salt: str | None = None
    priority: int = 0
    mm_features: list[MultiModalFeatureSpec] = []
    pooling_params: PoolingParams | None = None
```

Two subtleties that cost real debugging time upstream and here:

- The imports of `MultiModalFeatureSpec`, `PoolingParams`, and `SamplingParams` are at
  **runtime**, not under `TYPE_CHECKING`. msgspec resolves a Struct's annotations when it
  builds a decoder, so a type that exists only for the type checker makes
  `Decoder(EngineCoreRequest)` raise `NameError` — which the in-process client never
  triggers, because it never decodes anything.
- `FinishReason` is an `IntEnum` for compact serialization, with `__str__` mapping to
  `"stop" | "length" | "abort" | "error" | "repetition"`.

`EngineCoreEvent` carries `QUEUED`, `SCHEDULED`, `PREEMPTED` with timestamps **from the
engine core's clock**. Every queue-wait and prefill-time metric is computed from the
interval between two of these — which is why they cannot come from a frontend clock.

## Stop conditions, and why the order is the specification

[`v1/core/sched/utils.py`](../../pvllm/v1/core/sched/utils.py), Tier A, 30 lines, called
after every appended token:

```python
def check_stop(request, max_model_len) -> bool:
    # First, and returning early: below min_tokens, nothing stops the request.
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]

    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id      # the API reports which token
        return True

    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    return False
```

**`min_tokens` is checked first and returns early**, so a request below its minimum cannot
stop for *any* reason, including EOS. Move that check below the EOS test and `min_tokens`
silently stops working exactly when the model emits EOS early — which is the only time it
matters.

Two things deliberately absent:

- **`ignore_eos`** is not tested here. The input processor leaves `eos_token_id` unset when
  it is requested, so the policy is decided once rather than at every check. Upstream does
  the same.
- **Stop *strings*** are not here at all. They need text, and text needs incremental
  detokenization — so they live in the output processor. Chapter [08](08-tokenizers.md).

There is a matching rule in the scheduler: tokens are appended **one at a time** with a
stop check after each, and the batch is trimmed the moment one hits. A request that stops
on its second of three sampled tokens must not emit the third — which matters constantly
under speculative decoding, where multi-token batches are the norm.

## `n > 1` is a frontend concern

[`v1/engine/parallel_sampling.py`](../../pvllm/v1/engine/parallel_sampling.py). The engine
core has no notion of `n`. One request is one sequence, and asking for four completions of
one prompt is **four requests that happen to share a prompt**.

This is not a shortcut; it is the behaviour worth reproducing:

- the four children queue independently and are preempted independently;
- they share the prompt's KV through the *ordinary* prefix cache, not a special case;
- a client sending `n=4` sees four times the decode pressure and one response.

`ParentRequest` owns the child ids and the aggregation, and guarantees the response arrives
as **one** `RequestOutput` carrying `n` `CompletionOutput`s in index order, whatever order
the children finished in. A seeded parent offsets its children's seeds so they stay
distinct — otherwise `n=4` with a seed would return the same completion four times.

## What comes back

[`pvllm/outputs.py`](../../pvllm/outputs.py):

```python
@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None = None
    logprobs: list[dict[int, Any]] | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None

@dataclass
class RequestOutput:
    request_id: str
    prompt: str | None
    prompt_token_ids: list[int] | None
    outputs: list[CompletionOutput]        # length n
    finished: bool
    num_cached_tokens: int = 0             # ← how much the prefix cache saved
    kv_transfer_params: dict | None = None
```

`num_cached_tokens` is the field to watch when you are tuning anything about prompts: it
is how many of this request's prompt tokens came from the prefix cache instead of being
computed.

## Try it

Stop conditions and `n > 1`, both observable:

```python
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams

llm = LLM(model="dense-0.6b", max_model_len=1024)

# n > 1: one response, four completions, shared prompt
out = llm.generate(["the shared prompt"], SamplingParams(n=4, max_tokens=6))[0]
print(len(out.outputs), [c.text for c in out.outputs])

# a seed makes it reproducible
a = llm.generate(["seeded"], SamplingParams(max_tokens=6, seed=42))[0].outputs[0].text
b = llm.generate(["seeded"], SamplingParams(max_tokens=6, seed=42))[0].outputs[0].text
print(a == b, repr(a))

# min_tokens beats an early stop
out = llm.generate(["hi"], SamplingParams(max_tokens=20, min_tokens=10))[0]
print(len(out.outputs[0].token_ids), out.outputs[0].finish_reason)
```

## Check yourself

- Why is `RequestStatus` an `IntEnum` whose member order cannot be changed?
- A request has `num_computed_tokens=500`, `num_tokens=520`, and two draft tokens. How
  many tokens does the scheduler try to give it?
- Why is `min_tokens` checked before EOS rather than after?
- Where are stop *strings* handled, and why not next to `check_stop`?
- A client sends `n=4`. How many `Request` objects exist in the engine core, and how do
  they avoid prefilling the prompt four times?

## Next

[08. Tokenizers and detokenization](08-tokenizers.md) — the two token counts a product
reads, and the streaming rule nobody expects.
