# 17. Engine core and frontends

> **Files:** [`pvllm/v1/engine/core.py`](../../pvllm/v1/engine/core.py), [`core_client.py`](../../pvllm/v1/engine/core_client.py), [`llm_engine.py`](../../pvllm/v1/engine/llm_engine.py), [`async_llm.py`](../../pvllm/v1/engine/async_llm.py), [`output_processor.py`](../../pvllm/v1/engine/output_processor.py), [`input_processor.py`](../../pvllm/v1/engine/input_processor.py), [`pvllm/entrypoints/llm.py`](../../pvllm/entrypoints/llm.py)
> **Upstream:** same paths (Tier B)
> **Prerequisites:** chapters [12](12-scheduler.md), [13](13-worker-and-model-runner.md), [16](16-clock-and-determinism.md).

The scheduler decides, the worker executes. This chapter is the thing that calls both of them
in a loop, and the two ways a program drives it.

## `EngineCore` — the loop, the clock, the trace

```python
class EngineCore:
    def __init__(self, vllm_config, executor_class=None, log_stats=True, trace=None):
        self.clock = current_platform.build_clock(...)  # ← owns the clock
        self.trace = trace or self._open_trace()  # ← owns the trace
        self.executor = executor_class(vllm_config, self.clock)
        kv_cache_config = self._initialize_kv_caches()
        self.scheduler = Scheduler(vllm_config, kv_cache_config, trace=self.trace)
        self.structured_output_manager = StructuredOutputManager(vllm_config)
```

It owns exactly three things nobody else may own: **the clock**, **the trace**, and **the step
loop**. The clock comes from the platform rather than being constructed directly, so the core
never learns that its timebase is simulated — the same reason it does not import a worker class
(chapter [03](03-simulation-boundary.md)).

### Startup: `_initialize_kv_caches`

This is where the memory model meets the KV cache layout, and it runs in upstream's order:

```python
# a profiling step, then the memory model
available = self.executor.determine_available_memory()[0]
specs = self.executor.get_kv_cache_specs()[0]  # per-layer specs
groups = get_kv_cache_groups(specs)  # partition into groups
layers_per_group = max(len(g.layer_names) for g in groups)
page_size = groups[0].kv_cache_spec.page_size_bytes * layers_per_group
num_blocks = available // page_size

kv_cache_config = KVCacheConfig(num_blocks=int(num_blocks), kv_cache_groups=groups)
self.executor.initialize_from_config([kv_cache_config])
self.executor.compile_or_warm_up_model()
```

The divisor is **layers per group**, not the model's layer count and not the group count,
because the groups *share* the pool — which is what makes hybrid attention save anything at all
(chapter [11](11-hybrid-kv-groups.md)). For a dense model there is one group holding every layer
and the two forms coincide.

### `step()`

```python
def step(self):
    planned = self._plan_step()  # schedule + stamp SCHEDULED + grammar bitmask
    if planned is None:
        return {}, False
    self._transfer_kv(planned)  # pull external KV, charge the clock
    return self._finish_step(planned, self.executor.execute_model(planned))
```

`step` and `step_async` differ in exactly one line — the one that spends the step's duration —
and share `_plan_step` / `_finish_step`, "so a clock mode can never change what the engine
decided, only how long the process spent deciding it."

`_plan_step` does three things in order:

1. `scheduler.schedule()`;
2. read the clock **once** and stamp `SCHEDULED` on every request the scheduler admitted —
   before the model runs, "so the queue wait ends where the request's own work begins, not after
   the whole batch's forward pass";
3. compute the structured-output bitmask, which depends on how far each grammar has advanced and
   is therefore only computable once this step's batch is known.

`_finish_step` folds the model output back through the scheduler, charges any KV-connector
writes, stamps the outputs from the one clock, emits the step trace record, and merges in any
grammar-compilation failures.

That last one is worth a note because it is a pattern you will see repeatedly: a malformed schema
**fails its own request** with `FINISHED_ERROR` rather than raising. "Taking down the engine step
that noticed it would let one bad request deny service to every other." And the failure travels
the same path a completion does, because the frontend is still holding that request's queue — a
failure delivered nowhere is a request that hangs.

### `add_request` and `abort_requests`

```python
def add_request(self, request: EngineCoreRequest) -> None:
    arrival_time = self.clock.time()  # ← stamped here, R19.1
    req = Request.from_engine_core_request(request, arrival_time=arrival_time)
    if self.log_stats:
        req.record_event(EngineCoreEventType.QUEUED, arrival_time)
    if req.use_structured_output:
        self.structured_output_manager.grammar_init(req)  # compile *while* it queues
    self.scheduler.add_request(req)
    self.trace.emit(
        "request", t=arrival_time, request_id=req.request_id, event="arrived"
    )
```

`abort_requests` traces only ids the core actually holds: "a trace claiming an abort that did not
happen is worse than one that is silent, because it is read as evidence."

## `EngineCoreClient` — how the frontend reaches the core

```
class EngineCoreClient(ABC):
    def add_request(self, request) -> None
    def abort_requests(self, request_ids) -> None
    def get_output(self) -> dict[int, EngineCoreOutputs]
    async def get_output_async(self) -> dict[int, EngineCoreOutputs]
    @property
    def clock_time(self) -> float          # ← on the interface since the first commit
    def make_stats(self) -> dict[str, Any]
    def reset_prefix_cache(self) -> bool
    def has_requests(self) -> bool
```

Three implementations: `InprocClient` (default, a direct call), `SyncMPClient` and
`AsyncMPClient` (chapter [18](18-multiprocess-engine.md)).

`clock_time` being on this interface from the beginning is the design decision that made
multiprocessing "a transport change rather than a redesign" (chapter
[16](16-clock-and-determinism.md)).

Note that `make_client` refuses a custom `executor_class` in multiprocess mode: the child
receives its configuration by pickle, and a class defined in a test module does not survive the
spawn.

## `InputProcessor` — the way in

```
def process_inputs(self, request_id, prompt, sampling_params=None, *, client_index=0, ...)
    -> EngineCoreRequest
```

Validate the sampling surface, tokenize, check the prompt against `max_model_len`, resolve
`max_tokens`, build the wire type. **It does not stamp `arrival_time`** — upstream reads
`time.time()` at exactly this point, and that is the one place clock ownership forces a
divergence.

`Request` refuses to be built with an unresolved `max_tokens`:

```
ValueError: sampling_params.max_tokens must be resolved before a Request is built;
the processor resolves it against max_model_len
```

## `OutputProcessor` — the way out

Holds one `RequestState` per in-flight request: the detokenizer, the arrival time, the token
counts, the parent request for `n > 1`. Per step it:

1. skips outputs for requests aborted between scheduling and delivery;
2. handles a pooling output as a single terminal result (chapter
   [27](27-pooling-and-embeddings.md));
3. advances the detokenizer and checks **stop strings** — the frontend's own stop condition;
4. absorbs `QUEUED` / `SCHEDULED` events into the timing;
5. builds `RequestOutput` with the right `output_kind` semantics;
6. retires finished requests, recording what they contributed to the metrics.

Three details with real consequences:

**Stop strings must be aborted upward.**

```python
if stop_string is not None and finish_reason is None:
    finish_reason = FinishReason.STOP
    self.stopped_by_string.append(engine_output.request_id)
```

The engine core still believes those requests are running, so it must be told to abort them —
otherwise they keep generating and holding blocks. `take_stopped_by_string()` is *taken* rather
than read, so a request is aborted once.

**The core's `QUEUED` stamp supersedes the frontend's.** In process the two are equal; over a
process boundary the frontend only knows the *last step's* time when it submits, so its estimate
is stale by up to a step and every queue time computed from it would be inflated. Taking the
core's stamp makes the timing identical in both transports.

**Only the *first* `SCHEDULED` event ends the queue wait.** A preempted request is admitted
again later, and taking the newer stamp would report a queue time longer than the request's whole
life.

And one that is pure C6 conformance: for an `n > 1` request, a *record* is emitted **per child**
(upstream observes `n` completions, `n` latency samples), but the **value of `n`** is observed
once per client request. Collapsing those two cardinalities into one gate made a dashboard built
against real vLLM read a third of the true completion rate under `n=3` traffic.

## `LLMEngine` — the synchronous frontend

```python
llm_engine.add_request(request_id, prompt, params)
while llm_engine.has_unfinished_requests():
    for output in llm_engine.step():
        ...
```

One `step()` per call, outputs returned as they finish. This is what the offline `LLM` class
drives, and it is why offline runs are byte-for-byte deterministic: `LLM.generate` submits every
prompt *before* the first step, so batch composition is fixed by the workload alone.

## `LLM` — the offline entrypoint

```python
from pvllm.entrypoints.llm import LLM

llm = LLM(model="dense-0.6b", max_model_len=2048)  # any EngineArgs field as a kwarg
llm.generate(prompts, sampling_params)  # -> list[RequestOutput]
llm.chat(messages, sampling_params)  # applies a chat template
llm.embed(prompts)  # -> list[PoolingRequestOutput]
llm.shutdown()  # or use it as a context manager
```

One of the two compatibility surfaces the project promises (the other is HTTP). Note the import
path: `pvllm.entrypoints.llm`, not `pvllm`.

## `AsyncLLM` — what the HTTP server sits on

```python
async for output in engine.generate(prompt, sampling_params, request_id):
    ...
```

Three mechanisms, and the third is the one that matters most.

**One output handler for the whole engine.**

```python
async def _run_output_handler(self):
    while True:
        if not self.engine_core.has_requests():
            await asyncio.sleep(0)
            if not self._queues:
                return
            continue
        now = self.engine_core.clock_time
        engine_core_outputs = await self.engine_core.get_output_async()  # ← awaited
        for client_outputs in engine_core_outputs.values():
            for output in self.output_processor.process_outputs(...):
                self._queues[output.request_id].put_nowait(output)
            stopped = self.output_processor.take_stopped_by_string()
            if stopped:
                self.engine_core.abort_requests(stopped)
        await asyncio.sleep(0)  # let the loop turn, so streaming actually streams
```

One loop, not one per request: a step produces output for the whole batch at once, and the
engine core must be stepped from exactly one place. `get_output_async` is **awaited** — under a
real clock that is where the step's modeled duration is spent, and holding the loop through it
would stop the server streaming for exactly as long as the step it is streaming through.

**Per-request queues.** `generate` registers an `asyncio.Queue` under the request id and awaits
it. For `n > 1`, the queue is registered under the *parent's* id and the children's aggregate is
routed there.

**Cancellation — the load-bearing behaviour.**

```
finally:
    # Runs on normal completion *and* on cancellation.
    self._queues.pop(request_id, None)
    live = self.output_processor.abort_requests(children or [request_id])
    if live:
        self.engine_core.abort_requests(live)
```

When a client disconnects, `asyncio` cancels the generator awaiting its queue, the `finally`
aborts the request in the engine core, and the scheduler frees its blocks **within one step**.

> Getting this wrong is invisible until capacity runs out under real traffic — the disconnected
> requests keep generating and holding blocks forever — which is exactly the failure a product
> wants to test for and cannot, against an engine that leaks them too.

Two guards worth knowing about, both learned the hard way:

- **Duplicate request ids are rejected before anything is mutated.** `/v1/responses` lets a
  client choose the id, so a collision is reachable from outside. Registering the queue first
  would rebind the in-flight request's queue to the new consumer, and the ensuing failure would
  run *this* call's `finally` against the *other* request — aborting it and leaving a stale
  scheduler entry that kills the next `schedule()` with a `KeyError`, taking the output handler
  and every later request on every endpoint with it.
- **A dead engine reaches every in-flight request.** The handler's `except` puts an
  `EngineDeadError` on every queue rather than leaving them hanging.

## Try it

The synchronous loop, one step at a time:

```python
from pvllm.v1.engine.llm_engine import LLMEngine
from pvllm.engine.arg_utils import EngineArgs
from pvllm.sampling_params import SamplingParams

engine = LLMEngine.from_engine_args(EngineArgs(model="dense-0.6b", max_model_len=512))
engine.add_request("a", "first prompt", SamplingParams(max_tokens=4))
engine.add_request("b", "second prompt", SamplingParams(max_tokens=6))

step = 0
while engine.has_unfinished_requests():
    step += 1
    for out in engine.step():
        tag = "finished" if out.finished else "partial "
        print(
            f"step {step}: {out.request_id} {tag} ({len(out.outputs[0].token_ids)} tokens)"
        )
print("total steps:", step)
engine.shutdown()
```

```
step 1: a partial  (1 tokens)
step 1: b partial  (1 tokens)
step 2: a partial  (2 tokens)
step 2: b partial  (2 tokens)
step 3: a partial  (3 tokens)
step 3: b partial  (3 tokens)
step 4: a finished (4 tokens)
step 4: b partial  (4 tokens)
step 5: b partial  (5 tokens)
step 6: b finished (6 tokens)
total steps: 6
```

Note `step()` returns an output for **every** request that produced a token, not only the finished
ones — the default `output_kind` is cumulative, so each one carries the full text so far. Six steps
for a four-token and a six-token request: request `a` leaves the batch at step 4 and `b` decodes
alone afterwards.

And the async one, with cancellation:

```python
import asyncio
from pvllm.engine.arg_utils import AsyncEngineArgs
from pvllm.v1.engine.async_llm import AsyncLLM
from pvllm.sampling_params import SamplingParams


async def main():
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(model="dense-0.6b", max_model_len=512)
    )
    gen = engine.generate("hello", SamplingParams(max_tokens=50), request_id="r1")
    async for out in gen:
        n = len(out.outputs[0].token_ids)
        if n >= 5:
            break  # walk away after five of fifty tokens
    await gen.aclose()  # what a disconnect does: runs the `finally`
    await asyncio.sleep(0)
    print("tokens received:", n)
    print("engine still tracking:", engine.output_processor.num_requests)


asyncio.run(main())
```

```
tokens received: 5
engine still tracking: 0
```

The request asked for 50 tokens and produced 5. `aclose()` is what a real disconnect does — it runs
the generator's `finally`, which aborts the request in the engine core and frees its blocks. Without
it, the abort waits for garbage collection, which is exactly the leak the `finally` exists to prevent.

## Check yourself

- What three things does `EngineCore` own exclusively?
- Why do `step` and `step_async` share `_plan_step` and `_finish_step`?
- Why is `SCHEDULED` stamped before the model runs rather than after?
- A stop string ends a request. Why must the frontend then call `abort_requests`?
- Why is there one output handler task rather than one per request?
- What frees a disconnected client's KV blocks, and how long does it take?

## Next

[18. The multiprocess engine](18-multiprocess-engine.md) — the same core, in its own process.
