# 26. KV disaggregation

> **Files:** [`pvllm/distributed/kv_transfer/base.py`](../../pvllm/distributed/kv_transfer/base.py), [`sim_connector.py`](../../pvllm/distributed/kv_transfer/sim_connector.py), [`pvllm/sim/kv_store.py`](../../pvllm/sim/kv_store.py), [`pvllm/config/kv_transfer.py`](../../pvllm/config/kv_transfer.py)
> **Upstream:** `vllm/distributed/kv_transfer/kv_connector/v1/base.py` and `simple_cpu_offload_connector.py` (Tier B); the store is Tier **D**
> **Prerequisites:** chapters [10](10-prefix-caching.md), [15](15-cost-model.md).

Prefill is compute-bound; decode is memory-bound. So run them on different machines: a prefill node
computes a prompt's KV and publishes it, a decode node pulls it instead of recomputing. That is
**disaggregated prefill**, and the question it exists to answer is arithmetic:

> **Is pulling the KV cheaper than recomputing it?**

Both sides of that comparison are here — the store's bandwidth and latency on one side, the cost
model's prefill time on the other — so a deployment can be told which way the answer goes for its
prompt lengths and its store, without a GPU or a network.

## The connector interface

A KV connector has **two halves that live in different places and never call each other directly**:

| Side | Methods | Job |
|---|---|---|
| scheduler | `get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`, `request_finished` | decide what to load, react to the blocks allocated for it, pack per-step instructions |
| worker | `start_load_kv`, `wait_for_save` | perform the transfer |

> The split is the whole design. The scheduler must decide what to load *before* the step runs,
> without blocking on a network; the worker must do the moving without knowing why.

Reproducing the split rather than collapsing it is what makes the scheduling behaviour real. A request
whose KV is still arriving sits in `WAITING_FOR_REMOTE_KVS` and is not admitted — and that is a state a
product's latency depends on.

## Where it hooks into the step

Reading the scheduler and the engine core together:

```python
# scheduler admission, AFTER the local prefix cache lookup
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
num_computed_tokens = num_new_local_computed_tokens

if self.connector is not None:
    num_external_tokens, _ = self.connector.get_num_new_matched_tokens(
        request, num_computed_tokens
    )
    num_computed_tokens += num_external_tokens
```

**After the local lookup**, "because pulling KV the engine already has in memory would be strictly
worse than using it." The local prefix cache always wins.

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens + num_external_tokens,
    new_computed_blocks=new_computed_blocks,
)
```

The externally-held tokens count as **computed** — the request will not run them — but blocks still have
to exist to *receive* them. Advancing `num_computed_tokens` without allocating for the pulled KV would
be "a write past the end of the block table", which the slot-mapping oracle catches (chapter
[13](13-worker-and-model-runner.md)).

Then the engine core, before the model runs:

```python
def _transfer_kv(self, scheduler_output) -> None:
    seconds = connector.start_load_kv(metadata)
    if seconds > 0.0:
        self.clock.advance(seconds)  # ← the pull costs real modeled time
```

and after it:

```python
saved = self.scheduler.connector.wait_for_save(scheduler_output.kv_connector_metadata)
if saved > 0.0:
    self.clock.advance(saved)  # ← the publish costs too
```

**Both sides pay.** The producer for its writes, the consumer for its reads — charged on the engine's
clock, inside the step that issued them, "so the transfer shows up next to the prefill it replaced
rather than being free."

That symmetry was a bug once, and the fix is instructive: the write charge used to be gated on the step
also having loads. "A pure prefill node never loads anything, and gating on the load metadata meant its
writes were modeled, accumulated, and then never spent. A disaggregated pair whose whole question is
'does publishing cost less than recomputing' answered it with the publish side free."

## The store

```python
@dataclass
class SimKVStore:
    bandwidth_bytes_per_second: float = 10e9
    latency_seconds: float = 0.001

    def transfer_seconds(self, num_bytes: int) -> float:
        return self.latency_seconds + num_bytes / self.bandwidth_bytes_per_second
```

It holds block **hashes**, never bytes, and answers two questions: is this prefix here, and how long
would moving it take. "That is the whole content of KV disaggregation from the engine's point of view."

Stores are shared through a module-level registry keyed by **name**, so two `pvllm` engines in one
process can hand KV to each other — keyed by name rather than passed by reference "because the two
engines are configured separately, which is the whole point of disaggregation."

**The numbers are yours to supply.** From the source: "an NVMe-backed LMCache and an S3 bucket differ by
four orders of magnitude, and which one you have decides whether disaggregation helps or hurts."

## Doing the arithmetic

The comparison, on one `dense-8b` request with a 2,048-token prompt:

```
recompute:   prefill 2048 tokens              ≈ 159 ms      (chapter 15's roofline)
pull:        2048 tokens × 128 KiB/token      = 256 MiB
             at 10 GB/s + 1 ms latency        ≈ 27 ms       → pull wins, 6x
             at 1 GB/s  + 1 ms latency        ≈ 269 ms      → recompute wins
             at 100 MB/s (object storage)     ≈ 2.7 s       → recompute wins hugely
```

The crossover is at roughly `kv_bytes / prefill_seconds` — about 1.6 GB/s for this model and prompt
length. Above it, disaggregation pays; below it, you are paying network to avoid arithmetic. That number
moves with the model (KV per token) and the prompt length (both sides scale, but latency is fixed), which
is exactly why you want to compute it rather than guess.

```bash
# a two-engine exercise: one publishes, one consumes
python -c "
from pvllm.sim.kv_store import get_store
s = get_store('shared', bandwidth_bytes_per_second=10e9, latency_seconds=0.001)
print('pull 256 MiB:', round(s.transfer_seconds(256 * 2**20) * 1000, 2), 'ms')
s2 = get_store('slow', bandwidth_bytes_per_second=1e8, latency_seconds=0.02)
print('same over object storage:', round(s2.transfer_seconds(256 * 2**20) * 1000, 2), 'ms')
"
```

## Configuration

```python
@dataclass
class KVTransferConfig:
    kv_connector: str | None = None  # "SimSharedStoreConnector"
    kv_role: str | None = None  # kv_producer | kv_consumer | kv_both
    kv_rank: int | None = None
    kv_parallel_size: int = 1
    kv_connector_extra_config: dict[str, Any]  # the store's bandwidth and latency
```

Real transports are **refused by name**:

```
NotImplementedError: KV connector 'LMCacheConnectorV1' is a real transport and is not available
here. pretending-vllm provides ['SimSharedStoreConnector'], which models an external store with a
configurable bandwidth and latency -- set those from a measurement of yours and the scheduling
around it is faithful.
```

Rather than substituting the simulated store, "because their bandwidth and failure modes are the whole
question a disaggregation experiment is asking."

`kv_role` decides whether an engine publishes, consumes, or both — and it is the flag whose effect the
arithmetic above is about.

## What is missing, and named rather than approximated

The source is unusually explicit here, and you should read this list before designing an experiment:

- **Loads are synchronous.** `get_num_new_matched_tokens` always reports `async=False`, so **no request
  ever sits in `WAITING_FOR_REMOTE_KVS`**. Upstream supports the async shape, where a request waits
  outside the running set while its KV arrives, and that changes admission behaviour — "so it is absent
  rather than half-built."
- **No deferred block release.** A connector asking to hold a request's blocks past completion (an async
  push that has not landed) raises `NotImplementedError` rather than having its answer discarded. The
  earlier version discarded it, "so the base class's contract could not be honoured by any connector that
  needed it, and a real async push could not be modeled at all."
- **No handshake, no failure mode, no partial transfer.** "A store that goes away mid-transfer, or
  returns corrupt KV, is a real failure this cannot produce."

## What it reports

```
vllm:external_prefix_cache_queries_total
vllm:external_prefix_cache_hits_total
```

Beside the local `vllm:prefix_cache_*` counters, so you can see the two tiers separately — which is the
whole point of having a second tier.

## Check yourself

- Why is the external store consulted *after* the local prefix cache rather than before?
- Why must blocks be allocated for tokens the request will never compute?
- Which side of a disaggregated pair pays, and in which step?
- For a `dense-8b` and a 2,048-token prompt, roughly what store bandwidth is break-even?
- Why is a real transport refused rather than mapped onto the simulated store?
- Name the async behaviour that is absent, and say what it would change.

## Next

[27. Pooling and embeddings](27-pooling-and-embeddings.md).
