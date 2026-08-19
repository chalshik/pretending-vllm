# 24. Parallelism

> **Files:** [`pvllm/config/parallel.py`](../../pvllm/config/parallel.py), [`pvllm/v1/engine/dp_client.py`](../../pvllm/v1/engine/dp_client.py), the parallelism terms in [`sim/cost_model.py`](../../pvllm/sim/cost_model.py) and [`sim/memory.py`](../../pvllm/sim/memory.py), the validation in [`v1/worker/sim_worker.py`](../../pvllm/v1/worker/sim_worker.py)
> **Upstream:** `vllm/config/parallel.py` (Tier C), `vllm/v1/engine/core_client.py`'s `DPLBAsyncMPClient` (Tier B)
> **Prerequisites:** chapters [14](14-memory-model.md), [15](15-cost-model.md).

Four axes, four different things bought. This is the chapter where a simulator earns its keep,
because getting these wrong on real hardware costs a reservation and a day.

| Axis | Splits | Buys | Costs |
|---|---|---|---|
| **TP** tensor parallel | every layer, across devices | latency **and** memory | an all-reduce per layer |
| **PP** pipeline parallel | layers into stages | memory only (here) | one hand-off per boundary |
| **DP** data parallel | nothing — whole replicas | throughput | a partitioned prefix cache |
| **EP** expert parallel | an MoE's experts | memory, decisively | a wider collective, and lockstep |

## Tensor parallelism

Every layer is sharded across `tp_size` devices. Heads divide, so KV divides too, and the
activations have to be all-reduced twice per layer.

```bash
python -c "
from pvllm.sim.cost_model import build_cost_model, StepProfile
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.model_db import load_model_card
m, d = load_model_card('dense-70b'), load_device_card('datacenter-80gb')

def cost(tp=1, pp=1, tokens=512, reqs=1):
    cm = build_cost_model('roofline', m, d, dtype='bfloat16', tp_size=tp, pp_size=pp)
    return cm.step_cost(StepProfile(num_tokens=tokens, num_reqs=reqs,
                                    query_lens=[tokens//reqs]*reqs, seq_lens=[1024]*reqs))

for tp in (1, 2, 4, 8):
    c = cost(tp=tp)
    print(f'prefill 512  tp={tp}: {c.duration*1000:7.2f} ms  comm={c.comm_seconds*1000:5.2f}')
for tp in (1, 2, 4, 8):
    c = cost(tp=tp, tokens=32, reqs=32)
    print(f'decode b=32  tp={tp}: {c.duration*1000:7.3f} ms  comm={c.comm_seconds*1000:5.3f}')
"
```

```
prefill 512  tp=1:  335.03 ms  comm= 0.00
prefill 512  tp=2:  173.65 ms  comm= 3.98
prefill 512  tp=4:   90.97 ms  comm= 3.98
prefill 512  tp=8:   49.64 ms  comm= 3.98
decode b=32  tp=1:   60.979 ms  comm=0.000
decode b=32  tp=2:   33.683 ms  comm=0.249
decode b=32  tp=4:   19.910 ms  comm=0.249
decode b=32  tp=8:   13.024 ms  comm=0.249
```

TP=8 gives **6.7×** on prefill and **4.7×** on decode — near-linear then sublinear, which is the
shape real hardware shows. The reason is visible in the breakdown: prefill divides a large compute
term against a fixed communication cost, while decode divides a memory term that has a floor.

And the memory side, from chapter [14](14-memory-model.md): weights shard (minus the embeddings) and
KV shards with the KV heads. That is what makes a 70B model fit at all.

The refusal that matters:

```
ValueError: tensor_parallel_size=3 does not divide the model's 8 KV heads. vLLM refuses this
configuration at startup, so reporting a capacity number for it would describe an engine that
cannot run.
```

**And the exception:** MLA models. Chapter [11](11-hybrid-kv-groups.md) — the latent is replicated on
every rank, so scaling TP on a DeepSeek-class model buys weights and compute and buys **nothing at
all** on the KV cache.

## Pipeline parallelism

Layers are split into stages; a batch traverses every stage before a token comes out.

```
pp=1:  335.03 ms
pp=2:  335.06 ms
pp=4:  335.11 ms
pp=8:  335.21 ms
```

**Same step latency, less memory per device.** That is not a bug in the model, it is a stated
limitation — and the source is emphatic about why the obvious alternative is worse:

> Dividing the compute and memory terms by `pp_size` (as an earlier version did) would report a step
> `pp_size` times faster than it is.

What is *not* modeled is the throughput gain: real PP overlaps microbatches so the steady state
approaches one stage's time. There are no virtual engines here, so PP shows up as "same latency, less
memory per device" — **correct for a single request and pessimistic for a saturated one**. Read PP
results as a memory-fit answer, not a throughput answer.

Two accounting details that were bugs first:

- **Layers per stage is a ceiling, not a floor.** With 28 layers over 8 stages the busiest stage
  holds 4, not 3, and the stage that has to fit the model is the busiest one.
- **Stages are not interchangeable.** Upstream's `get_pp_indices` puts the remainder on the
  partitions *before* the last, because the last carries the output embedding. `pipeline_stage_ranges`
  ports that exactly — 32 layers over 7 stages is `4,4,5,5,5,5,4`.
- **Graph capture is charged per stage, not per model.** Using the whole-model layer count charged an
  8-stage deployment eight times its real capture time, "and startup time is what an autoscaler's
  cold-start budget is set from."

## Data parallelism

Not sharding. Each replica is a **whole engine** — its own weights, its own device, its own KV pool,
its own scheduler — and a router picks one per request. A request lives entirely inside one replica for
its whole life.

Three consequences a capacity plan turns on, all reproduced rather than assumed:

**Capacity multiplies; a single request does not get faster.** Four replicas serve roughly four times
the throughput at the same per-request latency. A request too large for one replica's KV pool is too
large for the deployment.

**The prefix cache is partitioned.** Two requests sharing a long system prompt hit the cache only if
the router sent them to the same replica. "A workload whose hit rate looks excellent on one engine can
lose most of it at `--data-parallel-size 8`, and that is the single most surprising thing about turning
DP on."

**The router's policy is load, not round-robin.** Upstream's score, ported verbatim:

```python
score = max(self.engine_inflight[index], waiting + running)
if usage > 0.5:
    score += waiting * 6.0 * max(0.0, usage - 0.5)     # penalise queueing on a pressured pool
```

The greater of this client's exact in-flight count and the replica's own `waiting + running`, plus a
penalty that ramps in once KV usage passes half. Because "which replica gets this request" is exactly
what a DP experiment asks.

One refusal, and it is the honest kind:

```
NotImplementedError: --data-parallel-size 4 with --clock-mode real is not supported. The replicas
run concurrently on separate devices, and stepping them from one process spends their durations in
sequence -- a real-clock run would take 4x as long as the deployment it is modeling and exercise a
client's timeouts against a number that is not the answer. Use the virtual clock, which models the
concurrency correctly.
```

Also not implemented: DP over the multiprocess engine core. Refused by name.

## Expert parallelism

For a mixture-of-experts model, expert parallelism changes what the DP replicas *are*: instead of
independent copies they become shards of one expert set.

> **Note:** `enable_expert_parallel` is an `EngineArgs` field but is **not exposed as a CLI flag**
> today — `pvllm serve --enable-expert-parallel` is rejected as an unrecognized argument, even though
> the README advertises it. Reach it through the Python API:
> `LLM(model="moe-8x7b", data_parallel_size=8, enable_expert_parallel=True)`.

The memory effect is the largest single number in this chapter:

```bash
python -c "
from pvllm.sim.model_db import load_model_card
from pvllm.sim.memory import compute_weight_bytes
c, g = load_model_card('moe-8x7b'), 2**30
print('experts are', round(c.num_hidden_layers*c.expert_parameters_per_layer/c.num_parameters*100,1), '% of parameters')
print('tp=1, no EP :', round(compute_weight_bytes(c,'bfloat16',1)/g, 2), 'GiB')
print('tp=8, no EP :', round(compute_weight_bytes(c,'bfloat16',8)/g, 2), 'GiB')
print('tp=1, ep=8  :', round(compute_weight_bytes(c,'bfloat16',1,ep_size=8)/g, 2), 'GiB')
"
```

```
experts are 96.6 % of parameters
tp=1, no EP : 86.99 GiB
tp=8, no EP : 11.3 GiB
tp=1, ep=8  : 13.49 GiB
```

87 GiB does not fit an 80 GiB card. 13.5 GiB does. **That is the difference between a deployment and
no deployment**, and it comes from each device owning whole experts (`E / ep`) rather than a slice of
every one.

### EP does not reduce per-device FLOPs

This is the arithmetic the source writes out because dividing by `ep_size` is the obvious wrong move:

> Under TP each rank holds every expert sliced to `I/tp` and runs `tokens × top_k` pairs through the
> slice: work = total/tp. Under EP each rank holds `E/ep` whole experts and runs the
> `tokens_total × top_k / ep` pairs that route to them at full width: work = total/ep — but
> `tokens_total` is the union across the `dp` replicas, so with `ep = dp × tp` the two land on the same
> number. **EP moves *where* the weights live, not how much arithmetic each device does.**

The communication term follows the same care: EP replaces the MoE all-reduce with a dispatch/combine
pair, which is the same byte volume by another name — "EP does not make the MoE layer's communication
bigger, it makes it *wider*." What changes is the token *set*: the collective spans the DP replicas, so
it carries `dp_size` times the tokens. At `dp_size == 1` there is no all-to-all at all (upstream's
`use_all2all_kernels` requires `dp > 1`), so a TP-only EP run reports the same duration as the TP run.

### Lockstep, and the dummy steps

The MoE collective is taken across every EP rank, so **a replica with no work of its own cannot skip a
step**. It runs a dummy single-token forward pass to keep the collective whole:

```python
def execute_dummy_batch(self) -> float:
    """Step the device without scheduling anything.
    The scheduler is not consulted and no request state moves."""
```

The cost is reported, per replica, "because the arithmetic is invisible otherwise: one request on a
four-replica EP deployment costs three dummy steps per real one." And `lockstep_rounds` is reported
beside `num_dummy_steps` because the ratio *is* the imbalance:

> D dummy steps against R rounds says D/R replica-steps went to nothing on an average round.

This is device time that produced nothing, and nothing upstream counts it — "a DP+EP deployment can sit
at full device utilisation with near-zero goodput and no metric says so."

The refusal, for the same reason as `--data-parallel-size` on a real clock: EP needs a MoE model.

```
ValueError: enable_expert_parallel needs a mixture-of-experts model, and 'dense-8b' declares
num_experts=0. There is nothing to spread across devices; use --tensor-parallel-size to shard a
dense model.
```

## Choosing between them

The decision procedure the numbers above support:

1. **Does the model fit on one device?** If not, TP first — it buys memory *and* latency.
2. **Still does not fit at your maximum TP?** Add PP (memory), or EP if it is an MoE (much bigger
   memory win).
3. **Fits, but not enough throughput?** DP — but check what happens to your prefix cache hit rate
   first.
4. **MoE?** Compare `tensor_parallel_size=8` against `data_parallel_size=8,
   enable_expert_parallel=True` and look at both the memory *and* the dummy-step count.

## Try it

`tensor-parallel-size` is **not** one of `bench sweep`'s sweepable axes (the sweepable set is
`block-size`, `device-card`, `enable-chunked-prefill`, `enable-prefix-caching`,
`gpu-memory-utilization`, `input-len`, `max-num-batched-tokens`, `max-num-seqs`, `num-prompts`,
`output-len`, `request-rate`), so sweep TP with a shell loop:

```bash
for tp in 1 2 4 8; do
  pvllm bench latency --model dense-70b --device-card datacenter-80gb \
    --tensor-parallel-size $tp --cost-model-profile roofline \
    --max-model-len 8192 --input-len 512 --output-len 16 --output-json tp-$tp.json
done

pvllm bench latency --model moe-8x7b --tensor-parallel-size 8 \
  --cost-model-profile roofline --input-len 512 --output-len 32
```

Expert parallelism has no CLI flag, so its side of the comparison goes through the Python API:

```python
from pvllm.entrypoints.llm import LLM
from pvllm.sampling_params import SamplingParams

llm = LLM(model="moe-8x7b", max_model_len=4096, data_parallel_size=8,
          enable_expert_parallel=True, cost_model_profile="roofline")
llm.generate(["a prompt"], SamplingParams(max_tokens=16))
print(llm.llm_engine.make_stats())      # includes the dummy-step counters
```

```
Memory profile: ... weights=13.49GiB ... num_gpu_blocks=29785, max_concurrency=116.35x

{'engine_step': 16, 'elapsed': 8.58, 'lockstep': True,
 'num_dummy_steps': 112, 'dummy_step_seconds': 0.61, 'lockstep_rounds': 16,
 'per_engine_dummy_steps': [0, 16, 16, 16, 16, 16, 16, 16]}
```

Read that last line: **one request, one replica doing the work, and seven replicas paying 16 dummy
steps each** — 112 replica-steps of device time that produced nothing, for 16 real ones. The weights
fit (13.49 GiB instead of 87), and this is what it costs at low load. Nothing upstream reports this
number.

`tests/sim/test_parallelism.py` and `tests/sim/test_expert_parallelism.py` assert the arithmetic in
this chapter, including the EP-does-not-reduce-FLOPs identity.

## Check yourself

- Why is TP's speedup near-linear on prefill and sublinear on decode?
- Why does PP show the same step latency here, and in which direction is that wrong?
- What does data parallelism do to your prefix cache hit rate?
- Under EP, why does per-device *compute* not fall even though per-device *memory* does?
- What is a dummy step, why is it necessary, and what does its count tell you?
- Which parallelism axis refuses to run under `--clock-mode real`, and why?

## Next

[25. Speculative decoding](25-speculative-decoding.md).
