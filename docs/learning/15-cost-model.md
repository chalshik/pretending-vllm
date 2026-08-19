# 15. The cost model

> **Files:** [`pvllm/sim/cost_model.py`](../../pvllm/sim/cost_model.py), [`pvllm/sim/device.py`](../../pvllm/sim/device.py)
> **Upstream:** none — Tier **D**
> **Prerequisites:** chapters [13](13-worker-and-model-runner.md), [14](14-memory-model.md).
> **Contract:** **modeled.** Every duration this produces carries a `modeled` label wherever it surfaces.

Read the warning first, because it is the whole point of the chapter.

> This is the one place the project can be wrong in a way that **misleads** you. Everything
> else is either exactly right or obviously fake, whereas a latency figure looks like a
> measurement whatever its provenance.

Generated text is obviously nonsense, so nobody quotes it. A p99 of 6.24 ms looks exactly
like a measurement. It is not one. It ships **uncalibrated**.

## Two profiles

| Profile | What it does | Use for |
|---|---|---|
| `constant` (default) | fixed per-step, per-token, per-request costs | tests, and anything asserting on step *counts* |
| `roofline` | a compute term, a memory term, their maximum, plus communication, encoder, and launch overhead | anything asking "which way does this knob move things" |

`constant` is the default deliberately: a test asserting that a workload drains in 96 steps
should not also depend on the roofline's coefficients. It is fast, exactly reproducible, and
deliberately unrealistic — under it, weight loading is free and every step of the same shape
costs the same.

## The roofline

```python
# --- compute -------------------------------------------------------------
flops  = 2.0 * active_params_local * tokens                    # multiply + add per parameter
flops += 4.0 * layers * heads_local * head_dim * sum(q_i * s_i)  # attention: quadratic in context
t_compute = flops / (device.mfu * device.peak_flops)

# --- memory --------------------------------------------------------------
kv_bytes         = sum(seq_lens) * kv_bytes_per_token
activation_bytes = tokens * hidden_size * dtype_bytes * 4
bytes_moved      = weight_bytes_local + kv_bytes + activation_bytes
t_memory = bytes_moved / (device.bw_eff * device.memory_bandwidth)

# --- communication, encoder, launch overhead -----------------------------
t_comm     = ...     # TP allreduces, PP handoffs, EP dispatch/combine — chapter 24
t_encoder  = 2.0 * ENCODER_PARAMS * num_encoder_embeds / (mfu * peak_flops)   # chapter 23
t_overhead = num_kernels * device.launch_overhead

duration = max(t_compute, t_memory) + t_comm + t_overhead + t_encoder
```

The `max(t_compute, t_memory)` is the roofline. And this is the important claim:

> **The regimes are not coded for; both fall out of the `max`.**

- Prefill is compute-bound and linear in tokens because `flops` scales with tokens while
  `bytes_moved` is dominated by the fixed weight read.
- Decode is memory-bound and nearly flat because one token per request reads the same weights
  — until KV traffic grows with context and starts to dominate.

There is no `if prefill:` anywhere in the file. That is the bar this model is held to, rather
than absolute accuracy.

## Watch all four regimes

```bash
python -c "
from pvllm.sim.cost_model import build_cost_model, StepProfile
from pvllm.sim.hardware_db import load_device_card
from pvllm.sim.model_db import load_model_card
cm = build_cost_model('roofline', load_model_card('dense-8b'),
                      load_device_card('datacenter-80gb'), dtype='bfloat16')

print('prefill: duration vs tokens')
for t in (128, 512, 2048, 8192):
    c = cm.step_cost(StepProfile(num_tokens=t, num_reqs=1, query_lens=[t], seq_lens=[t]))
    print(f'  {t:5d} tok: {c.duration*1000:8.2f} ms  {c.as_dict()[\"bound_by\"]}')

print('decode: duration vs batch size (ctx 1024)')
for b in (1, 4, 16, 64, 256):
    c = cm.step_cost(StepProfile(num_tokens=b, num_reqs=b, query_lens=[1]*b, seq_lens=[1024]*b))
    print(f'  batch {b:4d}: {c.duration*1000:7.3f} ms  {c.as_dict()[\"bound_by\"]:7s} {b/c.duration:9.0f} tok/s')

print('decode: duration vs context (batch 32)')
for ctx in (256, 1024, 4096, 16384, 65536):
    c = cm.step_cost(StepProfile(num_tokens=32, num_reqs=32, query_lens=[1]*32, seq_lens=[ctx]*32))
    print(f'  ctx {ctx:6d}: {c.duration*1000:7.3f} ms  {c.as_dict()[\"bound_by\"]}')
"
```

```
prefill: duration vs tokens
    128 tok:    11.00 ms  compute
    512 tok:    39.28 ms  compute
   2048 tok:   159.36 ms  compute
   8192 tok:   750.79 ms  compute

decode: duration vs batch size (ctx 1024)
  batch    1:   7.771 ms  memory        129 tok/s
  batch    4:   7.921 ms  memory        505 tok/s
  batch   16:   8.522 ms  memory       1877 tok/s
  batch   64:  10.927 ms  memory       5857 tok/s
  batch  256:  20.814 ms  compute     12299 tok/s

decode: duration vs context (batch 32)
  ctx    256:   8.122 ms  memory
  ctx   1024:   9.324 ms  memory
  ctx   4096:  14.132 ms  memory
  ctx  16384:  33.363 ms  memory
  ctx  65536: 110.288 ms  memory
```

Everything chapter [01](01-llm-inference-fundamentals.md) claimed, reproduced by arithmetic:

1. **Prefill is linear.** 64× the tokens, 68× the time.
2. **Decode is nearly flat in batch size.** 64× the batch costs 1.4× the time — so throughput
   goes up 45×. *This is why batching works*, and it is the single most important shape in LLM
   serving.
3. **The regime flips.** At batch 256 the step becomes compute-bound and the marginal return
   on batch size collapses. That is the knee a capacity plan is looking for.
4. **Long context re-dominates.** At 64k context per request, KV traffic makes a decode step
   14× slower than at 256 — which is why long-context serving is a different problem.

Read the *ratios*. Do not quote the milliseconds.

## The breakdown

`StepCost` keeps the terms rather than collapsing to a float:

```python
@dataclass(frozen=True)
class StepCost:
    duration: float
    compute_seconds: float
    memory_seconds: float
    comm_seconds: float
    encoder_seconds: float
    overhead_seconds: float
    jitter_factor: float
    flops: float
    bytes_moved: float
```

Because "seeing that a step was memory-bound at 82% weight traffic explains a latency curve in
a way one number cannot". It surfaces at `/debug/cost_model`, which reports a **window** of
recent steps rather than just the last one — one step says what one step cost; a window says
whether the run is compute- or memory-bound as a whole, and where a curve bent. The window is
bounded (64 steps) because a debug endpoint that leaks memory is worse than no debug endpoint.

`encoder_seconds` is a separate field rather than folded into compute for a specific reason:
`is_compute_bound` compares compute against memory, and an encoder term added to the compute
side flipped that verdict on any step carrying an image — reporting a memory-bound decode step
as compute-bound for a reason that has nothing to do with the decode.

## Launch overhead and graph capture

```python
KERNELS_PER_LAYER_EAGER   = 12   # qkv, attention, o-proj, two norms, three MLP matmuls, residuals
KERNELS_PER_STEP_CAPTURED = 8    # with a captured graph, the whole step is a few launches

num_kernels = (KERNELS_PER_STEP_CAPTURED * pp_size if graph_hit
               else KERNELS_PER_LAYER_EAGER * layers_local)
t_overhead = num_kernels * device.launch_overhead
```

On a small model at small batch, this term is not negligible — which is exactly why CUDA graphs
exist. `--enforce-eager` forces the eager path and also makes `graph_capture_seconds` zero.

Note `graph_capture_seconds` uses `layers_per_stage`, not the whole model's layer count: capture
is startup work done by one rank over the layers *that rank holds*, not a traversal of the
pipeline the way a step is. Using the whole-model count charged an 8-stage deployment eight times
its real capture time — and startup time is what an autoscaler's cold-start budget is set from.

## Jitter

```python
if self.jitter_sigma > 0.0 and rng is not None:
    jitter_factor = max(0.0, 1.0 + float(rng.normal(0.0, self.jitter_sigma)))
    duration *= jitter_factor
```

Multiplicative noise, **seeded**, so a run with jitter is still reproducible. Clamped at zero
because a large sigma could otherwise draw a negative multiplier and run the clock backwards.
The generator is a named engine-level stream (`"jitter"`), not a per-request one: jitter is a
property of the step, not of any request.

Use it when you want a latency *distribution* rather than a single value — a p99 that is not
identical to the mean.

## What is not modeled, named

The project's honesty is concentrated here. Four gaps, each stated rather than approximated:

**Cascade attention.** The common-prefix block count is computed and carried through
`SchedulerOutput` and the attention metadata exactly as upstream does, and you can read it at
`/debug/cost_model` — but the cost model ignores it. So shared-prefix workloads are modeled
**pessimistically**: a real backend taking the optimisation would be faster than this says.
Prefix caching itself, the much larger effect, is fully modeled.

**Pipeline microbatch overlap.** PP shards memory and layers but a batch still traverses every
stage, so the step costs the whole model's work. Real PP overlaps microbatches so the steady
state approaches one stage's time. Result: PP here is "same latency, less memory per device" —
correct for a single request, pessimistic for a saturated one. Dividing the terms by `pp_size`
(as an earlier version did) would report a step `pp_size` times faster than it is.

**Streams and overlap.** Upstream's worker overlaps H2D copies with compute on separate CUDA
streams. Modeling that would mean modeling the overlap, and a stub that pretends to be a stream
while doing nothing would misreport the very thing it was added to represent. If overlap ever
matters, it becomes an explicit term here — not a fake stream.

**CPU time.** The clock advances only for modeled *device* work. This is why
`--async-scheduling` is refused: it exists to hide the scheduler's CPU time behind the forward
pass, and porting it would move the scheduling decisions without moving any latency, so
comparing the flag on and off would report that it buys nothing — the opposite of what real
hardware says.

## Calibration

The accuracy lives in three device-card fields:

```json
"mfu": 0.45, "bw_eff": 0.8, "link_eff": 0.75
```

Achieved fraction of peak compute, memory bandwidth, and interconnect. `launch_overhead` is a
fourth. They ship as **rules of thumb, not fits** — see each card's `provenance` field.

`tools/calibrate_cost_model.py` is where they get fitted against real hardware. Note it is
referenced by the design and does **not exist in the tree yet**; until it is run, treat every
duration as shape.

`ENCODER_PARAMS = 300_000_000` deserves a mention as an example of the honesty discipline: it is
a *count* (ViT-L/14 scale), not a multiple of `hidden_size`. The earlier form modeled a
256-patch image at a tenth of a microsecond — free, against a 20 ms step. "An image was
documented as expensive and priced at nothing, which is the one direction a cost model must not
be wrong in when the question is whether to cache encoder output at all."

## How to use this honestly

**Do:**

- compare two configurations and read which way things moved;
- find the knee in a sweep;
- check that a knob has the *sign* of effect you expected;
- exercise a client's timeout behaviour with `--clock-mode real`.

**Do not:**

- quote a latency to a stakeholder;
- size a fleet from the throughput number;
- set an SLO from the p99;
- compare across cost-model profiles (`constant` and `roofline` are not on the same scale).

Every surface that reports a duration says so. The benchmark output ends with:

```
Durations are MODELED by the simulated cost model, not measured. Treat them as shape, not truth.
```

`/debug/cost_model` tags every step `"provenance": "modeled"`. Prometheus latency metrics carry
it in their help text (chapter [20](20-observability.md)):

```
NOTE: this duration is MODELED by pretending-vllm's cost model, not measured.
See the fidelity contract in the README.
```

If you find a duration anywhere in this project without such a label, that is a bug worth
reporting.

## Check yourself

- Where in the source is the `if prefill:` branch that makes prefill compute-bound?
- 64× the decode batch costs 1.4× the step. What does that imply about throughput, and why?
- At what point in the batch-size sweep above does the regime flip, and what does that mean for
  tuning?
- Why is `encoder_seconds` its own field rather than part of `compute_seconds`?
- Which four things are explicitly not modeled, and which direction does each bias the numbers?
- Which three device-card fields would you change to calibrate against your hardware?

## Next

[16. Clock and determinism](16-clock-and-determinism.md) — how modeled time is spent, or not.
