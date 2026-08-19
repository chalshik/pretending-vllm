# 25. Speculative decoding

> **Files:** [`pvllm/config/speculative.py`](../../pvllm/config/speculative.py), the draft paths in [`sched/scheduler.py`](../../pvllm/v1/core/sched/scheduler.py) and [`worker/gpu/model_runner.py`](../../pvllm/v1/worker/gpu/model_runner.py), `accepted_draft_count` / `propose_drafts` in [`sim/model.py`](../../pvllm/sim/model.py)
> **Upstream:** `vllm/config/speculative.py` (Tier C); the scheduler paths are Tier **A**
> **Prerequisites:** chapters [12](12-scheduler.md), [13](13-worker-and-model-runner.md).

A small draft model proposes `k` continuations; the target model verifies all of them in **one**
forward pass. Every accepted draft is a token that cost no extra step.

The win is real and large when acceptance is high. When it is low, the verification work is wasted and
throughput **drops**. Which side of that trade a deployment lands on is the question, and it is why
this is modeled rather than stubbed.

## How it falls out of `num_computed_tokens`

No special mode. From chapter [12](12-scheduler.md):

```python
num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
```

where

```python
@property
def num_tokens_with_spec(self) -> int:
    return len(self._all_token_ids) + len(self.spec_token_ids)
```

A decoding request with three drafts in hand asks for **4** tokens: one for the position the model is
at, plus one per draft. "That is what makes it fall out of the same arithmetic every other request
uses, rather than needing a decode mode."

## The step, end to end

```
scheduler:  request has 3 drafts       → schedules 1 + 3 = 4 tokens
            records them in SchedulerOutput.scheduled_spec_decode_tokens
            clears request.spec_token_ids (a draft not verified now is stale)

runner:     accepted = sim_model.accepted_draft_count(req_id, 3)     # say 2
            emits accepted + 1 = 3 tokens
            proposes the next round's drafts → ModelRunnerOutput.spec_token_ids

scheduler:  rejected = 3 - 2 = 1
            request.num_computed_tokens -= 1          ← the rollback
            appends the 3 tokens one at a time, checking stop after each
            stores the new drafts on the request
```

Three details, each of which is a bug if missed.

**The rollback is mandatory.**

```python
rejected = num_drafts - max(0, len(generated) - 1)
if rejected > 0:
    request.num_computed_tokens -= rejected
```

> Everything in between was computed against drafts the target rejected, so it has to come back off
> the count — otherwise `num_computed_tokens` runs ahead of the request's actual history and the next
> step schedules a negative number of tokens, which is a loop that never terminates rather than an
> error that says anything.

**Speculation is lossless.** The runner emits the accepted prefix **plus one**: the target model's own
token at the first rejected position. So the output is the same sequence either way, produced in fewer
steps. And in this simulator, `propose_drafts` returns *the ids the request would emit anyway*, so a
run with speculation on produces the same text as one with it off — "which is the property that makes
the two comparable at all."

**Drafts do not survive a disturbance.** They are cleared on preemption ("proposed against a KV cache
this request no longer has"), and cleared when the step's budget trims the batch ("a draft proposed
against a token that has since been superseded is not a draft any more").

**Drafts are stored on the request, not carried in the output**, because whether they are still usable
depends on what the scheduler does next.

## Acceptance: the prefix rule

```python
def accepted_draft_count(self, request_id: str, num_drafts: int) -> int:
    rng = self.rng_factory.for_request(request_id)
    accepted = 0
    for _ in range(num_drafts):
        if float(rng.random()) >= self.spec_acceptance_rate:
            break  # ← stop at the FIRST rejection
        accepted += 1
    return accepted
```

Drawn per draft position, **stopping at the first rejection** — which is what verification actually
does. A run of drafts is accepted only as a *prefix*, because each one conditions the next: rejecting
the second invalidates the third whatever the target thought of it.

That prefix rule is why the expected accepted count is

```
E[accepted] = Σ  rate^i   for i in 1..k
```

rather than `k × rate`, and why **the return on `num_speculative_tokens` falls off so sharply once
acceptance drops** — the curve a product tuning it needs to see.

| acceptance | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| 0.9 | 0.90 | 1.71 | 3.10 | 5.13 |
| 0.7 | 0.70 | 1.19 | 1.77 | 2.20 |
| 0.5 | 0.50 | 0.75 | 0.94 | 1.00 |
| 0.3 | 0.30 | 0.39 | 0.43 | 0.43 |

At 0.5 acceptance, going from k=4 to k=8 buys 0.06 extra tokens per step and costs eight positions of
verification work in every step. That is the shape you should be able to predict before you tune.

## Disabling by batch size

```python
if (
    self.spec_disable_by_batch_size is not None
    and len(self.running) > self.spec_disable_by_batch_size
):
    request.spec_token_ids = []
```

Upstream's knob, and its reasoning: "verification stops paying once the batch is large enough to
saturate the device on its own, because the wasted work competes with real decodes."

Checked **per step against the running count**, so a burst of traffic disables it and a lull turns it
back on. This is a knob that changes the answer, which is why it is modeled.

## The number you must measure yourself

```python
#: R14. How often a draft token is accepted, under speculative decoding. The
#: one number a simulator cannot derive: it is the agreement between a draft
#: model and a target model, and there is neither. Measure it on your real pair
#: and set it here; everything downstream -- the scheduling, the token
#: accounting, the spec_decode metrics -- is then faithful.
spec_acceptance_rate: float = 0.7
```

This is one of exactly **two** knobs in the whole project that a simulator cannot derive from anything
(the other is structured-output backend conformance, chapter [21](21-structured-output.md)).

The honest workflow: measure acceptance on your real draft/target pair once — a few hundred requests
on a GPU is enough — then sweep everything else here for free.

```bash
pvllm bench sweep --model dense-8b --device-card datacenter-80gb \
  --cost-model-profile roofline --spec-acceptance-rate 0.85 \
  --sweep max-num-seqs=1,4,16,64 -o spec.csv
```

## Proposal methods

```python
SPECULATIVE_METHODS = (
    "ngram",
    "eagle",
    "eagle3",
    "medusa",
    "mlp_speculator",
    "draft_model",
)
```

Each needs a real draft model or a real n-gram index over real text, so **none of them is executed**
here — "the field exists so a config round-trips, and the simulator proposes synthetic drafts whatever
it says." The method name therefore has no effect on the numbers. That is a shape-only fidelity, and
it is declared as such rather than implied.

`num_speculative_tokens` above 16 is refused, on the grounds that it is beyond what any real draft
model configuration uses.

## What it reports

```
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_accepted_tokens_total
```

Cumulative counters, so the ratio is your realised acceptance. The scheduler maintains them in
`update_from_output` alongside the rollback, which is the only place both numbers are known.

## What to look for in a trace

Speculation shows up as **fewer steps for the same tokens**:

```bash
pvllm trace view spec.jsonl
#   steps=61  tokens=1368     ← with speculation at high acceptance
#   steps=96  tokens=1368     ← without
```

The token count is identical (lossless), the step count is not. If your step count barely moves,
acceptance is too low to be paying for itself — and the wasted verification is making every other
request in the batch slightly slower.

## Check yourself

- Why does a request with three drafts ask the scheduler for four tokens?
- What is rolled back when a draft is rejected, and what happens if you skip it?
- Why is the accepted count a *prefix* rather than a count of independent successes?
- At acceptance 0.5, why does raising `k` from 4 to 8 buy almost nothing?
- Why is speculation disabled above a batch size?
- Which single number must you measure on real hardware for the rest of this to be faithful?

## Next

[26. KV disaggregation](26-kv-disaggregation.md).
