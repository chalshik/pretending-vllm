# 06. Configuration

> **Files:** [`pvllm/engine/arg_utils.py`](../../pvllm/engine/arg_utils.py), [`pvllm/config/`](../../pvllm/config), [`pvllm/envs.py`](../../pvllm/envs.py), [`pvllm/sim/model_db.py`](../../pvllm/sim/model_db.py), [`pvllm/sim/hardware_db.py`](../../pvllm/sim/hardware_db.py)
> **Upstream:** `vllm/engine/arg_utils.py`, `vllm/config/*` (Tier C), `vllm/envs.py` (Tier C)
> **Prerequisites:** chapter [05](05-first-run.md).

Configuration sounds like the boring chapter. It is not, for one reason: **most of the
numbers that decide how the engine behaves are derived, not set.** If you do not know
what gets derived and from what, you will spend an afternoon confused about why your
sweep did nothing.

## The pipeline

```
CLI flags  ─┐
LLM(**kw)  ─┼──► EngineArgs ──► create_engine_config() ──► VllmConfig ──► platform hook
env vars   ─┘     (flat)                                   (composite)     (last word)
```

Three stages, three different jobs.

**`EngineArgs`** ([`engine/arg_utils.py`](../../pvllm/engine/arg_utils.py)) is one flat
dataclass of everything a user can set, plus `add_cli_args` / `from_cli_args` so the same
fields are the CLI. This is the surface: `--max-num-seqs` is a field here, and so is
`--device-card`.

**`VllmConfig`** ([`config/vllm.py`](../../pvllm/config/vllm.py)) is the composite the
engine actually reads — one sub-config per concern:

```
model_config, cache_config, parallel_config, scheduler_config, device_config,
load_config, lora_config, speculative_config, structured_outputs_config,
observability_config, kv_transfer_config
```

**The platform hook** is the last word. `VllmConfig.__post_init__` calls
`current_platform.check_and_update_config(self)`, and that is where `worker_cls` stops
being `"auto"` (chapter [03](03-simulation-boundary.md)) — and, for a state-space model,
where `block_size` gets moved (chapter [11](11-hybrid-kv-groups.md)).

Order matters in `__post_init__`, and the code says why: the block-size warning below the
hook has to run *after* it, because the hook is what can change the block size.

## What gets derived

This table is the one to remember.

| Value | Derived from | Consequence if you forget |
|---|---|---|
| `max_model_len` | the model card's `max_position_embeddings` when unset | `dense-8b` defaults to **131072**, not 8192 — and `max_concurrency` drops accordingly |
| `resolved_dtype` | the card's declared dtype when `dtype="auto"` | every memory and cost number scales with it |
| `max_num_batched_tokens` | **8192** with chunked prefill, **2048** without | the whole shape of a trace |
| `num_gpu_blocks` | the memory model, at startup | chapter [14](14-memory-model.md) |
| `kv_cache_size_tokens`, `kv_cache_max_concurrency` | `num_gpu_blocks × block_size`, and blocks per request | the capacity answer |
| `max_num_encoder_input_tokens`, `encoder_cache_size` | the token budget, floored at the largest multimodal item | chapter [23](23-multimodal.md) |
| `block_size` | **raised** for a state-space model, from 16 to ~1040 | changes block counts *and* every block hash |
| `worker_cls` | the platform | the simulated worker |

The `max_model_len` one bites everybody once. Watch it happen:

```bash
python -c "
from pvllm.engine.arg_utils import EngineArgs
for kw in ({}, {'max_model_len': 8192}):
    c = EngineArgs(model='dense-8b', **kw).create_engine_config()
    print(kw, '-> max_model_len =', c.model_config.max_model_len)
"
```

```
{} -> max_model_len = 131072
{'max_model_len': 8192} -> max_model_len = 8192
```

That difference is not cosmetic. `max_concurrency` is the pool divided by the blocks one
request holds *at `max_model_len`*, so on a `dense-8b` at 80 GiB it reads **3.54x** at the
derived 131072 and **56.71x** at 8192 — same hardware, same pool, two very different
capacity answers to two different questions.

Asking for *more* than the architecture supports is an error rather than a clamp:

```
ValueError: max_model_len (200000) is larger than the maximum the model supports
(131072). Lower max_model_len.
```

## The defaults worth knowing

From [`config/cache.py`](../../pvllm/config/cache.py) and
[`config/scheduler.py`](../../pvllm/config/scheduler.py). These match upstream at the pin,
including the two that older tutorials get wrong.

| Setting | Default | Note |
|---|---|---|
| `block_size` | 16 | tokens of KV per block |
| `gpu_memory_utilization` | 0.92 | fraction of the device budget the engine may use |
| `enable_prefix_caching` | **True** | on by default since V1 |
| `prefix_caching_hash_algo` | `sha256` | not the builtin hash — that one is salted per process |
| `enable_chunked_prefill` | **True** | on by default since V1 |
| `max_num_seqs` | 1024 | concurrent *requests* in the batch |
| `max_num_batched_tokens` | 8192 (derived) | *tokens* per step |
| `max_num_partial_prefills` | 1 | how many requests may be mid-prefill at once |
| `policy` | `fcfs` | or `priority` |
| `long_prefill_token_threshold` | 0 (off) | cap on one request's share of a step |
| `seed` | 0 | reproduces the whole run |

**`max_num_seqs` and `max_num_batched_tokens` are the two budgets every step is checked
against**, and confusing them is the most common configuration mistake. The first bounds
how many requests can be *running*; the second bounds how many tokens a step may process
across all of them. A batch of 64 decoding requests uses 64 of `max_num_seqs` and 64 of
`max_num_batched_tokens`; one request prefilling a 5,000-token prompt uses 1 and 5,000.
Chapter [12](12-scheduler.md).

## `SimConfig`: the entire fake surface

Every simulator knob lives in one dataclass, reached as `vllm_config.sim_config`
([`config/device.py`](../../pvllm/config/device.py)). That is deliberate: the rest of the
config surface stays a mirror of upstream, and a reader can see everything invented in one
screen.

```python
@dataclass
class SimConfig:
    device_card: str = "datacenter-80gb"  # or a JSON path
    num_devices: int = 1

    clock_mode: ClockMode = "virtual"  # virtual | real | scaled
    time_scale: float = 1.0

    cost_model_profile: str = "constant"  # constant | roofline
    jitter_sigma: float = 0.0  # multiplicative N(0, sigma), seeded

    model_card: str | None = None  # override lookup by model name

    output_length_policy: str = "from_request"
    output_length_fixed: int = 128
    output_length_range: tuple[int, int] = (16, 256)
    output_length_lognormal: tuple[float, float] = (4.0, 0.75)
    content_policy: str = "pseudoword"

    spec_acceptance_rate: float = 0.7  # the one number you must measure yourself

    seed: int = 0
    trace_path: str | None = None
```

Three of these change what an experiment *means*:

- **`clock_mode`** — whether modeled time is actually spent. Chapter
  [16](16-clock-and-determinism.md).
- **`cost_model_profile`** — `constant` is deterministic and deliberately unrealistic (the
  default, so a test asserting on step *counts* does not depend on roofline coefficients).
  `roofline` reproduces the regimes. Chapter [15](15-cost-model.md).
- **`output_length_policy`** — the sleeper. `from_request` honours the client's
  `max_tokens`, which is what a test double should do. But a simulator where every request
  emits exactly `max_tokens` answers a *different question* than a real system: real
  requests stop early and at varying lengths, and that variation drives the batch
  composition the scheduler sees. For a capacity experiment, use `lognormal`.

Everything here is validated in `__post_init__` — a bad `clock_mode`, a negative
`jitter_sigma`, an out-of-range `spec_acceptance_rate`, an empty output-length range all
fail at construction rather than at step 40,000.

## Model cards and device cards

This is where hardware becomes a JSON file.

### A model card

[`pvllm/sim/models/dense-8b.json`](../../pvllm/sim/models/dense-8b.json):

```json
{
  "name": "dense-8b",
  "num_hidden_layers": 32, "hidden_size": 4096,
  "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
  "intermediate_size": 14336, "vocab_size": 128256,
  "max_position_embeddings": 131072,
  "dtype": "bfloat16", "architecture": "LlamaForCausalLM",
  "tie_word_embeddings": false,
  "provenance": "Representative of an 8B dense GQA decoder (Llama-3.1-8B class).
                 Dimensions are approximations of that architecture family, not a
                 verified checkpoint config."
}
```

`ModelCard` ([`sim/model_db.py`](../../pvllm/sim/model_db.py)) turns those fields into
everything the memory and cost models need — parameter counts by category, KV bytes per
token, whether the model is MoE / MLA / state-space / hybrid-attention. The bundled set:

| Card | Shape | Why it is here |
|---|---|---|
| `dense-0.6b` | 28 layers, GQA | fast tests |
| `dense-8b` | 32 layers, GQA | the reference deployment |
| `dense-70b` | 80 layers | does not fit one 80 GB card — forces parallelism |
| `moe-8x7b` | 8 experts, 2 active | 96.6% of parameters are experts → chapter [24](24-parallelism.md) |
| `mla-16b` | multi-head latent attention | KV that does *not* shard under TP → chapter [11](11-hybrid-kv-groups.md) |
| `hybrid-4b` | 5 windowed layers per full one | six KV cache groups → chapter [11](11-hybrid-kv-groups.md) |
| `hybrid-ssm-8b` | Mamba + attention | constant-size recurrent state → chapter [11](11-hybrid-kv-groups.md) |
| `tiny-test` | 2 layers, 1024 vocab | unit tests |

`ALIASES` maps common Hugging Face ids onto them, so a client asking for
`meta-llama/Llama-3.1-8B-Instruct` gets `dense-8b`. **An unknown id is an error, not a
guess.**

Read them yourself:

```bash
python -c "
from pvllm.sim.model_db import load_model_card
c = load_model_card('dense-8b')
print('params      ', round(c.num_parameters/1e9, 2), 'B')
print('active      ', round(c.num_active_parameters/1e9, 2), 'B')
print('kv/token    ', c.kv_bytes_per_token(), 'bytes')
print('kv/token tp8', c.kv_bytes_per_token(tp_size=8), 'bytes')
"
```

```
params       8.03 B
active       8.03 B
kv/token     131072 bytes
kv/token tp8 16384 bytes
```

### A device card

[`pvllm/sim/hardware/datacenter-80gb.json`](../../pvllm/sim/hardware/datacenter-80gb.json):

```json
{
  "name": "datacenter-80gb",
  "memory_bytes": 85899345920,
  "memory_bandwidth": 3350000000000,
  "peak_flops": {"bfloat16": 494700000000000, "float32": 61300000000000},
  "interconnect_bandwidth": 450000000000,
  "launch_overhead": 0.0000045,
  "load_bandwidth": 2000000000,
  "num_devices": 1,
  "mfu": 0.45, "bw_eff": 0.8, "link_eff": 0.75,
  "provenance": "Uncalibrated approximation ... published vendor peaks for the device
                 class, NOT measurements ... The mfu/bw_eff/link_eff factors are rules
                 of thumb, not fits. Replace via tools/calibrate_cost_model.py."
}
```

| Card | Memory | Bandwidth | BF16 peak | MFU |
|---|---|---|---|---|
| `datacenter-80gb` | 80 GiB | 3.35 TB/s | 495 TFLOPS | 0.45 |
| `workstation-24gb` | 24 GiB | 1.01 TB/s | 165 TFLOPS | 0.35 |
| `tiny-2gb` | 2 GiB | 0.20 TB/s | 10 TFLOPS | 0.30 |

The last three fields are efficiency factors — achieved fraction of peak compute, memory
bandwidth, and interconnect. They are the calibration surface: the cost model's accuracy
lives entirely in them, and they ship as rules of thumb. Note the `provenance` field: it
is required reading, and it exists so no number here can be quoted as a measurement by
accident.

**Write your own.** Point `--device-card ./my-card.json` at a file with your device's
published specs and the memory arithmetic becomes exact for your hardware (the *cost*
arithmetic stays modeled).

## Environment variables

[`pvllm/envs.py`](../../pvllm/envs.py) mirrors vLLM's `VLLM_*` surface one-to-one as
`PVLLM_*`. The rename is deliberate: a real vLLM installed side by side for diffing must
not be reconfigured by this one's variables, and vice versa.

| Variable | Default | Meaning |
|---|---|---|
| `PVLLM_DEBUG_INVARIANTS` | `0` | turns on block-accounting and slot-mapping assertions. The test suite always sets it |
| `PVLLM_ENABLE_V1_MULTIPROCESSING` | `0` | **off, unlike upstream** — chapter [18](18-multiprocess-engine.md) |
| `PVLLM_TRACE_PATH` | unset | where the JSONL trace goes |
| `PVLLM_USE_V2_MODEL_RUNNER` | unset | tri-state, mirroring upstream. Setting it to `0` *raises*: there is no V1 runner here to fall back to |
| `PVLLM_ATTENTION_BACKEND` | unset | exists only to **refuse**: pinning `FLASH_ATTN` asks for kernel behaviour nothing models |
| `PVLLM_PLUGINS` | unset | restricts which entry-point plugins load |
| `PVLLM_LOGGING_LEVEL`, `PVLLM_LOGGING_COLOR`, `NO_COLOR` | — | logging |

Variables are read **lazily** through a module-level `__getattr__`, exactly as upstream
does, so a test can monkeypatch `os.environ` and see the change without reimporting.

**One deliberate exception to the prefix rule:** the Responses API's response store is gated on
`VLLM_ENABLE_RESPONSES_API_STORE`, with upstream's own name and default, "so one runbook flag flips
both". Chapter [19](19-openai-server.md).

## Reading the resolved config

Two ways. In Python:

```python
from pvllm.engine.arg_utils import EngineArgs

config = EngineArgs(model="dense-8b", max_model_len=8192).create_engine_config()
print(config)  # a one-line summary
print(config.sim_config)
print(config.scheduler_config.max_num_batched_tokens)  # 8192, derived
```

Over HTTP, with `--enable-debug-endpoints`:

```bash
curl -s localhost:8000/debug/config | python -m json.tool
```

That endpoint reports the fully resolved config **including what was derived**, which is
exactly the question you want answered when a sweep behaved unexpectedly.

## Try it

Watch a derived value move:

```bash
python -c "
from pvllm.engine.arg_utils import EngineArgs
for chunked in (True, False):
    c = EngineArgs(model='dense-0.6b', enable_chunked_prefill=chunked).create_engine_config()
    print('chunked_prefill =', chunked, '-> max_num_batched_tokens =',
          c.scheduler_config.max_num_batched_tokens)
"
```

```
chunked_prefill = True -> max_num_batched_tokens = 8192
chunked_prefill = False -> max_num_batched_tokens = 2048
```

Upstream uses a smaller budget when chunking is off, because a step must then hold a
whole prompt — so a large budget would let one prompt monopolise a step with no way to
split it.

## Check yourself

- You set `--max-num-seqs 8` and nothing changed. Name two other settings that could be
  the actual binding constraint.
- Why does `max_num_batched_tokens` default lower when chunked prefill is off?
- Why must the block-size warning in `VllmConfig.__post_init__` run *after* the platform
  hook?
- Which config object would you inspect to answer "is anything about this run fake?"
- Why are the env vars `PVLLM_*` rather than `VLLM_*`?

## Next

[07. Requests and sampling](07-requests-and-sampling.md) — the object the whole engine
manipulates.
