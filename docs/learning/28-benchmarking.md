# 28. Benchmarking and sweeps

> **Files:** [`pvllm/benchmarks/`](../../pvllm/benchmarks) — `latency.py`, `throughput.py`, `serve.py`, `sweep.py`, and `lib/` (`arrivals.py`, `metrics.py`, `runner.py`); CLI in [`entrypoints/cli/benchmark/`](../../pvllm/entrypoints/cli/benchmark)
> **Upstream:** `vllm/benchmarks/*` and `vllm/benchmarks/sweep/param_sweep.py` (Tier B)
> **Prerequisites:** chapter [15](15-cost-model.md) — especially its honesty section.

> This is the part that costs a GPU reservation otherwise.

Four subcommands, mirroring upstream's `vllm bench` layout plus a sweep runner:

```bash
pvllm bench latency     # one batch, end to end
pvllm bench throughput  # a fixed set of prompts, as fast as possible
pvllm bench serve       # arrivals from a stochastic process
pvllm bench sweep       # a grid, one tidy CSV row per cell
```

## `bench latency` — one batch

```bash
pvllm bench latency --model dense-8b --device-card datacenter-80gb \
  --cost-model-profile roofline --input-len 512 --output-len 32 --batch-size 8
```

Reports TTFT, TPOT, e2e, and queue time, each with mean/median/p90/p99. Chapter
[05](05-first-run.md) has the full output.

Two defaults that differ from a real benchmark harness, for reasons worth knowing:

- **`--num-iters` defaults low.** A run without `--jitter-sigma` is deterministic, so extra iterations
  repeat the same number exactly.
- **`--num-iters-warmup` defaults to zero.** There are no caches to warm and no kernels to autotune, so
  a warmup here only warms the *prefix cache* — "which would make the measured iterations faster than
  the first, for a reason that has nothing to do with what is being measured."

## `bench serve` — arrivals as a process

```bash
pvllm bench serve --model dense-8b --device-card datacenter-80gb \
  --cost-model-profile roofline --request-rate 8 --burstiness 1.0 --num-prompts 100
```

Requests do not arrive in a batch in production; they arrive as a stochastic process, and **the
queueing behaviour a benchmark is trying to measure is a function of that.** Inter-arrival gaps are
sampled from a gamma distribution parameterised by `--request-rate` and `--burstiness`, which is
upstream's arithmetic exactly.

`--burstiness` is the gamma shape: `1.0` is Poisson, below 1 is burstier, above 1 is more regular. Real
traffic is usually burstier than Poisson, and burstiness is what turns a comfortable average into a p99
problem.

**One deliberate change from upstream:** the generator comes from `RngFactory`, so a benchmark run is
reproducible from its seed. Upstream draws from the global `np.random`, "which is fine when you are
measuring hardware — the noise averages out over a real run. Here it would not: a simulated run is
otherwise exactly reproducible, and an arrival process that varied between runs would be the only
source of variance in the whole system, which would make every A/B comparison of two configs partly a
comparison of two different workloads."

The purity lint enforces it: `np.random` is unreachable from `arrivals.py`, *including in a type
annotation*, so there is no version of that file that quietly regresses. Hence the `GammaSource`
`Protocol` — the generator arrives as a structural type, the same way `Clock` and `TraceSink` cross
their boundaries (chapter [03](03-simulation-boundary.md)).

### The split that matters

`bench serve` reports **TTFT split into queue wait and prefill**:

> the distinction a capacity decision turns on, since a request that waited 200 ms and prefilled in 5
> needs more concurrency while one that prefilled for 205 ms needs a smaller batch.

Same TTFT, opposite fixes. This is the single most useful number in the whole benchmark suite, and it
is the reason to reach for `bench serve` over `bench latency`.

## `bench sweep` — the reason the project exists

```bash
pvllm bench sweep --model dense-8b --device-card datacenter-80gb \
  --cost-model-profile roofline --max-model-len 4096 \
  --input-len 512 --output-len 16 --num-prompts 8 \
  --sweep max-num-seqs=1,2,4,8 -o sweep.csv
```

```
seqs  out_tok/s   ttft_ms  queue_ms  tpot_ms  steps
   1     122.99     494.6     455.3     6.05    128
   2     190.44     328.9     252.1     6.08     64
   4     262.39     273.9     122.0     6.13     32
   8     323.51     302.2       0.0     6.23     16
```

Read that table the way a capacity plan should:

- **Throughput rises** 123 → 324 output tok/s, but with falling returns (1.55×, 1.38×, 1.23×).
- **Queue time collapses** 455 → 0 ms. At `max_num_seqs=8` the whole workload fits in the batch, so
  nothing waits.
- **TTFT falls, then rises.** 494 → 274 → **302**. That inflection between 4 and 8 is **the knee**:
  past it, the queue is already empty and a larger batch only makes each step slower.
- **TPOT creeps up** 6.05 → 6.23 ms — the price of the bigger batch, and small, because decode is
  memory-bound (chapter [15](15-cost-model.md)).
- **Step count halves** each time, which is the mechanical explanation for all of the above.

Everything else in the chapter is machinery. *This table is the deliverable*: which way each knob moves
things, and where it stops helping.

### Sweepable parameters

```
max-num-seqs             max-num-batched-tokens    block-size
gpu-memory-utilization   device-card               enable-prefix-caching
enable-chunked-prefill   request-rate              num-prompts
input-len                output-len
```

Multiple `--sweep` flags form a **grid**. Output is **tidy CSV** — one row per cell, one column per
variable — "because a sweep over two parameters is already a shape nobody wants to reshape by hand, and
every plotting library takes long form."

Every row carries `provenance=modeled`, so a CSV that escapes into a spreadsheet still says what it is.

## Reading results honestly

Every surface here repeats the same warning, and it is not decoration:

```
Benchmark durations are MODELED by the simulated cost model, not measured (R9.5). They reproduce
qualitative regimes -- where the knee is, which way a knob moves things -- and will not tell you
your p99.
```

**Do:**

- find the knee (as above);
- check the *sign* of an effect before spending GPU hours confirming it;
- compare two configurations against each other;
- rule out configurations that cannot fit at all — that part is analytic and exact (chapter
  [14](14-memory-model.md));
- exercise a client against `--clock-mode real` to test its timeout behaviour.

**Do not:**

- quote the throughput number;
- set an SLO from the p99;
- compare across `--cost-model-profile` values;
- believe a shared-prefix workload's latency — cascade attention is unmodeled, so those are
  pessimistic (chapter [15](15-cost-model.md)).

Two more properties, both stated in the source:

- **Arrivals are seeded from `--seed`**, so rerunning a comparison reruns the same workload — unlike
  upstream, where the arrival process draws from global random state.
- **A sweep cell is a fresh engine.** Each cell re-runs startup, so per-cell startup cost is included
  and the prefix cache does not leak between cells.

## The workflow this enables

```bash
# 1. does it fit at all?  (analytic — trust this)
pvllm bench latency --model dense-70b --tensor-parallel-size 8 \
  --device-card datacenter-80gb --input-len 128 --output-len 8

# 2. where is the knee for concurrency?  (shape — trust the shape)
pvllm bench sweep --model dense-70b --tensor-parallel-size 8 \
  --cost-model-profile roofline --sweep max-num-seqs=8,16,32,64,128 -o seqs.csv

# 3. what does the step budget do at that concurrency?
pvllm bench sweep --model dense-70b --tensor-parallel-size 8 --max-num-seqs 32 \
  --cost-model-profile roofline --sweep max-num-batched-tokens=1024,2048,4096,8192 -o tokens.csv

# 4. how much does prefix caching buy on MY prompt shape?
pvllm bench sweep --model dense-70b --tensor-parallel-size 8 \
  --sweep enable-prefix-caching=true,false -o cache.csv

# 5. under a realistic arrival process, what is the queue doing?
pvllm bench serve --model dense-70b --tensor-parallel-size 8 --max-num-seqs 32 \
  --request-rate 12 --burstiness 0.5 --num-prompts 200
```

Five experiments, minutes on a laptop, no reservation. Then take the two or three candidate
configurations to real hardware and measure the numbers you were never going to get from here.

## Check yourself

- Why does `--num-iters-warmup` default to zero?
- What does `--burstiness` control, and why does it matter more than `--request-rate` for p99?
- Why does this project's arrival process use a seeded generator where upstream uses global
  `np.random`?
- In the sweep table, why does TTFT fall then rise, and what is that inflection called?
- Which single split in `bench serve`'s TTFT tells you whether to add concurrency or reduce batch size?
- Which of a sweep's outputs can you trust as a number rather than a shape?

## Next

[29. Conformance and fidelity](29-conformance-and-fidelity.md) — how the contract is checked.
