# The pretending-vllm learning path

A tutorial series that starts from "what is LLM inference" and ends with "I can add a
feature to this engine and defend it in review."

Every chapter teaches **two things at once**:

1. **How real vLLM works** — the concepts, the data structures, the decisions. The
   target is vLLM **v0.27.1**, the version this repository is pinned to
   ([UPSTREAM.md](../../UPSTREAM.md)). Where upstream has changed recently, the
   chapters say so, because a lot of tutorials on the internet describe the V0 engine
   that no longer exists.
2. **How this repository stands in for it** — which files, why they exist, and exactly
   where the real engine stops and the simulator starts.

That second half is the point of the project: everything above the device is real vLLM
logic, and only the device and the model are fake. So reading this series is a way of
reading vLLM itself, with a debugger you can actually run on a laptop.

## How to read it

Chapters are numbered and build on each other. If you read them in order, nothing is
referenced before it is explained.

| # | Chapter | You will understand |
|---|---|---|
| [00](00-orientation.md) | Orientation | What this project is, why it exists, and the one diagram that explains it |
| [01](01-llm-inference-fundamentals.md) | LLM inference fundamentals | Prefill, decode, the KV cache, and why batching an LLM is unusual |
| [02](02-vllm-v1-architecture.md) | The vLLM V1 architecture | The layers of a real vLLM engine and what each one owns |
| [03](03-simulation-boundary.md) | The simulation boundary | What is real, what is fake, and how the seam is enforced |
| [04](04-repository-tour.md) | Repository tour | Every directory and file, and why it is there |
| [05](05-first-run.md) | Your first run | Install, generate, serve, and read a trace |
| [06](06-configuration.md) | Configuration | `EngineArgs` → `VllmConfig`, and what gets *derived* |
| [07](07-requests-and-sampling.md) | Requests and sampling | `Request`, `SamplingParams`, the status machine, output types |
| [08](08-tokenizers.md) | Tokenizers and detokenization | Byte-level mock vs. a real tokenizer, and incremental decoding |
| [09](09-kv-cache-blocks.md) | KV cache blocks | Paged attention's bookkeeping: blocks, refcounts, eviction order |
| [10](10-prefix-caching.md) | Prefix caching | Block hashing, the chain rule, cache keys, hit rates |
| [11](11-hybrid-kv-groups.md) | Hybrid KV cache groups | Sliding windows, MLA, Mamba, and why one pool serves them all |
| [12](12-scheduler.md) | The scheduler | Continuous batching, chunked prefill, preemption — the centerpiece |
| [13](13-worker-and-model-runner.md) | Worker and model runner | Persistent batch, attention metadata, slot mapping |
| [14](14-memory-model.md) | The memory model | Where `num_gpu_blocks` comes from, and what a capacity answer rests on |
| [15](15-cost-model.md) | The cost model | The roofline, the regimes it reproduces, and how it can mislead you |
| [16](16-clock-and-determinism.md) | Clock and determinism | Virtual/real/scaled time, seeded RNG, the purity rules |
| [17](17-engine-core-and-frontends.md) | Engine core and frontends | `step()`, output processing, `LLMEngine`, `AsyncLLM` |
| [18](18-multiprocess-engine.md) | The multiprocess engine | ZeroMQ, msgspec, backpressure, and the determinism trade |
| [19](19-openai-server.md) | The OpenAI server | Endpoints, streaming, cancellation, error envelopes |
| [20](20-observability.md) | Observability | Prometheus metrics, JSONL traces, the timeline viewer, `/debug/*` |
| [21](21-structured-output.md) | Structured output | Async grammar compilation and admission gating |
| [22](22-lora.md) | LoRA | Adapter slots as a queueing constraint, and cache partitioning |
| [23](23-multimodal.md) | Multimodal | Placeholders, the encoder budget, the encoder cache |
| [24](24-parallelism.md) | Parallelism | TP, PP, DP, EP — what each one actually buys |
| [25](25-speculative-decoding.md) | Speculative decoding | Draft/verify accounting, and the one number you must measure |
| [26](26-kv-disaggregation.md) | KV disaggregation | Connectors, external prefix stores, pull-vs-recompute |
| [27](27-pooling-and-embeddings.md) | Pooling and embeddings | A request that prefills and stops |
| [28](28-benchmarking.md) | Benchmarking and sweeps | `pvllm bench`, arrival processes, reading a sweep honestly |
| [29](29-conformance-and-fidelity.md) | Conformance and fidelity | The C1–C7 contract and how it is checked |
| [30](30-testing-and-tooling.md) | Testing and tooling | Purity lint, mutation catalogue, spec sync, goldens |
| [31](31-extending-the-port.md) | Extending the port | Adding a feature, bumping the pin, refusing loudly |

## Shorter paths

- **"I just want to understand vLLM."** 01 → 02 → 09 → 10 → 12 → 13 → 17. About an
  hour, and it covers the parts that make vLLM different from a naive server.
- **"I need to capacity-plan a deployment."** 01 → 09 → 11 → 14 → 15 → 24 → 28. This
  is the path that ends in numbers, so read 15's honesty section twice.
- **"I am driving this engine from a product."** 05 → 07 → 19 → 20 → 16. What the HTTP
  surface guarantees, what it reports, and how to make time real.
- **"I want to contribute."** 03 → 04 → 12 → 29 → 30 → 31.

## Conventions

Each chapter opens with a box like this:

> **Files:** the source this chapter explains
> **Upstream:** the vLLM counterpart and its fidelity tier (A–D, see
> [chapter 03](03-simulation-boundary.md))

Inside:

- **Why it exists** — the problem, before the solution.
- **How it works** — the mechanism, with links into the source.
- **Real vLLM vs. pretending-vllm** — every divergence, named. If a chapter has nothing
  in this section, nothing diverges.
- **Try it** — commands that run on a laptop with no GPU. Output shown was produced by
  running them.
- **Check yourself** — questions worth being able to answer before moving on.

Two labels appear throughout, and they mean specific things:

- **`[modeled]`** — a number produced by the cost model. It has the right shape and the
  wrong value. Never quote one as a measurement.
- **`[exact]`** — a decision or a count that must match real vLLM, per the fidelity
  contract. A divergence here is a bug by definition.
