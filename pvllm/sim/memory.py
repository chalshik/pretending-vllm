"""The memory ledger and the analytic sizing that derives num_gpu_blocks. R10.

Upstream: (none -- simulator)
Tier: D

This is where "hardware becomes a JSON file" pays off: given a model card and a device
card, how many KV blocks fit, and how many concurrent requests at what context length.

**One honest caveat, and it matters.** The fidelity contract calls memory "analytic,
exact given the model card and device card". That is true of every term here *except
the activation peak*. Upstream measures it: `determine_available_memory` runs a real
profiling forward pass at `max_num_batched_tokens` and reads the allocator's high-water
mark. There is nothing to measure here, so it is estimated from the architecture with
the coefficients below.

The consequence is worth stating plainly, because it propagates: `num_gpu_blocks` is
exact *given* an activation peak, and the activation peak is modeled. On a large model
the term is small next to the weights and the error is negligible; on a small model at
a large batch it can be the difference of a few percent of blocks. `ACTIVATION_*`
coefficients are the knobs, and `MemoryProfile.activation_is_modeled` is True so
anything reporting these numbers can say so.

Everything else -- weights, KV bytes per block, the utilization budget -- is arithmetic
on declared quantities and is exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pvllm.logger import init_logger
from pvllm.sim.hardware_db import DeviceCard
from pvllm.sim.model_db import DTYPE_BYTES, ModelCard

logger = init_logger(__name__)

# --- activation-peak coefficients ------------------------------------------
#
# Live tensors at the peak of one transformer layer's forward pass, as multiples of
# the token count. Rules of thumb, not measurements: roughly, hidden-sized residual
# plus q/k/v/o projections in flight, and the gate/up pair in the MLP. Adjust these
# rather than the formula if calibration shows a systematic gap.
ACTIVATION_HIDDEN_MULTIPLIER = 6
ACTIVATION_INTERMEDIATE_MULTIPLIER = 2

#: Allocator fragmentation, CUDA context, NCCL buffers -- everything real vLLM finds
#: already resident before it allocates anything. Upstream measures this too.
DEFAULT_NON_TORCH_OVERHEAD_BYTES = 1 << 30  # 1 GiB


class SimOutOfMemoryError(RuntimeError):
    """Raised when the ledger cannot satisfy an allocation. R10.1.

    Shaped like the upstream message so a product that pattern-matches on OOM text
    behaves the same way against either engine.
    """


@dataclass(frozen=True)
class MemoryProfile:
    """The resolved memory picture for one device. R10.2."""

    capacity_bytes: int
    usable_bytes: int
    weight_bytes: int
    activation_peak_bytes: int
    non_torch_overhead_bytes: int
    graph_bytes: int
    kv_pool_bytes: int
    kv_bytes_per_token: int
    kv_bytes_per_block: int
    num_gpu_blocks: int
    max_concurrency: float

    #: Always True: the activation term is estimated, not measured. Surfaced so
    #: anything reporting these numbers can label them (R9.5's spirit, applied to
    #: memory).
    activation_is_modeled: bool = True

    def summary(self) -> str:
        """The startup line, in the shape upstream logs (R10.4)."""
        gib = 1 << 30
        return (
            f"Memory profile: capacity={self.capacity_bytes / gib:.2f}GiB, "
            f"usable={self.usable_bytes / gib:.2f}GiB, "
            f"weights={self.weight_bytes / gib:.2f}GiB, "
            f"activation_peak={self.activation_peak_bytes / gib:.2f}GiB (modeled), "
            f"non_torch={self.non_torch_overhead_bytes / gib:.2f}GiB, "
            f"kv_pool={self.kv_pool_bytes / gib:.2f}GiB, "
            f"num_gpu_blocks={self.num_gpu_blocks}, "
            f"max_concurrency={self.max_concurrency:.2f}x"
        )


def compute_weight_bytes(
    model: ModelCard, dtype: str, tp_size: int = 1, ep_size: int | None = None
) -> int:
    """Parameter bytes resident on one device.

    Tensor parallelism shards the layers but not the embedding tables, so the
    embedding term is excluded from the division. Dividing everything by TP is the
    common shortcut and it understates per-device memory on models with large
    vocabularies -- 128k-vocab models put over a gigabyte in embeddings alone.

    R13.4. Under expert parallelism the experts are divided differently from
    everything else: each device owns whole experts rather than a slice of every one,
    across `ep_size = data_parallel_size * tensor_parallel_size` devices, while
    attention and the norms keep sharding by `tp_size`. For a sparse MoE that is the
    dominant term by a wide margin -- Mixtral-8x7B is 46.7B parameters of which 45.1B
    are experts -- so it is the difference between the model fitting and not.
    """
    dtype_bytes = DTYPE_BYTES[dtype]
    embedding = model.embedding_parameters
    # The router is excluded: it is a per-layer linear every rank computes in full,
    # so it belongs with the dense weights however the experts are divided.
    experts = model.num_hidden_layers * model.expert_parameters_per_layer
    dense = model.num_parameters - embedding - experts
    # `None` means "not expert-parallel", and the experts then shard by `tp_size`
    # like every other layer -- which is what tensor parallelism does to an MoE. A
    # default of 1 here would leave them *unsharded* whenever EP was off, so a
    # `--tensor-parallel-size 8` MoE would report 85 GiB per device instead of 11 and
    # refuse to start on hardware that fits it.
    expert_divisor = tp_size if ep_size is None else max(1, ep_size)
    if ep_size is None or not model.is_moe:
        local_experts = experts // expert_divisor
    else:
        # Ceiling, and per *expert* rather than per byte: a rank owns whole experts,
        # so 8 experts over 3 ranks is 3/3/2 and the device that has to fit the model
        # is the one holding 3. Flooring would report the average and promise a fit
        # the busiest rank does not have -- and at ep > num_experts it would report
        # zero expert bytes for a rank that still holds one whole expert.
        per_rank = -(-model.num_experts // expert_divisor)
        one_expert = model.expert_parameters_per_layer // max(1, model.num_experts)
        local_experts = model.num_hidden_layers * per_rank * one_expert
    return int(
        embedding * dtype_bytes
        + (dense * dtype_bytes) // tp_size
        + local_experts * dtype_bytes
    )


def compute_activation_peak_bytes(
    model: ModelCard,
    dtype: str,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    tp_size: int = 1,
) -> int:
    """Estimate the activation high-water mark. R10.3.

    **Modeled, not measured** -- see the module docstring.

    The logits buffer is called out separately because it is frequently the largest
    single activation and scales with vocabulary rather than with hidden size: a
    128k-vocab model sampling 256 sequences holds 128 MiB of fp32 logits, which
    dwarfs the per-token activations of a small model.
    """
    dtype_bytes = DTYPE_BYTES[dtype]
    hidden_local = model.hidden_size
    intermediate_local = model.intermediate_size // tp_size

    per_token = (
        hidden_local * ACTIVATION_HIDDEN_MULTIPLIER
        + intermediate_local * ACTIVATION_INTERMEDIATE_MULTIPLIER
    ) * dtype_bytes
    activations = max_num_batched_tokens * per_token

    # Logits are computed in fp32 regardless of the model dtype, as upstream does.
    logits = max_num_seqs * model.vocab_size * 4

    return int(activations + logits)


def compute_lora_bytes(
    model: ModelCard,
    dtype: str,
    max_loras: int,
    max_lora_rank: int,
    num_target_modules: int = 4,
    tp_size: int = 1,
    pp_size: int = 1,
) -> int:
    """Device memory the resident LoRA adapters occupy. R16.1.

    A LoRA layer replaces a `[d_in, d_out]` update with `A @ B`, where `A` is
    `[d_in, r]` and `B` is `[r, d_out]`. For the attention projections both
    dimensions are the hidden size, so one adapted projection costs `2 * r * d`
    parameters, and an adapter costs that for every targeted projection in every
    layer.

    **Which projections are targeted is a config-wide assumption here**, defaulting to
    the four attention projections. Upstream reads it from each adapter's own config,
    so an adapter that also targets the MLP costs more than this reports -- the MLP
    projections are several times wider. The direction of the error is optimistic,
    which is worth knowing when the answer is a capacity number.
    """
    if max_loras < 1 or max_lora_rank < 1:
        return 0
    hidden = model.hidden_size
    # `A` is `[d_in, r]` and `B` is `[r, d_out]`. Under tensor parallelism only *one*
    # of them shards -- a column-parallel projection splits `B` and replicates `A`, a
    # row-parallel one the reverse -- so half the adapter is replicated on every
    # rank. Dividing the whole thing by `tp_size` understated per-device memory by
    # nearly a factor of two at high TP, in the optimistic direction: the engine
    # reported KV capacity that does not exist.
    sharded = max_lora_rank * hidden // tp_size
    replicated = max_lora_rank * hidden
    per_projection = sharded + replicated

    # Layers divide across pipeline stages, so a stage holds only its own share of
    # each adapter. Charging every stage the full set overstated the cost by
    # `pp_size` and cost real KV blocks at high PP.
    layers_local = max(1, -(-model.num_hidden_layers // pp_size))
    per_adapter = per_projection * num_target_modules * layers_local
    return per_adapter * DTYPE_BYTES[dtype] * max_loras


def windowed_blocks_for_one_request(
    sliding_window: int, block_size: int, max_model_len: int, max_in_flight_tokens: int
) -> int:
    """Blocks a windowed request holds at its *peak*. R6.7, R10.6.

    Not `ceil((window - 1) / block) + 1`. That is the steady state, and a request
    passes through a larger one on the way: eviction runs in `update_from_output`,
    *after* the step has already allocated slots for everything it scheduled, so a
    prefill chunk of `max_num_batched_tokens` is resident alongside the window before
    anything is given back.

    Under-counting here is not a small error in a reported figure -- it is a silent
    hang. R10.6's startup guard compares the pool against this number, so a pool that
    clears the steady state but not the peak passes startup and then never schedules
    the request: `allocate_slots` returns `None` every step, forever, with no error
    and no log line. A window of 64 against a 1024-token step budget needs 69 blocks
    and the old arithmetic asked for 5.
    """
    peak_tokens = min(sliding_window - 1 + max_in_flight_tokens, max_model_len)
    return -(-peak_tokens // block_size) + 1


def state_blocks_for_one_request(
    block_size: int, max_model_len: int, max_in_flight_tokens: int
) -> int:
    """State pages a recurrent request holds at its *peak*. R6.7, R10.6.

    A recurrent state is one page: position N's state already summarises every token
    before it, so `MambaManager` frees every snapshot older than the newest and the
    request settles back to a single live page whatever its context length. That is
    the capacity claim state-space layers exist to make, and charging
    `ceil(max_model_len / block_size)` -- which is what fell out of inheriting full
    attention's accounting -- contradicted it by a factor that grew with context.

    The peak is above the resting one for the same reason a window's is: eviction
    runs in `update_from_output`, after the step has allocated slots for everything
    it scheduled, so a prefill chunk's worth of boundaries is briefly resident
    alongside the live state. Worst case is a request sitting one token below a block
    boundary when a full chunk lands, which is `1 + ceil(chunk / block_size)` -- the
    same shape as a window of one token, because that is what a recurrent state is.
    """
    peak = 1 + -(-min(max_in_flight_tokens, max_model_len) // block_size)
    return min(peak, -(-max_model_len // block_size))


def compute_memory_profile(
    model: ModelCard,
    device: DeviceCard,
    *,
    dtype: str,
    kv_cache_dtype: str | None,
    block_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    tp_size: int = 1,
    pp_size: int = 1,
    non_torch_overhead_bytes: int = DEFAULT_NON_TORCH_OVERHEAD_BYTES,
    graph_bytes: int = 0,
    num_gpu_blocks_override: int | None = None,
    lora_bytes: int = 0,
    sliding_window: int | None = None,
    kv_cache_groups: Sequence[Any] | None = None,
    ep_size: int = 1,
) -> MemoryProfile:
    """Derive the KV pool and `num_gpu_blocks`. R10.2, R10.5, R10.6.

    `kv_cache_groups` is the resolved group layout (R6.7). Passed rather than
    re-derived because the engine core sizes the pool from exactly this, and the two
    disagreeing is how a capacity number becomes a lie: the profile would report one
    `num_gpu_blocks` in the startup line and the scheduler would be handed another.
    A dense model has one group holding every layer, which is the case the older
    `sliding_window` argument covers on its own.
    """
    capacity = device.memory_bytes
    usable = int(capacity * gpu_memory_utilization)

    # Pipeline stages are treated as equal shares, which they are not: stage 0 also
    # holds the embedding table and the last holds the lm_head, and on a large-vocab
    # model those are over a gigabyte. So this understates the end stages and
    # overstates the middle ones. The error is bounded by the embedding size and
    # shrinks as `pp_size` falls; a capacity plan for a 128k-vocab model at high
    # `pp_size` should treat the end stages as tighter than reported.
    # Ceiling, not floor: with 28 layers over 8 stages the busiest stage holds 4,
    # not 3. Flooring reported a pool the engine could not actually build, and
    # `num_gpu_blocks` then disagreed with the pool the scheduler was handed.
    layers_local = -(-model.num_hidden_layers // pp_size)
    # R13.4. `ep_size > 1` means expert-parallel; `None` tells `compute_weight_bytes`
    # to shard the experts by `tp_size` like everything else. Passing 1 would leave
    # them unsharded, which is the trap that function's own comment names.
    weight_bytes = (
        compute_weight_bytes(
            model, dtype, tp_size, ep_size=ep_size if ep_size > 1 else None
        )
        * layers_local
    ) // max(1, model.num_hidden_layers)
    activation_peak = compute_activation_peak_bytes(
        model, dtype, max_num_batched_tokens, max_num_seqs, tp_size
    )

    kv_bytes_per_token = model.kv_bytes_per_token(kv_cache_dtype, tp_size) // pp_size
    # R6.7. A block backs one page in each of a *group's* layers, and the groups
    # share the pool. For a dense model that is every layer and this is the whole
    # model's per-token cost; for a hybrid one it is the layers of one group, so a
    # block is smaller and the pool holds proportionally more of them.
    layers_per_group = layers_local
    if kv_cache_groups:
        layers_per_group = max(len(group.layer_names) for group in kv_cache_groups)
        # Derived the way `EngineCore._initialize_kv_caches` derives it -- per-layer
        # page size times layers per group -- rather than by rescaling the model's
        # per-token cost. The rescaling involved two integer divisions that drifted
        # whenever `num_hidden_layers % pp_size` was non-zero, so the profile printed
        # one `num_gpu_blocks` in the startup line and the scheduler was handed
        # another. Not cosmetic: R10.6's "no request could ever be served" guard ran
        # on the larger number, so a config that could not fit a single request
        # passed startup and then hung forever with no error and no log line.
        kv_bytes_per_block = (
            kv_cache_groups[0].kv_cache_spec.page_size_bytes * layers_per_group
        )
    else:
        kv_bytes_per_block = block_size * kv_bytes_per_token

    # R16.1. Adapter weights are resident on the device and come out of the same
    # budget as everything else, so serving eight adapters shrinks the KV pool.
    kv_pool = (
        usable
        - weight_bytes
        - activation_peak
        - non_torch_overhead_bytes
        - graph_bytes
        - lora_bytes
    )

    # R10.5: fail at startup, not at request time, and say what to change.
    if kv_pool <= 0:
        gib = 1 << 30
        raise SimOutOfMemoryError(
            f"No memory left for the KV cache. The model's weights "
            f"({weight_bytes / gib:.2f}GiB), modeled activation peak "
            f"({activation_peak / gib:.2f}GiB), LoRA adapters "
            f"({lora_bytes / gib:.2f}GiB), and non-torch overhead "
            f"({non_torch_overhead_bytes / gib:.2f}GiB) already exceed the "
            f"{usable / gib:.2f}GiB budget on a {capacity / gib:.2f}GiB device at "
            f"gpu_memory_utilization={gpu_memory_utilization}.\n"
            f"Try: raise gpu_memory_utilization, lower max_num_batched_tokens, "
            f"max_num_seqs, or max_loras, use a smaller model card, or pick a larger "
            f"device card."
        )

    num_gpu_blocks = (
        num_gpu_blocks_override
        if num_gpu_blocks_override is not None
        else kv_pool // kv_bytes_per_block
    )

    # R10.6: a max_model_len that cannot fit one request is a startup error. Left to
    # request time it would look like a request that queues forever for capacity that
    # will never exist.
    # R6.7. A windowed request never holds more than its window, so that -- not
    # max_model_len -- is what the pool has to fit. This is the whole capacity
    # argument for sliding windows: a 128k-context model with a 4k window needs 4k of
    # KV per request, not 128k.
    tokens_for_one_request = min(max_model_len, sliding_window or max_model_len)
    blocks_for_one_request = (tokens_for_one_request + block_size - 1) // block_size
    #: The null block is reserved once for the whole pool, not per request, so it
    #: comes off the pool rather than being added to each request's cost.
    usable_blocks_adjustment = 0

    if kv_cache_groups and len(kv_cache_groups) > 1:
        # R6.7. A hybrid request holds blocks in *every* group, and the groups do not
        # cost the same: a full-attention group grows with the conversation, a
        # windowed one stops at its window. The blend is the point -- a 5:1 model's
        # KV is neither bounded nor unbounded, and reporting either figure would
        # answer a capacity question with the wrong model's number.
        blocks_for_one_request = 0
        for group in kv_cache_groups:
            window = getattr(group.kv_cache_spec, "sliding_window", None)
            if window is not None and window < max_model_len:
                blocks_for_one_request += windowed_blocks_for_one_request(
                    window, block_size, max_model_len, max_num_batched_tokens
                )
                usable_blocks_adjustment = 1
            elif type(group.kv_cache_spec).__name__ == "MambaSpec":
                # R6.7. *One* live state, not one per block boundary. The earlier
                # arithmetic here was the attention branch spelled twice -- the same
                # expression on both sides of the `elif`, so the branch this model
                # exists to need changed nothing, and it charged a recurrent state as
                # linear in context, which is the shape it exists not to have.
                blocks_for_one_request += state_blocks_for_one_request(
                    block_size, max_model_len, max_num_batched_tokens
                )
                usable_blocks_adjustment = 1
            else:
                blocks_for_one_request += (max_model_len + block_size - 1) // block_size
    elif sliding_window is not None and sliding_window < max_model_len:
        blocks_for_one_request = windowed_blocks_for_one_request(
            sliding_window, block_size, max_model_len, max_num_batched_tokens
        )
        usable_blocks_adjustment = 1
    # Against the blocks that can actually be handed out. A model whose groups shed
    # reserves block 0 as the null block, so `num_gpu_blocks - 1` is the real
    # ceiling -- and comparing the raw pool let a config through whose own reported
    # `max_concurrency` was below 1.0, which is a silent hang: `allocate_slots`
    # returns `None` every step, forever. The refusal message named a number that
    # itself did not work.
    if num_gpu_blocks - usable_blocks_adjustment < blocks_for_one_request:
        raise SimOutOfMemoryError(
            f"The KV cache holds {num_gpu_blocks} blocks "
            f"({num_gpu_blocks - usable_blocks_adjustment} allocatable), but a "
            f"single request at max_model_len={max_model_len} "
            f"(window {tokens_for_one_request}) needs {blocks_for_one_request} "
            f"(block_size={block_size}). No request could ever be served.\n"
            f"Try: lower max_model_len, raise gpu_memory_utilization, or pick a "
            f"larger device card."
        )

    # Against the blocks a request actually *holds*, out of the blocks that can
    # actually be handed out. Dividing the window into the raw pool overstated
    # concurrency by up to a third on a small window -- it ignored both the straddle
    # block and the reserved null block -- and a capacity plan reads this directly.
    max_concurrency = (num_gpu_blocks - usable_blocks_adjustment) / max(
        1, blocks_for_one_request
    )

    return MemoryProfile(
        capacity_bytes=capacity,
        usable_bytes=usable,
        weight_bytes=weight_bytes,
        activation_peak_bytes=activation_peak,
        non_torch_overhead_bytes=non_torch_overhead_bytes,
        graph_bytes=graph_bytes,
        kv_pool_bytes=int(kv_pool),
        kv_bytes_per_token=kv_bytes_per_token,
        kv_bytes_per_block=kv_bytes_per_block,
        num_gpu_blocks=int(num_gpu_blocks),
        max_concurrency=max_concurrency,
    )


class MemoryLedger:
    """Named memory pools on one simulated device. R10.1.

    Exists so that an allocation that would not fit *fails*, rather than the
    simulator quietly pretending the device is bigger than its card says. That is
    the difference between answering a capacity question and assuming one away.
    """

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self._pools: dict[str, int] = {}

    @property
    def allocated_bytes(self) -> int:
        return sum(self._pools.values())

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.allocated_bytes

    def allocate(self, pool: str, num_bytes: int) -> None:
        if num_bytes < 0:
            raise ValueError(f"cannot allocate {num_bytes} bytes to pool {pool!r}")
        if num_bytes > self.free_bytes:
            gib = 1 << 30
            raise SimOutOfMemoryError(
                f"Simulated device out of memory. Tried to allocate "
                f"{num_bytes / gib:.2f}GiB for {pool!r}, but only "
                f"{self.free_bytes / gib:.2f}GiB of {self.capacity_bytes / gib:.2f}GiB "
                f"is free. Pools: "
                f"{ {k: f'{v / gib:.2f}GiB' for k, v in sorted(self._pools.items())} }"
            )
        self._pools[pool] = self._pools.get(pool, 0) + num_bytes

    def free(self, pool: str) -> None:
        self._pools.pop(pool, None)

    def get(self, pool: str) -> int:
        return self._pools.get(pool, 0)

    def pools(self) -> dict[str, int]:
        return dict(sorted(self._pools.items()))

    def __repr__(self) -> str:
        gib = 1 << 30
        return (
            f"MemoryLedger(capacity={self.capacity_bytes / gib:.2f}GiB, "
            f"allocated={self.allocated_bytes / gib:.2f}GiB, "
            f"pools={list(self._pools)})"
        )
