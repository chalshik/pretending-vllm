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

from dataclasses import dataclass

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


def compute_weight_bytes(model: ModelCard, dtype: str, tp_size: int = 1) -> int:
    """Parameter bytes resident on one device.

    Tensor parallelism shards the layers but not the embedding tables, so the
    embedding term is excluded from the division. Dividing everything by TP is the
    common shortcut and it understates per-device memory on models with large
    vocabularies -- 128k-vocab models put over a gigabyte in embeddings alone.
    """
    dtype_bytes = DTYPE_BYTES[dtype]
    embedding = model.embedding_parameters
    sharded = model.num_parameters - embedding
    return int(embedding * dtype_bytes + (sharded * dtype_bytes) // tp_size)


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
) -> MemoryProfile:
    """Derive the KV pool and `num_gpu_blocks`. R10.2, R10.5, R10.6."""
    capacity = device.memory_bytes
    usable = int(capacity * gpu_memory_utilization)

    weight_bytes = compute_weight_bytes(model, dtype, tp_size) // pp_size
    activation_peak = compute_activation_peak_bytes(
        model, dtype, max_num_batched_tokens, max_num_seqs, tp_size
    )

    kv_bytes_per_token = model.kv_bytes_per_token(kv_cache_dtype, tp_size) // pp_size
    kv_bytes_per_block = block_size * kv_bytes_per_token

    kv_pool = (
        usable - weight_bytes - activation_peak - non_torch_overhead_bytes - graph_bytes
    )

    # R10.5: fail at startup, not at request time, and say what to change.
    if kv_pool <= 0:
        gib = 1 << 30
        raise SimOutOfMemoryError(
            f"No memory left for the KV cache. The model's weights "
            f"({weight_bytes / gib:.2f}GiB), modeled activation peak "
            f"({activation_peak / gib:.2f}GiB), and non-torch overhead "
            f"({non_torch_overhead_bytes / gib:.2f}GiB) already exceed the "
            f"{usable / gib:.2f}GiB budget on a {capacity / gib:.2f}GiB device at "
            f"gpu_memory_utilization={gpu_memory_utilization}.\n"
            f"Try: raise gpu_memory_utilization, lower max_num_batched_tokens or "
            f"max_num_seqs, use a smaller model card, or pick a larger device card."
        )

    num_gpu_blocks = (
        num_gpu_blocks_override
        if num_gpu_blocks_override is not None
        else kv_pool // kv_bytes_per_block
    )

    # R10.6: a max_model_len that cannot fit one request is a startup error. Left to
    # request time it would look like a request that queues forever for capacity that
    # will never exist.
    blocks_for_one_request = (max_model_len + block_size - 1) // block_size
    if num_gpu_blocks < blocks_for_one_request:
        raise SimOutOfMemoryError(
            f"The KV cache holds {num_gpu_blocks} blocks, but a single request at "
            f"max_model_len={max_model_len} needs {blocks_for_one_request} "
            f"(block_size={block_size}). No request could ever be served.\n"
            f"Try: lower max_model_len, raise gpu_memory_utilization, or pick a "
            f"larger device card."
        )

    max_concurrency = num_gpu_blocks * block_size / max_model_len

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
