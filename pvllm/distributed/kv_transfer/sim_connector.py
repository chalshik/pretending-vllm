"""A connector over a simulated shared KV store. R17.2.

Upstream: vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py
Tier: B

Both halves of `KVConnectorBase` against `pvllm.sim.kv_store`. Two engines pointed at
the same store name hand KV to each other: the first writes a prompt's blocks as it
finishes them, the second finds them resident and pulls instead of prefilling.

That is disaggregated prefill, and the question it exists to answer is arithmetic:
**is pulling the KV cheaper than recomputing it?** Both sides of that comparison are
here -- the store's bandwidth and latency on one side, the cost model's prefill time
on the other -- so a deployment can be told which way the answer goes for its prompt
lengths and its store, without a GPU or a network.

The connector deliberately does *not* pretend to move bytes. What it reproduces is the
timing and the decision: which requests find their prefix externally, and what pulling
it costs on the engine's clock -- charged before the step that reads the KV, so the
transfer shows up next to the prefill it replaced rather than being free.

**What is missing, and named rather than approximated.** Loads are *synchronous*:
`get_num_new_matched_tokens` always reports `async=False`, so no request ever sits in
`WAITING_FOR_REMOTE_KVS`. Upstream supports the async shape, where a request waits
outside the running set while its KV arrives, and that changes admission behaviour --
so it is absent rather than half-built. There is also no handshake, no failure mode,
and no partial transfer: a store that goes away mid-transfer, or returns corrupt KV,
is a real failure this cannot produce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pvllm.distributed.kv_transfer.base import (
    KVConnectorBase,
    KVConnectorMetadata,
    KVConnectorRole,
)
from pvllm.logger import init_logger

if TYPE_CHECKING:
    from pvllm.config import VllmConfig
    from pvllm.v1.core.kv_cache_manager import KVCacheBlocks
    from pvllm.v1.core.sched.output import SchedulerOutput
    from pvllm.v1.request import Request

logger = init_logger(__name__)


class SimSharedStoreConnector(KVConnectorBase):
    """Reads and writes KV against a named `SimKVStore`."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole) -> None:
        super().__init__(vllm_config, role)
        from pvllm.sim.kv_store import get_store

        transfer = vllm_config.kv_transfer_config
        extra: dict[str, Any] = dict(
            transfer.kv_connector_extra_config if transfer else {}
        )
        self.store = get_store(
            str(extra.get("store_name", "default")),
            bandwidth_bytes_per_second=float(extra.get("bandwidth", 10e9)),
            latency_seconds=float(extra.get("latency", 0.001)),
            capacity_blocks=extra.get("capacity_blocks"),
        )
        #: Bytes one block occupies, for costing a transfer. Resolved late, because
        #: the KV layout is not known until the memory model has run.
        self.block_bytes = 0

        #: Pending work for the next `build_connector_meta`.
        self._pending_loads: dict[str, tuple[list[int], int]] = {}
        self._pending_saves: dict[str, tuple[list[int], int]] = {}
        #: R17.2. Cumulative modeled transfer time, for the metrics.
        self.load_seconds = 0.0
        self.save_seconds = 0.0

    def set_block_bytes(self, block_bytes: int) -> None:
        """Told by the engine core once the KV layout is resolved."""
        self.block_bytes = block_bytes

    # --- scheduler side ------------------------------------------------------

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """How much of this prompt the store already holds. R17.1.

        Only blocks *beyond* what the local prefix cache already covers: pulling KV
        the engine has in memory would be strictly worse than using it.
        """
        if not request.block_hashes or self.block_bytes == 0:
            return 0, False

        local_blocks = num_computed_tokens // self.block_size
        remaining = list(request.block_hashes[local_blocks:])
        if not remaining:
            return 0, False

        matched_blocks = self.store.longest_prefix([bytes(h) for h in remaining])
        if matched_blocks == 0:
            return 0, False

        # One block held back, for the same reason the local cache holds one back:
        # a request with nothing left to compute has no logits and no sampled token,
        # so it would never progress.
        total_blocks = len(request.block_hashes)
        if local_blocks + matched_blocks >= total_blocks:
            matched_blocks -= 1
        if matched_blocks <= 0:
            return 0, False

        # Synchronous: the transfer's modeled cost is charged inside the step that
        # needs it. Asynchronous loading is the shape upstream also supports, and it
        # would put the request in WAITING_FOR_REMOTE_KVS -- not implemented here,
        # and reported as sync rather than claimed and not done.
        return matched_blocks * self.block_size, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        """Record which blocks this step must fill from the store. R17.1."""
        # On `num_external_tokens`, not on whether `blocks` is empty -- upstream
        # documents the same trap, and it is easy to get backwards.
        if num_external_tokens <= 0:
            return
        block_ids = blocks.get_block_ids()
        if block_ids is None:
            return
        num_blocks = num_external_tokens // self.block_size
        self._pending_loads[request.request_id] = (
            list(block_ids[0][:num_blocks]),
            num_external_tokens,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Pack this step's transfers and reset. R17.1."""
        metadata = KVConnectorMetadata(
            loads=dict(self._pending_loads), saves=dict(self._pending_saves)
        )
        self._pending_loads.clear()
        self._pending_saves.clear()
        return metadata

    def request_finished(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Write this request's blocks to the store as it finishes. R17.2.

        Returns `False` for "blocks are free to release": the write is modeled as
        completing within this step, so nothing is still reading them. A real
        connector doing an async push would return `True` and hold them, which is the
        mechanism behind a prefill node keeping KV until the decode node has pulled
        it -- named here rather than implemented.
        """
        if not request.block_hashes or self.block_bytes == 0:
            return False, None

        hashes = [bytes(h) for h in request.block_hashes]
        num_bytes = len(hashes) * self.block_bytes
        self.save_seconds += self.store.write(hashes, num_bytes)
        return False, None

    # --- worker side ---------------------------------------------------------

    def start_load_kv(self, metadata: KVConnectorMetadata) -> float:
        """Pull this step's KV in. Returns the modeled duration."""
        if not metadata.loads or self.block_bytes == 0:
            return 0.0
        num_blocks = sum(len(block_ids) for block_ids, _ in metadata.loads.values())
        seconds = self.store.read(num_blocks * self.block_bytes)
        self.load_seconds += seconds
        return seconds

    def wait_for_save(self, metadata: KVConnectorMetadata) -> float:
        """Writes happen at request completion here, so this has nothing to wait on."""
        return 0.0

    def __repr__(self) -> str:
        return (
            f"SimSharedStoreConnector(store={self.store.name!r}, "
            f"role={self.role.name}, hit_rate={self.store.hit_rate:.2f})"
        )
