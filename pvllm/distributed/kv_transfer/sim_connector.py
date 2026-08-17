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

        # R17.1. The store is keyed by block hash, and block hashes only exist when
        # prefix caching is on -- `Request.block_hashes` is empty otherwise, and so
        # was every lookup and every publish. The connector became a total no-op
        # with no warning: a disaggregation experiment ran, reported zero hits, and
        # looked like a measurement of a store that does not help.
        cache_config = vllm_config.cache_config
        if not cache_config.enable_prefix_caching:
            raise ValueError(
                "a KV connector requires prefix caching: the store is keyed by block "
                "hash, and no block hashes are computed with "
                "--no-enable-prefix-caching. Nothing would be published or matched."
            )
        # The *effective* window, not just the flag. M5c let a model card carry its
        # own window, and checking only `cache_config.sliding_window` meant a hybrid
        # card slipped past this refusal -- the connector then started cleanly,
        # issued no lookup ever, and reported zero hits forever. A disaggregation
        # study on exactly the model class M5c added would have concluded the store
        # buys nothing, when nothing was ever asked of it.
        from pvllm.sim.model_db import load_model_card

        windowed = cache_config.sliding_window is not None
        if not windowed:
            try:
                card = load_model_card(
                    (
                        vllm_config.sim_config.model_card
                        if vllm_config.sim_config
                        else None
                    )
                    or vllm_config.model_config.model
                )
            except (KeyError, FileNotFoundError):
                card = None
            windowed = card is not None and card.sliding_window is not None
        if windowed:
            raise NotImplementedError(
                "a KV connector with sliding-window attention is not implemented. A "
                "windowed request drops blocks behind its window, so its published "
                "prefix is not a prefix any consumer can use, and upstream's "
                "connectors do not model this pairing either."
            )

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

        # R17.1. The store is shared by name, and a block hash is over token ids and
        # extra keys only -- it says nothing about which model computed the KV. Two
        # engines pointed at one store therefore matched each other's blocks even
        # with different models, different dtypes, or different tensor-parallel
        # shardings, and the consumer "hit" on KV of a different shape entirely.
        # Namespacing the key rather than the hash keeps C3's hash values exactly
        # upstream's while making the store's identity check the real one.
        model_config = vllm_config.model_config
        self._namespace = "|".join(
            str(part)
            for part in (
                model_config.model,
                model_config.resolved_dtype,
                vllm_config.cache_config.resolved_cache_dtype,
                vllm_config.cache_config.block_size,
                vllm_config.parallel_config.tensor_parallel_size,
                vllm_config.parallel_config.pipeline_parallel_size,
            )
        ).encode()

        # R17.1. `kv_producer` publishes and never pulls; `kv_consumer` pulls and
        # never publishes; `kv_both` (and an unset role) does both. Validated but
        # unread before, so a disaggregation experiment ran both halves as `kv_both`
        # whatever it configured -- and the prefill node reported hits on KV it had
        # just written itself.
        role_name = (transfer.kv_role if transfer else None) or "kv_both"
        self.may_load = role_name in ("kv_consumer", "kv_both")
        self.may_save = role_name in ("kv_producer", "kv_both")

        #: Bytes one block occupies, for costing a transfer. Resolved late, because
        #: the KV layout is not known until the memory model has run.
        self.block_bytes = 0

        #: Pending work for the next `build_connector_meta`.
        self._pending_loads: dict[str, tuple[list[int], int]] = {}
        self._pending_saves: dict[str, tuple[list[int], int]] = {}
        #: R17.2. Cumulative modeled transfer time, each side counted separately.
        #: No Prometheus metric reads these -- the comment used to say there was
        #: one -- but the conformance tests assert on both, which is what keeps the
        #: producer and consumer halves of a disaggregated run honest about who
        #: paid for what.
        self.load_seconds = 0.0
        self.save_seconds = 0.0
        #: Modeled write time not yet charged to the clock. Banked here because the
        #: write is decided on the scheduler side and paid for on the worker side.
        self._unpaid_save_seconds = 0.0

    def _key(self, block_hash: Any) -> bytes:
        """The store key for a block hash, namespaced by the model that computed it."""
        return self._namespace + b"|" + bytes(block_hash)

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
        if not self.may_load or not request.block_hashes or self.block_bytes == 0:
            return 0, False

        local_blocks = num_computed_tokens // self.block_size
        remaining = list(request.block_hashes[local_blocks:])
        if not remaining:
            return 0, False

        matched_blocks = self.store.longest_prefix([self._key(h) for h in remaining])
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
        # From where the local prefix cache left off, not from block 0. `blocks`
        # is the request's *whole* block table, so slicing from the front named the
        # locally-cached blocks -- already full of the right KV -- and left the
        # blocks the store's data was actually meant for unwritten. The request then
        # read uninitialised KV for the externally-matched span.
        local_blocks = request.num_computed_tokens // self.block_size
        num_blocks = num_external_tokens // self.block_size
        self._pending_loads[request.request_id] = (
            list(block_ids[0][local_blocks : local_blocks + num_blocks]),
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
        if not self.may_save or not request.block_hashes or self.block_bytes == 0:
            return False, None

        # Only blocks whose KV was actually computed. `block_hashes` covers the whole
        # prompt from the moment the request is built, so publishing it wholesale
        # meant an aborted request -- or one that never got a single step -- filled
        # the store with hashes for KV that does not exist. A consumer then "hit" on
        # them and read uninitialised blocks, and the reported store hit rate counted
        # a transfer that transferred nothing.
        computed_blocks = min(
            request.num_computed_tokens // self.block_size,
            len(request.block_hashes),
        )
        if computed_blocks <= 0:
            return False, None

        hashes = [self._key(h) for h in request.block_hashes[:computed_blocks]]
        num_bytes = len(hashes) * self.block_bytes
        seconds = self.store.write(hashes, num_bytes)
        self.save_seconds += seconds
        # Banked, not discarded: the producer's clock has to pay for its own writes
        # or a disaggregated pair looks like it publishes for free, which is the one
        # number the experiment is trying to weigh against recomputing.
        self._unpaid_save_seconds += seconds
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
        """Charge the step for the writes decided since the last one. R17.2.

        The write itself is issued from `request_finished`, on the scheduler side,
        where there is no clock (R19.1). This is the worker-side moment the step's
        modeled duration is assembled, so the banked time is paid here -- which is
        also where a real connector would block on its outstanding pushes.
        """
        seconds, self._unpaid_save_seconds = self._unpaid_save_seconds, 0.0
        return seconds

    def __repr__(self) -> str:
        return (
            f"SimSharedStoreConnector(store={self.store.name!r}, "
            f"role={self.role.name}, hit_rate={self.store.hit_rate:.2f})"
        )
