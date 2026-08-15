"""The KV connector interface. R17.1.

Upstream: vllm/distributed/kv_transfer/kv_connector/v1/base.py
Tier: B

A KV connector moves KV cache between an engine and somewhere else -- another engine,
a shared store, a disaggregated prefill node. It has two halves that live in different
places and never call each other directly:

* **scheduler side** decides whether a request's prefix is available externally
  (`get_num_new_matched_tokens`), reacts to the blocks allocated for it
  (`update_state_after_alloc`), and packs per-step instructions for the worker
  (`build_connector_meta`);
* **worker side** performs the transfer (`start_load_kv`, `wait_for_save`).

The split is the whole design. The scheduler must decide what to load *before* the
step runs, without blocking on a network; the worker must do the moving without
knowing why. Reproducing the split rather than collapsing it is what makes the
scheduling behaviour around disaggregation real -- a request whose KV is still
arriving sits in `WAITING_FOR_REMOTE_KVS` and is not admitted, and that is a state a
product's latency depends on.

What a simulator can carry faithfully here is the *timing and the state machine*, not
the bytes. `SimSharedStoreConnector` models a store with a bandwidth and a latency;
nothing is transferred, and a load's duration is what the store's parameters say it
would take.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pvllm.config import VllmConfig
    from pvllm.v1.core.kv_cache_manager import KVCacheBlocks
    from pvllm.v1.core.sched.output import SchedulerOutput
    from pvllm.v1.request import Request


class KVConnectorRole(enum.Enum):
    """Which half of the connector this instance is."""

    SCHEDULER = 0
    WORKER = 1


@dataclass
class KVConnectorMetadata:
    """One step's instructions, from the scheduler half to the worker half.

    Crosses the same boundary `SchedulerOutput` does, and for the same reason: the
    worker performs transfers without deciding them.
    """

    #: request_id -> (block ids, number of tokens) to pull in before this step.
    loads: dict[str, tuple[list[int], int]] = field(default_factory=dict)
    #: request_id -> (block ids, number of tokens) to push out after it.
    saves: dict[str, tuple[list[int], int]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.loads or self.saves)


class KVConnectorBase(ABC):
    """Both halves of a connector. R17.1."""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole) -> None:
        self.vllm_config = vllm_config
        self.role = role
        self.block_size = vllm_config.cache_config.block_size

    # --- scheduler side ------------------------------------------------------

    @abstractmethod
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """Tokens available externally beyond `num_computed_tokens`.

        Returns `(num_tokens, load_is_async)`. An async load means the request waits
        in `WAITING_FOR_REMOTE_KVS` while the transfer runs, which is the state that
        makes disaggregated prefill's latency visible.

        Upstream allows `None` for "ask me again later"; this does not, because a
        simulated store answers instantly and a tri-state nothing ever returns is
        a branch that would never be exercised.
        """

    @abstractmethod
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        """React to the blocks the KV manager allocated for an external load.

        Keyed on `num_external_tokens`, not on whether `blocks` is empty -- upstream
        documents the same trap: blocks can be non-empty with nothing to load.
        """

    @abstractmethod
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Pack this step's transfers, and reset the pending state."""

    def request_finished(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called once as a request finishes, before its blocks are freed.

        Returns `(blocks_are_still_in_use, kv_transfer_params)`. True means the
        connector is still reading them and the scheduler must not free them yet --
        the mechanism behind a prefill node holding KV until the decode node has
        pulled it.
        """
        return False, None

    # --- worker side ---------------------------------------------------------

    @abstractmethod
    def start_load_kv(self, metadata: KVConnectorMetadata) -> float:
        """Pull this step's KV in. Returns the modeled duration in seconds."""

    @abstractmethod
    def wait_for_save(self, metadata: KVConnectorMetadata) -> float:
        """Push this step's KV out. Returns the modeled duration in seconds."""

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """`(finished_sending, finished_receiving)` since the last call."""
        return set(), set()
