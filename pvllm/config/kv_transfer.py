"""KV transfer configuration.

Upstream: vllm/config/kv_transfer.py
Tier: C

R17.1. Selects a connector and the role this instance plays in a disaggregated pair.
`kv_connector_extra_config` carries the simulated store's parameters -- its bandwidth
and latency, which are the numbers that decide whether pulling KV beats recomputing
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KVTransferConfig:
    """Configuration for KV cache transfer between instances."""

    kv_connector: str | None = None
    kv_role: str | None = None
    kv_rank: int | None = None
    kv_parallel_size: int = 1
    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)

    #: Connectors this build provides. Upstream's are real transports (LMCache,
    #: NIXL, Mooncake); asking for one names it rather than substituting the
    #: simulated store, because their bandwidth and failure modes are the whole
    #: question a disaggregation experiment is asking.
    KNOWN_CONNECTORS = ("SimSharedStoreConnector",)

    def __post_init__(self) -> None:
        if self.kv_connector is not None and self.kv_connector not in (
            self.KNOWN_CONNECTORS
        ):
            raise NotImplementedError(
                f"KV connector {self.kv_connector!r} is a real transport and is not "
                f"available here. pretending-vllm provides "
                f"{list(self.KNOWN_CONNECTORS)}, which models an external store with "
                f"a configurable bandwidth and latency -- set those from a "
                f"measurement of yours and the scheduling around it is faithful."
            )
        if self.kv_role is not None and self.kv_role not in (
            "kv_producer",
            "kv_consumer",
            "kv_both",
        ):
            raise ValueError(
                f"unknown kv_role {self.kv_role!r}; expected kv_producer, "
                f"kv_consumer, or kv_both"
            )
