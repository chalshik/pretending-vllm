"""KV transfer configuration.

Upstream: vllm/config/kv_transfer.py
Tier: C

Present so the config surface matches. The connector interface and disaggregation
(R17) land in M4.
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

    def __post_init__(self) -> None:
        raise NotImplementedError(
            "KV transfer and disaggregation (requirement R17) land in M4"
        )
