"""LoRA configuration.

Upstream: vllm/config/lora.py
Tier: C

Present so the config surface matches. LoRA itself (R16) lands in M4, where the
adapter id joins the prefix cache extra keys and max_loras affects memory accounting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    max_lora_rank: int = 16
    max_loras: int = 1
    max_cpu_loras: int | None = None
    lora_dtype: str = "auto"

    def __post_init__(self) -> None:
        raise NotImplementedError(
            "LoRA (requirement R16) lands in M4. The adapter id participates in "
            "prefix cache extra keys and max_loras affects memory accounting, so a "
            "stub would give wrong cache-hit and capacity answers."
        )
