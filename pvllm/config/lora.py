"""LoRA configuration. R16.1.

Upstream: vllm/config/lora.py
Tier: C

Two fields here have observable consequences, and they are the reason this is not a
stub.

`max_loras` bounds how many *distinct* adapters may be resident at once, which the
scheduler enforces as an admission constraint: a request for a fifth adapter waits
when four slots are full, even though there is KV capacity and a free sequence slot.
That is a real source of queueing in a multi-tenant deployment and it is invisible
unless it is modeled.

`max_lora_rank` and `max_loras` together decide how much device memory the adapters
occupy, which comes out of the KV pool. Serving eight adapters at rank 64 is not free,
and a capacity answer that ignored it would be optimistic in exactly the direction
that hurts.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Ranks upstream accepts. Not arbitrary: the kernels are specialized per rank, and
#: a value outside this set is rejected rather than rounded, because rounding would
#: silently change the memory the adapter occupies.
SUPPORTED_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    max_lora_rank: int = 16
    max_loras: int = 1
    max_cpu_loras: int | None = None
    lora_dtype: str = "auto"
    #: R16.1. Which projections adapters target. Determines the parameter count per
    #: adapter, and therefore its memory. Upstream reads this per adapter; here it is
    #: a config-wide assumption, and the docstring on `adapter_bytes` says so.
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __post_init__(self) -> None:
        if self.max_lora_rank not in SUPPORTED_LORA_RANKS:
            raise ValueError(
                f"max_lora_rank must be one of {list(SUPPORTED_LORA_RANKS)}, got "
                f"{self.max_lora_rank}"
            )
        if self.max_loras < 1:
            raise ValueError(f"max_loras must be at least 1, got {self.max_loras}")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be at least max_loras "
                f"({self.max_loras}); the CPU cache holds adapters that are not "
                f"currently resident on the device"
            )
