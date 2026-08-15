"""A LoRA adapter request. R16.1.

Upstream: vllm/lora/request.py
Tier: C

A msgspec Struct, as upstream, because it crosses the engine-core boundary inside
`EngineCoreRequest` and has to serialize.

The field that matters most here is `lora_int_id`, and it matters for a reason that
has nothing to do with loading weights: it joins the prefix cache extra keys. Two
requests with byte-identical prompts and different adapters produce different KV, so
they must not share blocks. A simulator that got that wrong would report cache hit
rates far above what a real deployment sees -- and cache hit rate is one of the four
things the fidelity contract calls exact (C3).
"""

from __future__ import annotations

import msgspec


class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,
    array_like=True,
):
    """One adapter, as a request refers to it.

    `lora_int_id` must be globally unique per adapter. Upstream documents that it
    does not enforce this; neither does this, for the same reason -- the id is the
    caller's namespace, and two adapters sharing one would be a caller error that
    shows up as cache sharing between them.
    """

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = None
    #: R16.1. Adapter rank. Upstream reads this from the adapter's config on disk;
    #: there is no disk here, so it is declared, and it is what the memory model
    #: charges for.
    rank: int = 16

    def __post_init__(self) -> None:
        if self.lora_int_id < 1:
            raise ValueError(f"lora_int_id must be positive, got {self.lora_int_id}")
        if self.rank < 1:
            raise ValueError(f"rank must be positive, got {self.rank}")

    @property
    def adapter_id(self) -> int:
        return self.lora_int_id

    @property
    def name(self) -> str:
        return self.lora_name

    def __hash__(self) -> int:
        # By id alone, matching upstream: the id *is* the adapter's identity, and
        # hashing the path too would make two references to one adapter look like
        # two adapters to any set that holds them -- including the scheduler's
        # active-slot set, which would then admit more adapters than there are slots.
        return hash(self.lora_int_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LoRARequest) and self.lora_int_id == other.lora_int_id
