"""Parallelism configuration.

Upstream: vllm/config/parallel.py
Tier: C

Parallelism is simulated in process (NG5). What must be right is that per-device
memory and step time change correctly with the degree (R13.1), not that data moves.
Sharding lands in M4; the fields exist now so the config surface matches and so
`worker_cls` -- the hinge `SimPlatform.check_and_update_config` fills in -- is here
from the start.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParallelConfig:
    """Configuration for distributed execution."""

    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    data_parallel_rank: int = 0
    enable_expert_parallel: bool = False
    distributed_executor_backend: str | None = None
    #: Filled in from `"auto"` by the platform (B2). This is the seam.
    worker_cls: str = "auto"
    sd_worker_cls: str = "auto"

    world_size: int = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "pipeline_parallel_size",
            "tensor_parallel_size",
            "data_parallel_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )

        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size

        # Sharding changes per-device memory and the cost model's inputs, so
        # pretending it works would silently produce wrong capacity answers -- the
        # exact failure mode this project exists to avoid.
        if self.tensor_parallel_size > 1:
            raise NotImplementedError(
                "tensor parallelism (requirement R13.1) lands in M4. Until then a "
                "tensor_parallel_size above 1 would report single-device memory and "
                "step time, which is worse than refusing."
            )
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "pipeline parallelism (requirement R13.2) lands in M4"
            )
        if self.data_parallel_size > 1:
            raise NotImplementedError(
                "data parallel replicas (requirement R13.3) land in M4"
            )
        if self.enable_expert_parallel:
            raise NotImplementedError("expert parallelism lands in M4")
