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

        # R13.1/R13.2. Sharding changes per-device memory and the cost model's
        # inputs, both of which are modeled -- see `pvllm/sim/memory.py` and
        # `RooflineCostModel`. What is *not* modeled is the throughput gain
        # pipeline parallelism gets from overlapping microbatches, because there
        # are no virtual engines here; see the note in `RooflineCostModel`.
        if self.data_parallel_size > 1:
            raise NotImplementedError(
                "data parallel replicas (requirement R13.3) are independent engines "
                "sharing a router, not a sharded one. Run `data_parallel_size` "
                "separate pretending-vllm instances and put your own load balancer "
                "in front -- that is what the deployment does, and simulating the "
                "router here would tell you about the router rather than the engine."
            )
        if self.enable_expert_parallel:
            raise NotImplementedError(
                "expert parallelism shards MoE experts across devices, which changes "
                "the active-parameter count per device in a way the cost model does "
                "not express (requirement R13.4)"
            )
