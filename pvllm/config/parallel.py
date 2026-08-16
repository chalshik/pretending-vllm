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

from pvllm.logger import init_logger

logger = init_logger(__name__)


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
    #: R13.4. Devices the experts are spread across. Upstream derives it the same
    #: way -- `ep_size = tp_size` *after* flattening tp across dp
    #: (`FusedMoEParallelConfig.make`), which is `data_parallel_size *
    #: tensor_parallel_size`. That is the surprising part of turning EP on: the data
    #: parallel replicas stop being independent copies and jointly shard one set of
    #: experts, which is also why they then have to step in lockstep.
    expert_parallel_size: int = field(init=False)

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
        # R13.3. Data parallel replicas are independent whole engines behind a
        # router, not a sharded one: each holds its own weights, its own device and
        # its own KV pool. `world_size` is per replica, which is why it does not
        # include `data_parallel_size`.
        # R13.4. Expert parallelism replaces tensor parallelism *for the experts*:
        # each device owns whole experts rather than a slice of every one. So the
        # group spans both the tensor and data dimensions, and it is a no-op on a
        # single device -- upstream's `use_ep` requires `dp * tp > 1` for the same
        # reason.
        self.expert_parallel_size = (
            self.data_parallel_size * self.tensor_parallel_size
            if self.enable_expert_parallel
            else 1
        )
        if self.enable_expert_parallel and self.expert_parallel_size == 1:
            # Accepted and reported, not refused. Upstream's `use_ep` requires
            # `dp * tp > 1` and quietly stays False otherwise, so a script that runs
            # on real vLLM must run here (C7). Logged because "the flag did nothing"
            # is worth knowing and upstream does not say it.
            logger.info(
                "enable_expert_parallel has no effect at tensor_parallel_size=1 and "
                "data_parallel_size=1: expert parallelism spreads experts across "
                "devices and there is only one. Upstream ignores it the same way."
            )
