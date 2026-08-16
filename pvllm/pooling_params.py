"""Parameters for pooling requests. R2.2.

Upstream: vllm/pooling_params.py
Tier: C

A pooling request runs the prompt through the model and returns a vector instead of
generating tokens. It prefills and stops: there is no decode phase, no sampling, and
no `max_tokens`. That difference is the whole of it from the engine's point of view,
and it is why a pooling request is cheap in steps and expensive in nothing else.

**What is simulated and what is not.** The scheduling, the KV accounting, the prefix
caching, and the step count are real -- an embedding workload's capacity behaviour is
reproduced faithfully. The *vector* is synthetic: derived deterministically from the
prompt tokens, normalized, and meaningless. Two identical prompts give identical
vectors and two different prompts give different ones, which is what a product's
plumbing and caching need, but cosine similarity between them carries no semantic
information whatsoever. Do not build a retrieval evaluation on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PoolingTask = Literal["embed", "encode", "classify", "score"]

#: Tasks the simulator produces a vector for. `classify` and `score` need a head this
#: engine has no counterpart for -- their output is a label distribution, not an
#: embedding, and inventing one would be inventing a model.
SUPPORTED_TASKS = ("embed", "encode")


@dataclass
class PoolingParams:
    """API parameters for a pooling request."""

    #: `None` uses the model card's hidden size, which is what a real pooler emits.
    dimensions: int | None = None
    use_activation: bool | None = None
    task: PoolingTask | None = None

    def __post_init__(self) -> None:
        if self.task is not None and self.task not in SUPPORTED_TASKS:
            raise NotImplementedError(
                f"pooling task {self.task!r} needs a classification head, which has "
                f"no simulated counterpart: its output is a label distribution over "
                f"labels this engine does not have. "
                f"{list(SUPPORTED_TASKS)} are supported."
            )
        if self.dimensions is not None and self.dimensions < 1:
            raise ValueError(f"dimensions must be at least 1, got {self.dimensions}")

    def clone(self) -> PoolingParams:
        return PoolingParams(
            dimensions=self.dimensions,
            use_activation=self.use_activation,
            task=self.task,
        )


__all__ = ["SUPPORTED_TASKS", "PoolingParams", "PoolingTask"]
