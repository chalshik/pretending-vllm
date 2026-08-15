"""Weight loading configuration.

Upstream: vllm/config/load.py
Tier: C

No weights are ever read (NG1). What survives is the *timing*: R10.4 requires the
startup timeline be simulated and observable, and load bandwidth is what turns a
parameter count into a load duration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoadConfig:
    """Configuration for loading model weights."""

    load_format: str = "auto"
    download_dir: str | None = None
    ignore_patterns: list[str] | None = None

    def __post_init__(self) -> None:
        if self.load_format not in ("auto", "dummy"):
            raise NotImplementedError(
                f"load_format {self.load_format!r} reads real checkpoint files; "
                f"pretending-vllm materializes fake weights (requirement R10.4), so "
                f"only 'auto' and 'dummy' are meaningful"
            )
