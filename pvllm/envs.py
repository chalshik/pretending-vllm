"""Environment variable surface, mirroring vLLM's ``VLLM_*`` surface as ``PVLLM_*``.

Upstream: vllm/envs.py
Tier: C

Variables are read lazily through the module-level ``__getattr__``, exactly as
upstream does, so a test can monkeypatch ``os.environ`` and see the change without
reimporting.

Names are ``PVLLM_``-prefixed rather than ``VLLM_``-prefixed on purpose. D1 requires
that pretending-vllm and a real vLLM can be installed side by side for diffing and
conformance work; sharing an env var surface would make one silently reconfigure the
other. The mapping is one-to-one for every variable that exists upstream.

Note this module is exempt from the clock/RNG purity rule only in the sense that it
reads ``os.environ`` -- it does not read a clock or draw randomness. See
``tests/unit/test_purity.py``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Declarations for editors and type checkers. At runtime these are served by
    # the module-level __getattr__ below, so they are never actually assigned.
    PVLLM_CONFIGURE_LOGGING: int = 1
    PVLLM_LOGGING_LEVEL: str = "INFO"
    PVLLM_LOGGING_PREFIX: str = ""
    PVLLM_LOGGING_COLOR: str = "1"
    NO_COLOR: bool = False
    PVLLM_DEBUG_INVARIANTS: bool = False
    PVLLM_TRACE_PATH: str | None = None
    PVLLM_PLUGINS: list[str] | None = None
    PVLLM_USE_V2_MODEL_RUNNER: bool | None = None


def _maybe_convert_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def _get_plugins() -> list[str] | None:
    raw = os.getenv("PVLLM_PLUGINS")
    if raw is None:
        return None
    return [name for name in raw.split(",") if name]


environment_variables: dict[str, Callable[[], Any]] = {
    # --- logging -----------------------------------------------------------
    "PVLLM_CONFIGURE_LOGGING": lambda: int(os.getenv("PVLLM_CONFIGURE_LOGGING", "1")),
    "PVLLM_LOGGING_LEVEL": lambda: os.getenv("PVLLM_LOGGING_LEVEL", "INFO").upper(),
    "PVLLM_LOGGING_PREFIX": lambda: os.getenv("PVLLM_LOGGING_PREFIX", ""),
    "PVLLM_LOGGING_COLOR": lambda: os.getenv("PVLLM_LOGGING_COLOR", "1"),
    # Honoured by convention across CLI tooling; not a vLLM invention.
    "NO_COLOR": lambda: bool(os.getenv("NO_COLOR")),
    # --- simulator ---------------------------------------------------------
    # R21.1: turns on the invariant assertions (block accounting, budget bounds,
    # slot-mapping validation). On for the whole test suite, off by default in
    # production so the assertions cost nothing.
    "PVLLM_DEBUG_INVARIANTS": lambda: bool(
        int(os.getenv("PVLLM_DEBUG_INVARIANTS", "0"))
    ),
    # R19.3: where the JSONL event trace is written. Unset means no trace.
    "PVLLM_TRACE_PATH": lambda: os.getenv("PVLLM_TRACE_PATH") or None,
    # --- plugins -----------------------------------------------------------
    # Mirrors VLLM_PLUGINS: restricts which entry-point plugins are loaded.
    "PVLLM_PLUGINS": _get_plugins,
    # F1/D6: upstream resolves this to True by default for dense generate models,
    # and pretending-vllm only implements the V2 shape. Kept as a tri-state so the
    # config surface matches upstream; setting it to 0 raises rather than silently
    # falling back, since there is no V1 runner to fall back to.
    "PVLLM_USE_V2_MODEL_RUNNER": lambda: _maybe_convert_bool(
        os.getenv("PVLLM_USE_V2_MODEL_RUNNER", None)
    ),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    """Whether an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
