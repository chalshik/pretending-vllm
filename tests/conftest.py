"""Shared test configuration."""

from __future__ import annotations

import os

import pytest

# R21.1: the invariant assertions (block accounting, budget bounds, slot-mapping
# validation) run for the whole suite. They are the cheapest place to catch a KV
# manager bug, and they only pay off if they are always on in tests.
os.environ.setdefault("PVLLM_DEBUG_INVARIANTS", "1")

# Keep log output deterministic and uncoloured regardless of where the suite runs.
os.environ.setdefault("PVLLM_LOGGING_COLOR", "0")


@pytest.fixture(autouse=True)
def _reset_sim_globals():
    """Isolate the process-global device card between tests.

    `SimPlatform`'s device-introspection classmethods read a module-level card, which
    stands in for the real device an upstream platform would probe. Leaking it
    between tests would make results depend on execution order.
    """
    from pvllm.sim.hardware_db import reset_active_device_card

    reset_active_device_card()
    yield
    reset_active_device_card()
