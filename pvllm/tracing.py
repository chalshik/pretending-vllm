"""The trace sink interface.

Upstream: (none -- pvllm addition)
Tier: B

R19.3's event trace is a pretending-vllm addition; upstream's scheduler emits no such
thing. But the *interface* belongs above the simulation boundary, because the control
plane is what emits events -- only the writing of them is Tier D.

Defining the protocol here rather than in `pvllm/sim/trace.py` is what lets
`v1/core` and `v1/engine` reference it without importing the simulator (B1). The
boundary lint in `tests/unit/test_purity.py` catches the alternative.

It is a `Protocol`, so `pvllm.sim.trace.TraceWriter` satisfies it structurally without
importing anything from here either.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import msgspec

#: Bump when a field's meaning changes or a field is removed. Adding a field is
#: backward compatible; readers must ignore unknown keys.
TRACE_SCHEMA_VERSION = 1

_decoder = msgspec.json.Decoder()


@runtime_checkable
class TraceSink(Protocol):
    """Somewhere engine events can be recorded."""

    enabled: bool

    def emit(self, event_type: str, t: float | None = None, **fields: Any) -> None:
        """Append one event.

        `t` is modeled time from the engine core's clock, never wall-clock time
        (R19.1). Callers below the engine core leave it unset and let the core stamp
        the record, because they have no clock to read.
        """
        ...

    def close(self) -> None: ...


# --- reading ---------------------------------------------------------------
#
# The reader lives here rather than beside the writer because reading a trace is
# not a simulator activity: the viewer, the conformance suite, and any tool a user
# writes all consume traces from above the boundary. Only the writer is Tier D.


def read_trace(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL trace.

    Validates `seq` continuity, because a gap means records were lost -- which would
    otherwise read as a behavioral difference in a conformance diff rather than as
    a broken file.
    """
    expected = 0
    with Path(path).open("rb") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = _decoder.decode(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            seq = record.get("seq")
            if seq != expected:
                raise ValueError(
                    f"{path}:{lineno}: trace discontinuity -- expected seq {expected}, "
                    f"got {seq}. Records were dropped or the file was concatenated."
                )
            expected += 1
            yield record


def read_header(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read just the header record of a trace."""
    for record in read_trace(path):
        if record.get("type") != "header":
            raise ValueError(
                f"{path}: first record is {record.get('type')!r}, not a header"
            )
        return record
    raise ValueError(f"{path}: empty trace")
