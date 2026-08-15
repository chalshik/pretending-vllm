"""JSONL event trace. R19.3.

Upstream: (none -- simulator)
Tier: D

The primary artifact for both understanding and conformance. One record per engine
step, plus one per request lifecycle transition, written as newline-delimited JSON.

The trace is what makes inference transparent (D9): every scheduling decision, block
allocation, prefix cache hit, and preemption is in it. It is also what the conformance
suite compares (C1--C4), so the encoding is deterministic -- given the same seed and
config, two runs produce byte-identical traces. That means no wall-clock timestamps,
no set iteration order, and no dict ordering that depends on insertion history the
caller does not control.

Format::

    {"v":1,"seq":0,"type":"header","schema_version":1,"upstream_version":"0.27.1",...}
    {"v":1,"seq":1,"type":"request","t":1767225600.0,"request_id":"r0","event":"arrived",...}
    {"v":1,"seq":2,"type":"step","t":1767225600.0,"step":0,"scheduled":{"r0":128},...}

Every record carries ``seq``, a gap-free counter. A conformance diff that sees a gap
knows records were dropped rather than that behavior changed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO

import msgspec

from pvllm.tracing import (
    TRACE_SCHEMA_VERSION as TRACE_SCHEMA_VERSION,
)
from pvllm.tracing import (
    TraceSink as TraceSink,
)
from pvllm.tracing import (
    read_header as read_header,
)
from pvllm.tracing import (
    read_trace as read_trace,
)

_encoder = msgspec.json.Encoder()


class NullTraceWriter:
    """A sink that discards everything.

    Used when tracing is off. Kept as a real object rather than a ``None`` check at
    every call site so that turning tracing off cannot change control flow -- the
    engine emits unconditionally and pays one method call.
    """

    enabled = False

    def emit(self, event_type: str, t: float | None = None, **fields: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> NullTraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class TraceWriter:
    """Writes a JSONL event trace.

    Args:
        path: Destination file. Parent directories are created.
        stream: Alternative to ``path`` for tests -- any binary file-like object.
        seed: The run seed, recorded in the header so a trace is self-describing.
        clock_mode: One of ``virtual``, ``real``, ``scaled``. Recorded so a consumer
            knows whether the durations in the trace were modeled or slept.
        upstream_version: The pinned vLLM version. Recorded in every trace so a
            golden trace can never be silently compared across pins (see UPSTREAM.md).
        config: Optional resolved-config summary for the header.
        flush_every: Flush after this many records. ``1`` makes the trace readable
            while a run is in flight, at the cost of throughput.
    """

    enabled = True

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        stream: BinaryIO | None = None,
        seed: int,
        clock_mode: str,
        upstream_version: str,
        config: Mapping[str, Any] | None = None,
        flush_every: int = 64,
    ) -> None:
        if (path is None) == (stream is None):
            raise ValueError("pass exactly one of path or stream")

        self._owns_stream = stream is None
        if stream is None:
            assert path is not None
            resolved = Path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._stream: BinaryIO = resolved.open("wb")
            self.path: Path | None = resolved
        else:
            self._stream = stream
            self.path = None

        self._seq = 0
        self._flush_every = max(1, flush_every)
        self._since_flush = 0
        self._closed = False

        self.emit(
            "header",
            t=None,
            schema_version=TRACE_SCHEMA_VERSION,
            upstream_version=upstream_version,
            seed=seed,
            clock_mode=clock_mode,
            config=dict(config) if config is not None else None,
        )

    def emit(self, event_type: str, t: float | None = None, **fields: Any) -> None:
        """Append one record.

        ``t`` is modeled time from the engine's clock, never wall-clock time. It is
        ``None`` only for the header, which is emitted before the clock starts.
        """
        if self._closed:
            raise RuntimeError("cannot emit to a closed trace")

        record: dict[str, Any] = {
            "v": TRACE_SCHEMA_VERSION,
            "seq": self._seq,
            "type": event_type,
        }
        if t is not None:
            record["t"] = t
        record.update(fields)

        self._stream.write(_encoder.encode(record))
        self._stream.write(b"\n")

        self._seq += 1
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self._stream.flush()
            self._since_flush = 0

    @property
    def num_records(self) -> int:
        return self._seq

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.flush()
        finally:
            if self._owns_stream:
                self._stream.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
