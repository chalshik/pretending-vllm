"""Log record formatters.

Upstream: vllm/logging_utils/formatter.py
Tier: B

R12.3 requires that the periodic stats log line match upstream's format closely
enough that log parsers written against real vLLM keep working. That constrains the
formatter, not just the message strings, so this is a real port rather than a stub.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from pvllm import envs


def _shrink_path(relpath: Path) -> str:
    """Shorten a source path for display.

    Drops a leading ``pvllm`` component, keeps the first one (or two, under ``v1``)
    and last two components, and collapses the middle to ``...``:

    ``pvllm/v1/core/sched/scheduler.py`` -> ``v1/core/sched/scheduler.py``
    ``pvllm/model_executor/layers/quantization/utils/fp8.py``
        -> ``model_executor/.../utils/fp8.py``
    """
    parts = list(relpath.parts)
    new_parts: list[str] = []
    if parts and parts[0] == "pvllm":
        parts = parts[1:]
    if parts and parts[0] == "v1":
        new_parts += parts[:2]
        parts = parts[2:]
    elif parts:
        new_parts += parts[:1]
        parts = parts[1:]
    if len(parts) > 2:
        new_parts += ["...", *parts[-2:]]
    else:
        new_parts += parts
    return "/".join(new_parts)


class NewLineFormatter(logging.Formatter):
    """Adds the logging prefix to newlines so multi-line messages stay aligned."""

    def __init__(self, fmt: str, datefmt: str | None = None, style: str = "%") -> None:
        super().__init__(fmt, datefmt, style)  # type: ignore[arg-type]
        self.use_relpath = envs.PVLLM_LOGGING_LEVEL == "DEBUG"
        self.root_dir = Path(__file__).resolve().parent.parent.parent

    def format(self, record: logging.LogRecord) -> str:
        if self.use_relpath:
            abs_path = getattr(record, "pathname", None)
            relpath = Path(record.filename)
            if abs_path:
                try:
                    relpath = Path(abs_path).resolve().relative_to(self.root_dir)
                except ValueError:
                    relpath = Path(record.filename)
            record.fileinfo = _shrink_path(relpath)
        else:
            record.fileinfo = record.filename

        msg = super().format(record)
        if record.message != "":
            head = msg.split(record.message)[0]
            msg = msg.replace("\n", "\r\n" + head)
        return msg


class ColoredFormatter(NewLineFormatter):
    """Adds ANSI colour codes to level names for terminal output."""

    GREY = "\x1b[38;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    _LEVEL_COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: GREY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelno, self.GREY)
        original = record.levelname
        record.levelname = f"{color}{original}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original
