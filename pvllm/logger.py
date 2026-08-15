"""Logging setup.

Upstream: vllm/logger.py
Tier: B

The record format matches upstream's so that log parsers written against real vLLM
keep working (R12.3)::

    INFO 08-15 13:47:02 [scheduler.py:412] Engine step 17, 3 running, 1 waiting

``init_logger`` patches ``*_once`` methods onto the returned logger instead of using
a Logger subclass, mirroring upstream's approach.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Hashable
from functools import lru_cache
from types import MethodType
from typing import Any, cast

from pvllm import envs
from pvllm.logging_utils import ColoredFormatter, NewLineFormatter

_FORMAT = (
    f"{envs.PVLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
    "[%(fileinfo)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%m-%d %H:%M:%S"

_root_configured = False


def _use_color() -> bool:
    if envs.NO_COLOR or envs.PVLLM_LOGGING_COLOR == "0":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


@lru_cache
def _print_once(logger_name: str, level: int, msg: str, args: tuple[Any, ...]) -> None:
    logging.getLogger(logger_name).log(level, msg, *args)


def _debug_once(self: logging.Logger, msg: str, *args: Hashable) -> None:
    """As `debug`, but subsequent calls with the same message are dropped."""
    _print_once(self.name, logging.DEBUG, msg, args)


def _info_once(self: logging.Logger, msg: str, *args: Hashable) -> None:
    """As `info`, but subsequent calls with the same message are dropped."""
    _print_once(self.name, logging.INFO, msg, args)


def _warning_once(self: logging.Logger, msg: str, *args: Hashable) -> None:
    """As `warning`, but subsequent calls with the same message are dropped."""
    _print_once(self.name, logging.WARNING, msg, args)


_METHODS_TO_PATCH = {
    "debug_once": _debug_once,
    "info_once": _info_once,
    "warning_once": _warning_once,
}


class _PvllmLogger(logging.Logger):
    """Type information only. The methods are patched onto instances."""

    def debug_once(self, msg: str, *args: Hashable) -> None: ...
    def info_once(self, msg: str, *args: Hashable) -> None: ...
    def warning_once(self, msg: str, *args: Hashable) -> None: ...


def _configure_root_logger() -> None:
    global _root_configured
    if _root_configured or not envs.PVLLM_CONFIGURE_LOGGING:
        _root_configured = True
        return

    formatter_cls = ColoredFormatter if _use_color() else NewLineFormatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter_cls(_FORMAT, datefmt=_DATE_FORMAT))
    handler.setLevel(envs.PVLLM_LOGGING_LEVEL)

    root = logging.getLogger("pvllm")
    root.setLevel(envs.PVLLM_LOGGING_LEVEL)
    root.addHandler(handler)
    root.propagate = False
    _root_configured = True


def init_logger(name: str) -> _PvllmLogger:
    """Retrieve a logger, ensuring the pvllm root logger is configured first."""
    _configure_root_logger()
    logger = logging.getLogger(name)
    for method_name, method in _METHODS_TO_PATCH.items():
        setattr(logger, method_name, MethodType(method, logger))
    return cast(_PvllmLogger, logger)
