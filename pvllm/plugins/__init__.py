"""Entry-point plugin loading.

Upstream: vllm/plugins/__init__.py
Tier: B

Only the platform group is implemented. The others exist as constants so the surface
matches upstream and so a future milestone adding one is a fill-in rather than a
redesign.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pvllm import envs

logger = logging.getLogger(__name__)

# Loaded in every process when `pvllm.platforms.current_platform` is first read.
PLATFORM_PLUGINS_GROUP = "pvllm.platform_plugins"
DEFAULT_PLUGINS_GROUP = "pvllm.general_plugins"
STAT_LOGGER_PLUGINS_GROUP = "pvllm.stat_logger_plugins"


def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
    """Load plugins registered under the given entry point group."""
    from importlib.metadata import entry_points

    allowed_plugins = envs.PVLLM_PLUGINS

    discovered_plugins = entry_points(group=group)
    if len(discovered_plugins) == 0:
        logger.debug("No plugins for group %s found.", group)
        return {}

    logger.debug("Available plugins for group %s:", group)
    for plugin in discovered_plugins:
        logger.debug("- %s -> %s", plugin.name, plugin.value)

    plugins: dict[str, Callable[[], Any]] = {}
    for plugin in discovered_plugins:
        if allowed_plugins is not None and plugin.name not in allowed_plugins:
            continue
        try:
            plugins[plugin.name] = plugin.load()
        except Exception:
            logger.exception("Failed to load plugin %s", plugin.name)
    return plugins
