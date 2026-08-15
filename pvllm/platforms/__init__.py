"""Platform resolution.

Upstream: vllm/platforms/__init__.py
Tier: B

`current_platform` is resolved lazily on first attribute access, mirroring upstream.
The laziness is not incidental: an out-of-tree plugin needs `from pvllm.platforms
import Platform` in order to subclass it, so the platform cannot be resolved while
this module is still being imported.

Out-of-tree plugins registered under the `pvllm.platform_plugins` entry-point group
take precedence over builtins, exactly as upstream (F11). `SimPlatform` is the only
builtin, and it always activates -- there is no hardware to detect.
"""

from __future__ import annotations

import traceback
from itertools import chain
from typing import TYPE_CHECKING, Any

from pvllm.logger import init_logger
from pvllm.platforms.interface import CpuArchEnum, Platform, PlatformEnum
from pvllm.plugins import PLATFORM_PLUGINS_GROUP, load_plugins_by_group
from pvllm.utils import resolve_obj_by_qualname

logger = init_logger(__name__)


def sim_platform_plugin() -> str | None:
    from pvllm.platforms.sim import sim_platform_plugin as _plugin

    return _plugin()


builtin_platform_plugins = {
    "sim": sim_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    platform_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)

    activated_plugins = []
    for name, func in chain(builtin_platform_plugins.items(), platform_plugins.items()):
        try:
            assert callable(func)
            if func() is not None:
                activated_plugins.append(name)
        except Exception:
            logger.exception("Platform plugin %s raised while probing", name)

    activated_builtin = list(set(activated_plugins) & set(builtin_platform_plugins))
    activated_oot = list(set(activated_plugins) & set(platform_plugins))

    qualname: str | None
    if len(activated_oot) >= 2:
        raise RuntimeError(
            f"Only one platform plugin can be activated, but got: {activated_oot}"
        )
    if len(activated_oot) == 1:
        qualname = str(platform_plugins[activated_oot[0]]())
        logger.info("Platform plugin %s is activated", activated_oot[0])
    elif len(activated_builtin) >= 2:
        raise RuntimeError(
            f"Only one platform plugin can be activated, but got: {activated_builtin}"
        )
    elif len(activated_builtin) == 1:
        qualname = builtin_platform_plugins[activated_builtin[0]]()
        logger.debug("Automatically detected platform %s.", activated_builtin[0])
    else:
        qualname = "pvllm.platforms.interface.UnspecifiedPlatform"
        logger.debug("No platform detected, running on UnspecifiedPlatform")

    assert qualname is not None
    return qualname


_current_platform: Platform | None = None
_init_trace: str = ""

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str) -> Any:
    if name == "current_platform":
        global _current_platform, _init_trace
        if _current_platform is None:
            qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(qualname)()
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    if name in globals():
        return globals()[name]
    raise AttributeError(f"No attribute named {name!r} exists in {__name__}.")


__all__ = [
    "PLATFORM_PLUGINS_GROUP",
    "CpuArchEnum",
    "Platform",
    "PlatformEnum",
    "_init_trace",
    "current_platform",
    "resolve_current_platform_cls_qualname",
]
