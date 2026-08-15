"""Import helpers.

Upstream: vllm/utils/import_utils.py
Tier: B
"""

from __future__ import annotations

import importlib
from typing import Any


def resolve_obj_by_qualname(qualname: str) -> Any:
    """Resolve an object by its fully-qualified name."""
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
