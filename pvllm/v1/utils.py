"""Shared v1 utilities.

Upstream: vllm/v1/utils.py
Tier: A
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Generic, TypeVar, overload

T = TypeVar("T")


class ConstantList(Generic[T], Sequence[T]):
    """A read-only *view* of a list.

    Wraps by reference rather than subclassing `list`, which would copy: the whole
    point is that the view reflects later appends to the underlying list while
    refusing to be the thing that does the appending.

    `Request` exposes `output_token_ids` and `all_token_ids` through this because the
    two must advance together. Appending to one directly desyncs them, and the
    resulting bug -- a token counted in one place but not the other -- surfaces far
    from its cause.
    """

    def __init__(self, x: list[T]) -> None:
        self._x = x

    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            "this is a read-only view of a list; mutate through the owning object so "
            "related state stays consistent"
        )

    append = extend = insert = pop = remove = clear = _immutable
    __setitem__ = __delitem__ = __iadd__ = _immutable

    def index(self, item: T, start: int = 0, stop: int | None = None) -> int:
        return self._x.index(item, start, stop if stop is not None else len(self._x))

    def count(self, item: T) -> int:
        return self._x.count(item)

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, item: slice) -> list[T]: ...

    def __getitem__(self, item: int | slice) -> T | list[T]:
        return self._x[item]

    def __len__(self) -> int:
        return len(self._x)

    def __iter__(self) -> Iterator[T]:
        return iter(self._x)

    def __contains__(self, item: object) -> bool:
        return item in self._x

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ConstantList):
            return self._x == other._x
        return self._x == other

    def __repr__(self) -> str:
        return f"ConstantList({self._x})"
