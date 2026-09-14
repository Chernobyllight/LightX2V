"""Mapping adapter for zero-copy CPU offload weight views.

Private values are removed as loaders consume them. Shared values remain
available to multiple block instances, while consumption is recorded for
post-load validation.

This class deliberately does not close its owner.  Adopted tensor views normally
outlive this short-lived mapping, so the model must retain ``owner`` and close it
only after all asynchronous transfers and tensor users have finished.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from typing import Generic, TypeVar

_ValueT = TypeVar("_ValueT")


class SharedWeightViewMap(Mapping[str, _ValueT], Generic[_ValueT]):
    """Combine consumable private values with persistent shared tensor views."""

    def __init__(
        self,
        private: Mapping[str, _ValueT] | None = None,
        shared: Mapping[str, _ValueT] | None = None,
        *,
        owner: object | None = None,
    ) -> None:
        self._private = dict(private or {})
        self._shared = dict(shared or {})
        overlap = self._private.keys() & self._shared.keys()
        if overlap:
            raise ValueError(f"private and shared weights overlap: {sorted(overlap)!r}")
        if not self._shared:
            raise ValueError("shared weight views must be non-empty")
        if owner is None:
            raise ValueError("owner is required for shared weight views")
        self._owner = owner
        self._consumed_shared: set[str] = set()

    @property
    def owner(self) -> object:
        return self._owner

    @property
    def consumed_shared_keys(self) -> frozenset[str]:
        return frozenset(self._consumed_shared)

    def is_shared(self, key: str) -> bool:
        return key in self._shared

    def take(self, key: str) -> _ValueT:
        """Consume a private value or record use of a persistent shared view."""

        if key in self._shared:
            self._consumed_shared.add(key)
            return self._shared[key]
        return self._private.pop(key)

    def __getitem__(self, key: str) -> _ValueT:
        if key in self._private:
            return self._private[key]
        return self._shared[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._private
        yield from self._shared

    def __len__(self) -> int:
        return len(self._private) + len(self._shared)


def consume_weight(
    weight_mapping: MutableMapping[str, _ValueT] | SharedWeightViewMap[_ValueT],
    key: str,
) -> tuple[_ValueT, bool]:
    """Return one weight and whether it is backed by shared pinned memory."""

    if isinstance(weight_mapping, SharedWeightViewMap):
        return weight_mapping.take(key), weight_mapping.is_shared(key)
    return weight_mapping.pop(key), False
