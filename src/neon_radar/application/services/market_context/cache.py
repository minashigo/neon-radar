"""Context Cache with TTL for minimizing API calls."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")

@dataclass
class CacheEntry(Generic[T]):
    data: T
    expires_at: float


class ContextCache:
    """In-memory TTL cache for Market Context objects."""

    def __init__(self) -> None:
        self._cache: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        """Get item if it exists and hasn't expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None

        if time.time() > entry.expires_at:
            del self._cache[key]
            return None

        return entry.data

    def set(self, key: str, data: Any, ttl_seconds: float) -> None:
        """Store item in cache with a TTL."""
        self._cache[key] = CacheEntry(
            data=data,
            expires_at=time.time() + ttl_seconds
        )
