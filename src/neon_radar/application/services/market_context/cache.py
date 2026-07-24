"""Context Cache with TTL for minimizing API calls."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from neon_radar.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

@dataclass
class CacheEntry(Generic[T]):
    data: T
    expires_at: float


class ContextCache:
    """In-memory TTL cache for Market Context objects, with optional on-disk JSON caching for historical series."""

    def __init__(self, directory: Path | None = None) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._dir = directory
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        """Get item from memory cache if it exists and hasn't expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None

        if time.time() > entry.expires_at:
            del self._cache[key]
            return None

        return entry.data

    def set(self, key: str, data: Any, ttl_seconds: float) -> None:
        """Store item in memory cache with a TTL."""
        self._cache[key] = CacheEntry(
            data=data,
            expires_at=time.time() + ttl_seconds
        )

    def get_json(self, key: str, deserializer: Any) -> Any | None:
        """Get item from JSON disk cache."""
        if not self._dir:
            return None

        path = self._dir / f"{key}.json"
        if not path.is_file():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))

            if payload.get("schema_version") != "v1":
                return None

            if time.time() > payload.get("expires_at", 0):
                return None
            return deserializer(payload["data"])
        except Exception as exc:
            logger.warning("Failed to read JSON cache %s: %s", path, exc)
            return None

    def set_json(self, key: str, data: Any, ttl_seconds: float) -> None:
        """Store item in JSON disk cache with TTL."""
        if not self._dir:
            return

        path = self._dir / f"{key}.json"
        try:
            payload = {
                "schema_version": "v1",
                "expires_at": time.time() + ttl_seconds,
                "data": asdict(data)
            }
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            logger.warning("Failed to write JSON cache %s: %s", path, exc)
