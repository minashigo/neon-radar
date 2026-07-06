"""On-disk cache for :class:`KlineSeries`.

The cache stores each ``(symbol, timeframe)`` pair in a single JSON
file under a configurable directory. This trades fine-grained
invalidation (you can't drop just one candle) for **simplicity** and
**predictability** — a file either exists and is fresh, or it does
not.

Design notes
------------
* Serialisation is deliberately JSON, not pickle. The cache must be
  debuggable with ``cat`` and ``jq``.
* Each candle is stored as a dict. This costs ~3x the bytes of a
  binary format, but the cache is for tens of MB at most — acceptable.
* TTL is checked against ``st_mtime``, not against the latest
  candle's timestamp. This means a refresh on an otherwise quiet
  market will keep the cache valid; a clock-rewind attack on the
  filesystem would not bypass TTL (we use absolute mtime).
* The cache is **fail-soft**: if a file is corrupt or unreadable, we
  log a warning and return ``None`` so the caller falls back to the
  exchange. We never raise from the cache layer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neon_radar.config.models import TimeFrame
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class CacheEntry:
    """Metadata about a cached :class:`KlineSeries`."""

    symbol: Symbol
    timeframe: TimeFrame
    path: Path
    age_seconds: float
    candle_count: int


class KlineCache:
    """Filesystem cache for :class:`KlineSeries`.

    Parameters
    ----------
    directory
        Directory for cache files. Created if missing.
    ttl_seconds
        Maximum age for a cache entry. ``0`` disables TTL (cache is
        effectively immutable; only manual deletion invalidates it).
    clock
        Callable returning current Unix time in seconds. Injectable
        for tests.
    """

    def __init__(
        self,
        directory: Path,
        *,
        ttl_seconds: int = 300,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be non-negative, got {ttl_seconds}")
        self._dir = directory
        self._ttl = ttl_seconds
        self._clock = clock
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _path_for(self, symbol: Symbol, timeframe: TimeFrame) -> Path:
        # Flat filename — works on every filesystem. Symlinks safe.
        return self._dir / f"{symbol}_{timeframe.value}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, symbol: Symbol, timeframe: TimeFrame) -> KlineSeries | None:
        """Return cached series if present and fresh, else ``None``."""
        path = self._path_for(symbol, timeframe)
        if not path.is_file():
            return None
        age = self._clock() - path.stat().st_mtime
        if self._ttl > 0 and age > self._ttl:
            logger.debug("Cache miss (stale): %s (age=%.1fs)", path.name, age)
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Cache read failed for %s: %s", path, exc)
            return None
        try:
            return self._deserialize(data)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Cache data invalid for %s: %s", path, exc)
            return None

    def put(self, series: KlineSeries) -> None:
        """Persist ``series`` to disk. Atomic via temp file + rename."""
        path = self._path_for(series.symbol, series.timeframe)
        payload = json.dumps(self._serialize(series), ensure_ascii=False)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("Cache write failed for %s: %s", path, exc)

    def invalidate(self, symbol: Symbol, timeframe: TimeFrame) -> bool:
        """Remove the cache file for ``(symbol, timeframe)``."""
        path = self._path_for(symbol, timeframe)
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError as exc:
            logger.warning("Cache invalidate failed for %s: %s", path, exc)
            return False

    def clear(self) -> int:
        """Delete all cache files in the directory. Returns count deleted."""
        count = 0
        for p in self._dir.glob("*.json"):
            try:
                p.unlink()
                count += 1
            except OSError:
                pass
        return count

    def list_entries(self) -> tuple[CacheEntry, ...]:
        """Return metadata about every cached series (for diagnostics)."""
        entries: list[CacheEntry] = []
        for path in sorted(self._dir.glob("*.json")):
            name = path.stem
            try:
                symbol_str, tf_str = name.rsplit("_", 1)
                symbol = Symbol(symbol_str)
                timeframe = TimeFrame(tf_str)
            except (ValueError, KeyError):
                continue
            try:
                age = self._clock() - path.stat().st_mtime
                candle_count = len(json.loads(path.read_text(encoding="utf-8"))["candles"])
            except (OSError, json.JSONDecodeError, KeyError):
                continue
            entries.append(
                CacheEntry(
                    symbol=symbol,
                    timeframe=timeframe,
                    path=path,
                    age_seconds=age,
                    candle_count=candle_count,
                )
            )
        return tuple(entries)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(series: KlineSeries) -> dict[str, object]:
        return {
            "symbol": str(series.symbol),
            "timeframe": series.timeframe.value,
            "candles": [
                {
                    "open_time": c.open_time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "close_time": c.close_time,
                    "quote_volume": c.quote_volume,
                    "trades": c.trades,
                }
                for c in series
            ],
        }

    @staticmethod
    def _deserialize(data: dict[str, object]) -> KlineSeries:
        return KlineSeries(
            symbol=Symbol(str(data["symbol"])),
            timeframe=TimeFrame(str(data["timeframe"])),
            candles=tuple(
                OHLCV(
                    open_time=int(c["open_time"]),
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=float(c["volume"]),
                    close_time=c.get("close_time"),  # type: ignore[arg-type]
                    quote_volume=c.get("quote_volume"),  # type: ignore[arg-type]
                    trades=c.get("trades"),  # type: ignore[arg-type]
                )
                for c in data["candles"]  # type: ignore[union-attr]
            ),
        )
