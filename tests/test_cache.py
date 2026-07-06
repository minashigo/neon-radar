"""Tests for :class:`KlineCache`."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import pytest

from neon_radar.config.models import TimeFrame
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.infrastructure.cache import KlineCache

if TYPE_CHECKING:
    from pathlib import Path


def _series(symbol: str, timeframe: TimeFrame, n: int = 3) -> KlineSeries:
    candles = tuple(
        OHLCV(
            open_time=1_700_000_000_000 + i,
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000.0,
        )
        for i in range(n)
    )
    return KlineSeries(symbol=Symbol(symbol), timeframe=timeframe, candles=candles)


class TestKlineCache:
    def test_directory_created(self, tmp_path: Path) -> None:
        target = tmp_path / "cache" / "nested"
        KlineCache(target)
        assert target.is_dir()

    def test_put_and_get(self, tmp_path: Path) -> None:
        cache = KlineCache(tmp_path, ttl_seconds=300)
        s = _series("BTCUSDT", TimeFrame.H4)
        cache.put(s)
        loaded = cache.get(Symbol("BTCUSDT"), TimeFrame.H4)
        assert loaded is not None
        assert len(loaded) == 3
        assert loaded[0].open_time == 1_700_000_000_000

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        cache = KlineCache(tmp_path)
        assert cache.get(Symbol("BTCUSDT"), TimeFrame.H4) is None

    def test_ttl_zero_never_expires(
        self, tmp_path: Path
    ) -> None:
        cache = KlineCache(tmp_path, ttl_seconds=0)
        s = _series("BTCUSDT", TimeFrame.H4)
        cache.put(s)
        # Even after 1 hour, still fresh.
        time.sleep(0.01)
        loaded = cache.get(Symbol("BTCUSDT"), TimeFrame.H4)
        assert loaded is not None

    def test_ttl_expires(self, tmp_path: Path) -> None:
        # Use a fake clock to control age.
        fake_now = [1_700_000_000.0]

        def clock() -> float:
            return fake_now[0]

        cache = KlineCache(tmp_path, ttl_seconds=60, clock=clock)
        cache.put(_series("BTCUSDT", TimeFrame.H4))
        # Pin the mtime so the fake clock matches.
        path = tmp_path / "BTCUSDT_4h.json"
        import os

        os.utime(path, (fake_now[0], fake_now[0]))
        # Right after put: fresh.
        assert cache.get(Symbol("BTCUSDT"), TimeFrame.H4) is not None
        # Advance clock past TTL.
        fake_now[0] += 61
        assert cache.get(Symbol("BTCUSDT"), TimeFrame.H4) is None

    def test_corrupt_file_returns_none(self, tmp_path: Path) -> None:
        cache = KlineCache(tmp_path)
        (tmp_path / "BTCUSDT_4h.json").write_text("not valid json {{{")
        assert cache.get(Symbol("BTCUSDT"), TimeFrame.H4) is None

    def test_invalidate(self, tmp_path: Path) -> None:
        cache = KlineCache(tmp_path)
        cache.put(_series("BTCUSDT", TimeFrame.H4))
        assert cache.invalidate(Symbol("BTCUSDT"), TimeFrame.H4) is True
        assert cache.get(Symbol("BTCUSDT"), TimeFrame.H4) is None
        # Idempotent.
        assert cache.invalidate(Symbol("BTCUSDT"), TimeFrame.H4) is True

    def test_clear(self, tmp_path: Path) -> None:
        cache = KlineCache(tmp_path)
        cache.put(_series("BTCUSDT", TimeFrame.H4))
        cache.put(_series("ETHUSDT", TimeFrame.D1))
        assert len(cache.list_entries()) == 2
        removed = cache.clear()
        assert removed == 2
        assert len(cache.list_entries()) == 0

    def test_list_entries(self, tmp_path: Path) -> None:
        cache = KlineCache(tmp_path)
        cache.put(_series("BTCUSDT", TimeFrame.H4, n=5))
        entries = cache.list_entries()
        assert len(entries) == 1
        assert entries[0].symbol == "BTCUSDT"
        assert entries[0].timeframe == TimeFrame.H4
        assert entries[0].candle_count == 5
        assert entries[0].age_seconds >= 0

    def test_rejects_negative_ttl(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            KlineCache(tmp_path, ttl_seconds=-1)

    def test_round_trip_with_extra_fields(self, tmp_path: Path) -> None:
        """close_time, quote_volume, trades survive round-trip."""
        cache = KlineCache(tmp_path)
        candle = OHLCV(
            open_time=1,
            open=100.0,
            high=110.0,
            low=99.0,
            close=105.0,
            volume=1000.0,
            close_time=10,
            quote_volume=105_000.0,
            trades=42,
        )
        series = KlineSeries(
            symbol=Symbol("BTCUSDT"),
            timeframe=TimeFrame.D1,
            candles=(candle,),
        )
        cache.put(series)
        loaded = cache.get(Symbol("BTCUSDT"), TimeFrame.D1)
        assert loaded is not None
        c = loaded[0]
        assert c.close_time == 10
        assert c.quote_volume == 105_000.0
        assert c.trades == 42

    def test_files_are_valid_json(self, tmp_path: Path) -> None:
        """Cache files must be inspectable with cat/jq."""
        cache = KlineCache(tmp_path)
        cache.put(_series("BTCUSDT", TimeFrame.H4, n=2))
        path = tmp_path / "BTCUSDT_4h.json"
        data = json.loads(path.read_text())
        assert data["symbol"] == "BTCUSDT"
        assert data["timeframe"] == "4h"
        assert isinstance(data["candles"], list)
        assert len(data["candles"]) == 2
