"""Tests for the domain layer."""

from __future__ import annotations

import dataclasses

import pytest

from neon_radar.config.models import TimeFrame
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol, TickerStats


def _candle(
    open_time: int,
    open_: float = 100.0,
    high: float = 110.0,
    low: float = 90.0,
    close: float = 105.0,
    volume: float = 1000.0,
) -> OHLCV:
    return OHLCV(
        open_time=open_time,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class TestSymbol:
    def test_uppercases(self) -> None:
        assert Symbol("btcusdt") == "BTCUSDT"

    def test_base_quote(self) -> None:
        s = Symbol("BTCUSDT")
        assert s.base() == "BTC"
        assert s.quote() == "USDT"

    def test_rejects_invalid(self) -> None:
        with pytest.raises(ValueError):
            Symbol("BTC-USDT")
        with pytest.raises(ValueError):
            Symbol("")

    def test_is_str_subclass(self) -> None:
        assert isinstance(Symbol("BTCUSDT"), str)
        # Behaves as a string in dict keys.
        d = {Symbol("BTCUSDT"): 1}
        assert d["BTCUSDT"] == 1


class TestOHLCV:
    def test_basic_construction(self) -> None:
        c = _candle(1_700_000_000_000)
        assert c.is_bullish
        assert c.range == 20.0
        assert c.body == 5.0

    def test_datetime_is_utc(self) -> None:
        c = _candle(1_700_000_000_000)
        dt = c.datetime
        assert dt.tzinfo is not None
        assert dt.timestamp() == 1_700_000_000.0

    def test_rejects_high_less_than_low(self) -> None:
        with pytest.raises(ValueError):
            OHLCV(
                open_time=0,
                open=100,
                high=90,
                low=110,
                close=100,
                volume=0,
            )

    def test_rejects_negative_volume(self) -> None:
        with pytest.raises(ValueError):
            _candle(0, volume=-1)

    def test_immutable(self) -> None:
        c = _candle(0)
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            c.close = 200  # type: ignore[misc]


class TestKlineSeries:
    def test_empty(self) -> None:
        s = KlineSeries(symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.D1)
        assert s.is_empty
        assert s.latest() is None
        assert len(s) == 0

    def test_iteration(self) -> None:
        candles = tuple(_candle(t) for t in (1, 2, 3))
        s = KlineSeries(
            symbol=Symbol("BTCUSDT"),
            timeframe=TimeFrame.H4,
            candles=candles,
        )
        assert list(s) == list(candles)
        assert s[0] == candles[0]
        assert s.latest() == candles[-1]

    def test_rejects_unsorted(self) -> None:
        with pytest.raises(ValueError, match="not sorted"):
            KlineSeries(
                symbol=Symbol("BTCUSDT"),
                timeframe=TimeFrame.D1,
                candles=(_candle(2), _candle(1)),
            )

    def test_last_n(self) -> None:
        candles = tuple(_candle(t) for t in range(10))
        s = KlineSeries(
            symbol=Symbol("BTCUSDT"),
            timeframe=TimeFrame.D1,
            candles=candles,
        )
        tail = s.last_n(3)
        assert len(tail) == 3
        assert tail[0] is candles[-3]

    def test_normalises_string_symbol(self) -> None:
        s = KlineSeries(symbol="btcusdt", timeframe=TimeFrame.D1)  # type: ignore[arg-type]
        assert isinstance(s.symbol, Symbol)
        assert s.symbol == "BTCUSDT"


class TestTickerStats:
    def test_basic(self) -> None:
        t = TickerStats(
            symbol="BTCUSDT",
            last_price=30_000,
            price_change_percent=2.5,
            high_24h=31_000,
            low_24h=29_000,
            volume_24h=100_000,
            quote_volume_24h=3_000_000_000,
        )
        assert t.is_bullish_24h
        assert t.symbol == "BTCUSDT"
        assert isinstance(t.symbol, Symbol)
