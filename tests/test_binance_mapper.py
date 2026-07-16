"""Tests for the Binance JSON → domain mappers."""

from __future__ import annotations

import pytest

from neon_radar.config.models import TimeFrame
from neon_radar.domain.exceptions import ParseError
from neon_radar.domain.funding import FundingRate, OpenInterest
from neon_radar.domain.models import KlineSeries, Symbol, TickerStats
from neon_radar.infrastructure.exchanges.binance.mapper import (
    map_funding_rate_from_premium_index,
    map_kline,
    map_klines,
    map_open_interest,
    map_ticker,
)

# ---------------------------------------------------------------------------
# map_kline
# ---------------------------------------------------------------------------


class TestMapKline:
    def test_full_row(self) -> None:
        raw = [
            1_700_000_000_000,  # open_time
            "100.50",  # open
            "110.00",  # high
            "99.00",  # low
            "105.75",  # close
            "1234.567",  # volume
            1_700_000_005_999,  # close_time
            "130000.00",  # quote_volume
            42,  # trades
            "600.0",  # taker buy base
            "60000.0",  # taker buy quote
            "0",  # ignore
        ]
        c = map_kline(raw)
        assert c.open_time == 1_700_000_000_000
        assert c.open == pytest.approx(100.50)
        assert c.high == pytest.approx(110.00)
        assert c.low == pytest.approx(99.00)
        assert c.close == pytest.approx(105.75)
        assert c.volume == pytest.approx(1234.567)
        assert c.close_time == 1_700_000_005_999
        assert c.quote_volume == pytest.approx(130000.00)
        assert c.trades == 42

    def test_minimal_row(self) -> None:
        # Binance sometimes returns shorter rows for older endpoints.
        raw = [1_700_000_000_000, "100", "110", "99", "105", "1000"]
        c = map_kline(raw)
        assert c.open_time == 1_700_000_000_000
        assert c.close_time is None
        assert c.quote_volume is None
        assert c.trades is None

    def test_rejects_non_list(self) -> None:
        with pytest.raises(ParseError, match="must be a list"):
            map_kline({"openTime": 1})  # type: ignore[arg-type]

    def test_rejects_too_short(self) -> None:
        with pytest.raises(ParseError, match="expected at least 6"):
            map_kline([1, 2, 3])

    def test_rejects_non_numeric_open(self) -> None:
        with pytest.raises(ParseError, match=r"kline\.open"):
            map_kline([1, "not-a-number", 2, 3, 4, 5])

    def test_rejects_negative_volume_via_ohlcv_validator(self) -> None:
        # Volume is parsed as float first, then OHLCV validator rejects.
        raw = [1, "100", "110", "99", "105", "-1"]
        with pytest.raises((ParseError, ValueError)):
            map_kline(raw)


# ---------------------------------------------------------------------------
# map_klines
# ---------------------------------------------------------------------------


class TestMapKlines:
    def test_basic(self) -> None:
        raw = [
            [1, "100", "110", "99", "105", "1000"],
            [2, "105", "115", "104", "110", "2000"],
        ]
        s = map_klines(raw, symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.H4)
        assert isinstance(s, KlineSeries)
        assert len(s) == 2
        assert s[0].open_time == 1
        assert s[1].close == pytest.approx(110)

    def test_empty(self) -> None:
        s = map_klines([], symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.D1)
        assert s.is_empty

    def test_rejects_non_list(self) -> None:
        with pytest.raises(ParseError):
            map_klines({"klines": []}, symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.D1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# map_ticker
# ---------------------------------------------------------------------------


class TestMapTicker:
    def test_basic(self) -> None:
        raw = {
            "symbol": "BTCUSDT",
            "lastPrice": "30123.45",
            "priceChangePercent": "2.500",
            "highPrice": "31000.00",
            "lowPrice": "29500.00",
            "volume": "12345.6",
            "quoteVolume": "371234567.89",
            "sumOpenInterest": "54321.0",
            "time": 1_700_000_000_000,
        }
        t = map_ticker(raw)
        assert isinstance(t, TickerStats)
        assert t.symbol == "BTCUSDT"
        assert t.last_price == pytest.approx(30123.45)
        assert t.price_change_percent == pytest.approx(2.5)
        assert t.open_interest == pytest.approx(54321.0)
        assert t.timestamp == 1_700_000_000_000

    def test_missing_field(self) -> None:
        with pytest.raises(ParseError):
            map_ticker({"symbol": "BTCUSDT"})  # type: ignore[arg-type]

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(ParseError):
            map_ticker([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# map_funding_rate_from_premium_index
# ---------------------------------------------------------------------------


class TestMapFundingRate:
    def test_basic(self) -> None:
        raw = {
            "symbol": "BTCUSDT",
            "markPrice": "30100.50",
            "indexPrice": "30100.00",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1_700_000_000_000,
            "time": 1_699_999_000_000,
        }
        fr = map_funding_rate_from_premium_index(raw, symbol=Symbol("BTCUSDT"))
        assert isinstance(fr, FundingRate)
        assert fr.rate == pytest.approx(0.0001)
        assert fr.mark_price == pytest.approx(30100.50)
        assert fr.next_funding_time == 1_700_000_000_000
        assert fr.is_positive

    def test_negative_rate(self) -> None:
        fr = map_funding_rate_from_premium_index(
            {"lastFundingRate": "-0.0001", "symbol": "BTCUSDT"},
            symbol=Symbol("BTCUSDT"),
        )
        assert not fr.is_positive

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(ParseError):
            map_funding_rate_from_premium_index(
                "not a dict",
                symbol=Symbol("BTCUSDT"),  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# map_open_interest
# ---------------------------------------------------------------------------


class TestMapOpenInterest:
    def test_basic(self) -> None:
        raw = {
            "symbol": "BTCUSDT",
            "sumOpenInterest": "12345.6",
            "sumOpenInterestValue": "371234567.89",
            "time": 1_700_000_000_000,
        }
        oi = map_open_interest(raw, symbol=Symbol("BTCUSDT"))
        assert isinstance(oi, OpenInterest)
        assert oi.value == pytest.approx(12345.6)
        assert oi.value_quote == pytest.approx(371234567.89)

    def test_only_base(self) -> None:
        oi = map_open_interest(
            {"symbol": "BTCUSDT", "sumOpenInterest": "1.0"},
            symbol=Symbol("BTCUSDT"),
        )
        assert oi.value == pytest.approx(1.0)
        assert oi.value_quote is None
