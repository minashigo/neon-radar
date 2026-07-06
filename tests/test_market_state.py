"""Tests for ``MarketState``."""

from __future__ import annotations

import pytest

from neon_radar.config.models import TimeFrame
from neon_radar.domain.funding import FundingRate, OpenInterest
from neon_radar.domain.indicators.base import (
    IndicatorKind,
    IndicatorSeries,
    IndicatorSnapshot,
    IndicatorValue,
)
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import (
    OHLCV,
    KlineSeries,
    Symbol,
    TickerStats,
)


def _series(symbol: str, tf: TimeFrame, n: int = 5) -> KlineSeries:
    candles = tuple(
        OHLCV(
            open_time=1_700_000_000_000 + i * tf.seconds * 1000,
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000.0,
        )
        for i in range(n)
    )
    return KlineSeries(symbol=Symbol(symbol), timeframe=tf, candles=candles)


def _indicator_series(name: str, n: int = 5) -> IndicatorSeries:
    snaps = tuple(
        IndicatorSnapshot(
            timestamp=1_700_000_000_000 + i,
            values=(
                IndicatorValue("a", float(i)),
                IndicatorValue("b", float(i) * 2),
            ),
        )
        for i in range(n)
    )
    return IndicatorSeries(name=name, kind=IndicatorKind.META, snapshots=snaps)


class TestMarketState:
    def test_minimal(self) -> None:
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=1_700_000_000_000,
            primary_series=_series("BTCUSDT", TimeFrame.H4),
        )
        assert state.symbol == "BTCUSDT"
        assert state.higher_tf_series is None
        assert state.indicator_series == ()
        assert state.ticker is None
        assert state.funding_rate is None
        assert state.open_interest is None

    def test_full(self) -> None:
        funding = FundingRate(symbol="BTCUSDT", rate=0.0001)
        oi = OpenInterest(symbol="BTCUSDT", value=50_000.0)
        ticker = TickerStats(
            symbol="BTCUSDT",
            last_price=30_000.0,
            price_change_percent=1.5,
            high_24h=30_500.0,
            low_24h=29_500.0,
            volume_24h=1000.0,
            quote_volume_24h=30_000_000.0,
        )
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=1_700_000_000_000,
            primary_series=_series("BTCUSDT", TimeFrame.H4),
            higher_tf_series=_series("BTCUSDT", TimeFrame.D1),
            indicator_series=(
                _indicator_series("ema", 5),
                _indicator_series("rsi", 5),
            ),
            ticker=ticker,
            funding_rate=funding,
            open_interest=oi,
        )
        assert state.funding_rate is funding
        assert state.open_interest is oi

    def test_string_symbol_normalised(self) -> None:
        state = MarketState(
            symbol="btcusdt",  # type: ignore[arg-type]
            timestamp=1,
            primary_series=_series("btcusdt", TimeFrame.H4),
        )
        assert state.symbol == "BTCUSDT"

    def test_rejects_symbol_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            MarketState(
                symbol=Symbol("BTCUSDT"),
                timestamp=1,
                primary_series=_series("ETHUSDT", TimeFrame.H4),
            )

    def test_rejects_lower_higher_tf(self) -> None:
        with pytest.raises(ValueError, match="strictly higher"):
            MarketState(
                symbol=Symbol("BTCUSDT"),
                timestamp=1,
                primary_series=_series("BTCUSDT", TimeFrame.D1),
                higher_tf_series=_series("BTCUSDT", TimeFrame.H4),
            )

    def test_rejects_non_klineseries(self) -> None:
        with pytest.raises(TypeError):
            MarketState(
                symbol=Symbol("BTCUSDT"),
                timestamp=1,
                primary_series="not a series",  # type: ignore[arg-type]
            )

    def test_get_indicator_found(self) -> None:
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=1,
            primary_series=_series("BTCUSDT", TimeFrame.H4),
            indicator_series=(_indicator_series("ema"),),
        )
        ind = state.get_indicator("ema")
        assert ind is not None
        assert ind.latest_value("a") == 4.0  # last index

    def test_get_indicator_missing(self) -> None:
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=1,
            primary_series=_series("BTCUSDT", TimeFrame.H4),
        )
        assert state.get_indicator("nope") is None

    def test_get_indicator_value_helper(self) -> None:
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=1,
            primary_series=_series("BTCUSDT", TimeFrame.H4),
            indicator_series=(_indicator_series("multi"),),
        )
        assert state.get_indicator_value("multi", "b") == 8.0  # 4*2
        # Without field — returns first.
        assert state.get_indicator_value("multi") == 4.0
