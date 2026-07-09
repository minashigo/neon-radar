import pytest

from neon_radar.config.models import TimeFrame
from neon_radar.domain.indicators.base import (
    IndicatorKind,
    IndicatorSeries,
    IndicatorSnapshot,
    IndicatorValue,
)
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import KlineSeries, Symbol
from neon_radar.domain.scoring.rules.higher_tf_trend import HigherTimeframeTrendRule


def make_state(fast: float | None, slow: float | None) -> MarketState:
    series = KlineSeries(symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.H4, candles=())
    htf_series = KlineSeries(symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.D1, candles=())

    indicators = []
    if fast is not None:
        indicators.append(
            IndicatorSeries(
                name="htf_ema_20",
                kind=IndicatorKind.OVERLAY,
                snapshots=(IndicatorSnapshot(timestamp=0, values=(IndicatorValue("ema", fast),)),),
            )
        )
    if slow is not None:
        indicators.append(
            IndicatorSeries(
                name="htf_ema_50",
                kind=IndicatorKind.OVERLAY,
                snapshots=(IndicatorSnapshot(timestamp=0, values=(IndicatorValue("ema", slow),)),),
            )
        )

    return MarketState(
        symbol=Symbol("BTCUSDT"),
        timestamp=1000,
        primary_series=series,
        higher_tf_series=htf_series,
        indicator_series=tuple(indicators),
    )


class TestHigherTimeframeTrendRule:
    def test_missing_higher_tf_series_returns_none(self) -> None:
        rule = HigherTimeframeTrendRule()
        series = KlineSeries(symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.H4, candles=())
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=1000,
            primary_series=series,
            higher_tf_series=None,
        )
        assert rule.evaluate(state) is None

    def test_missing_indicators_returns_none(self) -> None:
        rule = HigherTimeframeTrendRule()
        state = make_state(fast=None, slow=100.0)
        assert rule.evaluate(state) is None

    def test_bullish_trend(self) -> None:
        rule = HigherTimeframeTrendRule(fast_period=20, slow_period=50)
        state = make_state(fast=105.0, slow=100.0)  # +5%

        signal = rule.evaluate(state)
        assert signal is not None
        assert signal.value == pytest.approx(1.0)
        assert signal.confidence == pytest.approx(1.0)
        assert signal.name == "higher_tf_trend"
        assert "↑" in signal.description

    def test_bearish_trend(self) -> None:
        rule = HigherTimeframeTrendRule()
        state = make_state(fast=95.0, slow=100.0)  # -5%

        signal = rule.evaluate(state)
        assert signal is not None
        assert signal.value == pytest.approx(-1.0)
        assert signal.confidence == pytest.approx(1.0)
        assert "↓" in signal.description

    def test_neutral_trend(self) -> None:
        rule = HigherTimeframeTrendRule(threshold_pct=0.01)
        state = make_state(fast=100.5, slow=100.0)  # +0.5% gap

        assert rule.evaluate(state) is None
