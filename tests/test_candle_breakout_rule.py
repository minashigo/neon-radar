"""Tests for the candle breakout rule."""

from __future__ import annotations

from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import Symbol
from neon_radar.domain.scoring import CandleBreakoutRule, RuleRegistry
from tests.conftest import make_series


def _state(closes: list[float]) -> MarketState:
    from neon_radar.config.models import TimeFrame

    series = make_series(closes, timeframe=TimeFrame.D1)
    return MarketState(
        symbol=Symbol("BTCUSDT"),
        timestamp=0,
        primary_series=series,
        indicator_series=(),
    )


class TestCandleBreakoutRule:
    def test_registered(self) -> None:
        assert RuleRegistry.is_registered("candle_breakout")

    def test_bullish_breakout_returns_positive(self) -> None:
        sig = CandleBreakoutRule().evaluate(_state([100.0, 90.0, 95.0]))
        assert sig is not None
        assert sig.value > 0
        assert sig.confidence > 0.5

    def test_bearish_breakout_returns_negative(self) -> None:
        sig = CandleBreakoutRule().evaluate(_state([100.0, 110.0, 105.0]))
        assert sig is not None
        assert sig.value < 0
        assert sig.confidence > 0.5

    def test_neutral_returns_none(self) -> None:
        sig = CandleBreakoutRule().evaluate(_state([100.0, 95.0, 96.0]))
        assert sig is None
