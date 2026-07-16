"""Tests for the MACD momentum rule."""

from __future__ import annotations

from neon_radar.application.services.indicator_pipeline import IndicatorSpec, compute_indicators
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import Symbol
from neon_radar.domain.scoring import MACDMomentumRule, RuleRegistry
from tests.conftest import make_series


def _state(closes: list[float]):
    """Build a MarketState with a MACD indicator series."""
    from neon_radar.config.models import TimeFrame

    series = make_series(closes, timeframe=TimeFrame.D1)
    specs = [IndicatorSpec(name="macd", params={}, tag="")]
    indicators = compute_indicators(series, specs)
    return MarketState(
        symbol=Symbol("BTCUSDT"),
        timestamp=0,
        primary_series=series,
        indicator_series=tuple(indicators),
    )


class TestMACDMomentumRule:
    def test_registered(self) -> None:
        assert RuleRegistry.is_registered("macd_momentum")

    def test_bullish_state_returns_positive(self) -> None:
        closes = [100.0] * 30 + [100.0 + i**2 for i in range(10)]
        sig = MACDMomentumRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.value > 0

    def test_bearish_state_returns_negative(self) -> None:
        closes = [200.0] * 30 + [200.0 - i**2 for i in range(10)]
        sig = MACDMomentumRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.value < 0

    def test_flat_or_unclear_returns_none(self) -> None:
        closes = [100.0] * 60
        assert MACDMomentumRule().evaluate(_state(closes)) is None
