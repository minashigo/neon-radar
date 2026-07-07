"""Tests for the Bollinger Bands scoring rule."""

from __future__ import annotations

from neon_radar.application.services.indicator_pipeline import IndicatorSpec, compute_indicators
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import Symbol
from neon_radar.domain.scoring import BollingerBandsRule, RuleRegistry
from tests.conftest import make_series


def _state(closes: list[float]) -> MarketState:
    from neon_radar.config.models import TimeFrame

    series = make_series(closes, timeframe=TimeFrame.D1)
    indicators = compute_indicators(series, [IndicatorSpec(name="bollinger", params={}, tag="")])
    return MarketState(
        symbol=Symbol("BTCUSDT"),
        timestamp=0,
        primary_series=series,
        indicator_series=tuple(indicators),
    )


class TestBollingerBandsRule:
    def test_registered(self) -> None:
        assert RuleRegistry.is_registered("bollinger_bands")

    def test_bullish_signal_when_price_above_upper_band(self) -> None:
        sig = BollingerBandsRule().evaluate(_state([100.0] * 25 + [120.0]))
        assert sig is not None
        assert sig.value > 0
        assert sig.confidence > 0.5

    def test_bearish_signal_when_price_below_lower_band(self) -> None:
        sig = BollingerBandsRule().evaluate(_state([100.0] * 25 + [80.0]))
        assert sig is not None
        assert sig.value < 0
        assert sig.confidence > 0.5

    def test_neutral_when_inside_bands(self) -> None:
        sig = BollingerBandsRule().evaluate(_state([100.0] * 26))
        assert sig is None
