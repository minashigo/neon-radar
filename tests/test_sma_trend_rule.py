"""Tests for the SMA trend rule."""

from __future__ import annotations

from neon_radar.application.services.indicator_pipeline import IndicatorSpec, compute_indicators
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import Symbol
from neon_radar.domain.scoring import RuleRegistry, SMATrendRule
from tests.conftest import make_series


def _state(closes: list[float]) -> MarketState:
    """Build a MarketState with SMA indicator series."""
    from neon_radar.config.models import TimeFrame

    series = make_series(closes, timeframe=TimeFrame.D1)
    specs = [
        IndicatorSpec(name="sma", params={"period": 20}, tag="20"),
        IndicatorSpec(name="sma", params={"period": 50}, tag="50"),
    ]
    indicators = compute_indicators(series, specs)
    return MarketState(
        symbol=Symbol("BTCUSDT"),
        timestamp=0,
        primary_series=series,
        indicator_series=tuple(indicators),
    )


class TestSMATrendRule:
    def test_registered(self) -> None:
        assert RuleRegistry.is_registered("sma_trend")

    def test_bullish_gap_returns_positive_signal(self) -> None:
        closes = [100.0] * 60 + [110.0] * 20
        sig = SMATrendRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.value > 0
        assert sig.confidence > 0

    def test_bearish_gap_returns_negative_signal(self) -> None:
        closes = [110.0] * 60 + [90.0] * 20
        sig = SMATrendRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.value < 0
        assert sig.confidence > 0

    def test_flat_series_returns_none(self) -> None:
        sig = SMATrendRule().evaluate(_state([100.0] * 80))
        assert sig is None
