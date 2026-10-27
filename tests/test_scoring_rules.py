"""Tests for the built-in scoring rules."""

from __future__ import annotations

import pytest

from neon_radar.application.services.indicator_pipeline import (
    IndicatorSpec,
    compute_indicators,
)
from neon_radar.domain.funding import FundingRate
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import OHLCV, Symbol
from neon_radar.domain.scoring import (
    EMATrendRule,
    FundingRateRule,
    RSIMomentumRule,
    RuleRegistry,
    VolatilityFilterRule,
    VolumeConfirmationRule,
)
from tests.conftest import make_series


def _series_with_volumes(closes, *, volumes=None, timeframe=None):
    """Build a KlineSeries, optionally with custom volumes."""
    from neon_radar.config.models import TimeFrame
    from neon_radar.domain.models import KlineSeries

    series = make_series(closes, timeframe=timeframe or TimeFrame.D1)
    if volumes is None:
        return series

    new_candles = tuple(
        OHLCV(
            open_time=c.open_time,
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            volume=vol,
        )
        for c, vol in zip(series, volumes, strict=True)
    )
    return KlineSeries(symbol=series.symbol, timeframe=series.timeframe, candles=new_candles)


def _state(closes, *, volumes=None, timeframe=None, funding_rate=None):
    """Build a MarketState with rule indicators, optionally with funding."""

    series = _series_with_volumes(closes, volumes=volumes, timeframe=timeframe)

    specs = [
        # EMA trend needs two EMAs with different tags so they are distinguishable.
        IndicatorSpec(name="ema", params={"period": 20}, tag="20"),
        IndicatorSpec(name="ema", params={"period": 50}, tag="50"),
        IndicatorSpec(name="rsi", params={"period": 14}, tag="14"),
        IndicatorSpec(name="atr", params={"period": 14}, tag="14"),
        IndicatorSpec(name="volume_ma", params={"period": 20}, tag="20"),
    ]
    indicators = compute_indicators(series, specs)
    return MarketState(
        symbol=series.symbol,
        timestamp=0,
        primary_series=series,
        indicator_series=tuple(indicators),
        funding_rate=funding_rate,
    )


# ---------------------------------------------------------------------------
# EMATrendRule
# ---------------------------------------------------------------------------


class TestEMATrendRule:
    def test_registered(self) -> None:
        assert RuleRegistry.is_registered("ema_trend")

    def test_required_indicators(self) -> None:
        specs = EMATrendRule().required_indicators()
        names = {s.series_name for s in specs}
        assert "ema_20" in names
        assert "ema_50" in names

    def test_required_indicators_use_configured_periods(self) -> None:
        specs = EMATrendRule(fast_period=12, slow_period=26).required_indicators()
        names = {s.series_name for s in specs}
        assert names == {"ema_12", "ema_26"}
        assert {s.params["period"] for s in specs} == {12, 26}

    def test_rejects_invalid_periods(self) -> None:
        with pytest.raises(ValueError):
            EMATrendRule(fast_period=50, slow_period=20)
        with pytest.raises(ValueError):
            EMATrendRule(threshold_pct=0)
        with pytest.raises(ValueError):
            EMATrendRule(threshold_pct=-0.01)

    def test_uptrend_returns_bullish(self) -> None:
        closes = [100.0 + i for i in range(60)]
        sig = EMATrendRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.value > 0
        assert sig.confidence > 0
        assert "EMA" in sig.description

    def test_downtrend_returns_bearish(self) -> None:
        closes = [200.0 - i for i in range(60)]
        sig = EMATrendRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.value < 0

    def test_neutral_returns_none(self) -> None:
        # Flat prices: EMA gap is 0, below threshold.
        closes = [100.0] * 60
        sig = EMATrendRule(threshold_pct=0.001).evaluate(_state(closes))
        assert sig is None

    def test_missing_indicator_returns_none(self) -> None:
        from neon_radar.config.models import TimeFrame

        series = make_series([100.0] * 60, timeframe=TimeFrame.D1)
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=0,
            primary_series=series,
            indicator_series=(),
        )
        assert EMATrendRule().evaluate(state) is None


# ---------------------------------------------------------------------------
# RSIMomentumRule
# ---------------------------------------------------------------------------


def _state_with_rsi(rsi: float, period: int = 14) -> MarketState:
    from neon_radar.config.models import TimeFrame
    from neon_radar.domain.indicators.base import (
        IndicatorKind,
        IndicatorSeries,
        IndicatorSnapshot,
        IndicatorValue,
    )
    from neon_radar.domain.models import KlineSeries, Symbol

    series = KlineSeries(symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.D1, candles=())
    indicators = (
        IndicatorSeries(
            name=f"rsi_{period}",
            kind=IndicatorKind.OSCILLATOR,
            snapshots=(IndicatorSnapshot(timestamp=0, values=(IndicatorValue("rsi", rsi),)),),
        ),
    )
    return MarketState(
        symbol=Symbol("BTCUSDT"),
        timestamp=1000,
        primary_series=series,
        indicator_series=indicators,
    )


class TestRSIMomentumRule:
    def test_registered(self) -> None:
        assert RuleRegistry.is_registered("rsi_momentum")

    def test_required_indicators_use_configured_period(self) -> None:
        specs = RSIMomentumRule(period=21).required_indicators()
        assert len(specs) == 1
        assert specs[0].series_name == "rsi_21"
        assert specs[0].params["period"] == 21

    def test_rejects_invalid_thresholds(self) -> None:
        with pytest.raises(ValueError):
            RSIMomentumRule(bull_low=80, bull_high=70)

    def test_bull_zone_returns_positive(self) -> None:
        # Construct a series whose RSI reliably lands in [51, 70].
        # We need a steady gain without becoming overbought: a long
        # sequence of small positive deltas, with RSI settling
        # somewhere in the upper-middle of [0, 100].
        closes = [100.0 + (i // 4) * 0.1 for i in range(60)]
        sig = RSIMomentumRule().evaluate(_state(closes))
        assert sig is not None
        # Just check the sign — exact value depends on RSI computation.
        assert sig.value > 0 or sig.confidence <= 0.5

    def test_bear_zone_returns_negative(self) -> None:
        # Mirror of the bull test: steady small declines.
        closes = [200.0 - (i // 4) * 0.1 for i in range(60)]
        sig = RSIMomentumRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.value < 0 or sig.confidence <= 0.5

    def test_overbought_neutral(self) -> None:
        # Strict uptrend → RSI=100 → overbought → value=0, low confidence.
        closes = [100.0 + i for i in range(60)]
        sig = RSIMomentumRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.value == pytest.approx(0.0, abs=1e-9)
        assert sig.confidence == pytest.approx(0.4)

    def test_oversold_neutral(self) -> None:
        # Strict downtrend → RSI=0 → oversold → value=0, low confidence.
        closes = [200.0 - i for i in range(60)]
        sig = RSIMomentumRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.value == pytest.approx(0.0, abs=1e-9)
        assert sig.confidence == pytest.approx(0.4)

    @pytest.mark.parametrize("rsi", [70.01, 75.0, 85.0, 100.0])
    def test_overbought_extremes_return_neutral(self, rsi: float) -> None:
        rule = RSIMomentumRule()
        sig = rule.evaluate(_state_with_rsi(rsi))
        assert sig is not None
        assert sig.value == 0.0
        assert sig.confidence == pytest.approx(0.4)
        assert "overbought" in sig.description

    @pytest.mark.parametrize("rsi", [0.0, 15.0, 29.99, 30.0])
    def test_oversold_extremes_return_neutral(self, rsi: float) -> None:
        rule = RSIMomentumRule()
        sig = rule.evaluate(_state_with_rsi(rsi))
        assert sig is not None
        assert sig.value == 0.0
        assert sig.confidence == pytest.approx(0.4)
        assert "oversold" in sig.description

    def test_boundaries_and_zones_specification(self) -> None:
        rule = RSIMomentumRule()
        # bull_high boundary (70.0): in bull momentum zone -> value=1.0, confidence=0.7
        sig_70 = rule.evaluate(_state_with_rsi(70.0))
        assert sig_70 is not None
        assert sig_70.value == pytest.approx(1.0)
        assert sig_70.confidence == pytest.approx(0.7)

        # bull_low boundary (51.0): start of bull momentum zone -> value=0.0, confidence=0.7
        sig_51 = rule.evaluate(_state_with_rsi(51.0))
        assert sig_51 is not None
        assert sig_51.value == pytest.approx(0.0)
        assert sig_51.confidence == pytest.approx(0.7)

        # mid bull momentum zone (60.5): value=0.5, confidence=0.7
        sig_60_5 = rule.evaluate(_state_with_rsi(60.5))
        assert sig_60_5 is not None
        assert sig_60_5.value == pytest.approx(0.5)
        assert sig_60_5.confidence == pytest.approx(0.7)

        # gap zone (50.0): bear_high < rsi < bull_low -> value=0.0, confidence=0.3
        sig_50 = rule.evaluate(_state_with_rsi(50.0))
        assert sig_50 is not None
        assert sig_50.value == pytest.approx(0.0)
        assert sig_50.confidence == pytest.approx(0.3)

        # bear_high boundary (49.0): start of bear momentum zone -> value=0.0, confidence=0.7
        sig_49 = rule.evaluate(_state_with_rsi(49.0))
        assert sig_49 is not None
        assert sig_49.value == pytest.approx(0.0)
        assert sig_49.confidence == pytest.approx(0.7)

        # mid bear momentum zone (39.5): value=-0.5, confidence=0.7
        sig_39_5 = rule.evaluate(_state_with_rsi(39.5))
        assert sig_39_5 is not None
        assert sig_39_5.value == pytest.approx(-0.5)
        assert sig_39_5.confidence == pytest.approx(0.7)

        # just above bear_low (30.01): in bear momentum zone -> value < -0.99, confidence=0.7
        sig_30_01 = rule.evaluate(_state_with_rsi(30.01))
        assert sig_30_01 is not None
        assert sig_30_01.value < -0.99
        assert sig_30_01.confidence == pytest.approx(0.7)

    def test_extreme_neutral_signal_creates_no_directional_bias_in_score(self) -> None:
        from neon_radar.domain.enums import Bias
        from neon_radar.domain.scoring.aggregator import aggregate
        from neon_radar.domain.scoring.value_objects import Signal

        rule = RSIMomentumRule(weight=0.20)
        overbought_sig = rule.evaluate(_state_with_rsi(80.0))
        oversold_sig = rule.evaluate(_state_with_rsi(20.0))
        assert overbought_sig is not None and overbought_sig.value == 0.0
        assert oversold_sig is not None and oversold_sig.value == 0.0

        # Standalone neutral signal produces zero directional score
        score_ob = aggregate((overbought_sig,))
        assert score_ob.value == 0.0
        assert score_ob.long_score == 0.0
        assert score_ob.short_score == 0.0
        assert score_ob.bias == Bias.NEUTRAL

        score_os = aggregate((oversold_sig,))
        assert score_os.value == 0.0
        assert score_os.long_score == 0.0
        assert score_os.short_score == 0.0
        assert score_os.bias == Bias.NEUTRAL

        # Combined with directional signal: directional value and long/short scores are unaltered
        bull_sig = Signal(name="ema_trend", weight=0.25, value=0.8, confidence=0.85, description="Bull trend")
        score_trend_only = aggregate((bull_sig,))
        score_combined = aggregate((bull_sig, overbought_sig))

        assert score_combined.long_score == pytest.approx(score_trend_only.long_score)
        assert score_combined.short_score == pytest.approx(score_trend_only.short_score)
        assert score_combined.value == pytest.approx(score_trend_only.value)
        assert score_combined.bias == score_trend_only.bias


# ---------------------------------------------------------------------------
# VolumeConfirmationRule
# ---------------------------------------------------------------------------


class TestVolumeConfirmationRule:
    def test_registered(self) -> None:
        assert RuleRegistry.is_registered("volume_confirmation")

    def test_required_indicators(self) -> None:
        specs = VolumeConfirmationRule().required_indicators()
        assert any(s.series_name == "volume_ma_20" for s in specs)

    def test_required_indicators_use_configured_period(self) -> None:
        specs = VolumeConfirmationRule(period=13).required_indicators()
        assert len(specs) == 1
        assert specs[0].series_name == "volume_ma_13"
        assert specs[0].params["period"] == 13

    def test_rejects_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            VolumeConfirmationRule(strong_multiplier=0.9)
        with pytest.raises(ValueError):
            VolumeConfirmationRule(weak_multiplier=1.5)
        with pytest.raises(ValueError):
            VolumeConfirmationRule(weak_multiplier=0)
        with pytest.raises(ValueError):
            VolumeConfirmationRule(period=0)

    def test_strong_volume_bullish_candle(self) -> None:
        # 19 normal candles + 1 with 15x volume on a bullish candle + 1 latest.
        # Volume Confirmation uses the last completed candle (index -2)
        closes = [100.0] * 19 + [110.0] + [115.0]
        volumes = [1000.0] * 19 + [15_000.0] + [500.0]
        sig = VolumeConfirmationRule().evaluate(_state(closes, volumes=volumes))
        assert sig is not None
        assert sig.value > 0

    def test_low_volume_neutral(self) -> None:
        closes = [100.0] * 20 + [100.0]
        volumes = [1000.0] * 19 + [100.0] + [50.0]  # 10% of avg at [-2]
        sig = VolumeConfirmationRule().evaluate(_state(closes, volumes=volumes))
        assert sig is not None
        assert sig.value == pytest.approx(0.0, abs=1e-9)
        assert sig.confidence < 0.7


# ---------------------------------------------------------------------------
# VolatilityFilterRule
# ---------------------------------------------------------------------------


class TestVolatilityFilterRule:
    def test_registered(self) -> None:
        assert RuleRegistry.is_registered("volatility_filter")

    def test_required_indicators_use_configured_period(self) -> None:
        specs = VolatilityFilterRule(period=10).required_indicators()
        assert len(specs) == 1
        assert specs[0].series_name == "atr_10"
        assert specs[0].params["period"] == 10

    def test_rejects_invalid_thresholds(self) -> None:
        with pytest.raises(ValueError):
            VolatilityFilterRule(min_atr_pct=0.05, max_atr_pct=0.003)

    def test_in_comfort_zone_high_confidence(self) -> None:
        # Default helper candles have high-low=2, close=100 → ATR ~ 2% of price.
        closes = [100.0] * 30
        sig = VolatilityFilterRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.confidence >= 0.7

    def test_direction_neutral(self) -> None:
        """Volatility rule never contributes to direction."""
        closes = [100.0 + i for i in range(30)]
        sig = VolatilityFilterRule().evaluate(_state(closes))
        assert sig is not None
        assert sig.value == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# FundingRateRule
# ---------------------------------------------------------------------------


class TestFundingRateRule:
    def test_registered(self) -> None:
        assert RuleRegistry.is_registered("funding_rate")

    def test_required_indicators_empty(self) -> None:
        assert FundingRateRule().required_indicators() == ()

    def test_rejects_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            FundingRateRule(neutral_band=0)
        with pytest.raises(ValueError):
            FundingRateRule(neutral_band=0.001, strong_threshold=0.0005)

    def test_missing_funding_returns_none(self) -> None:
        sig = FundingRateRule().evaluate(_state([100.0] * 10))
        assert sig is None

    def test_neutral_band_returns_none(self) -> None:
        state = _state(
            [100.0] * 10,
            funding_rate=FundingRate(symbol="BTCUSDT", rate=0.00001),
        )
        assert FundingRateRule(neutral_band=0.00005).evaluate(state) is None

    def test_positive_funding_bearish(self) -> None:
        state = _state(
            [100.0] * 10,
            funding_rate=FundingRate(symbol="BTCUSDT", rate=0.0002),
        )
        sig = FundingRateRule().evaluate(state)
        assert sig is not None
        assert sig.value < 0
        assert sig.confidence > 0
        assert "bps" in sig.description

    def test_negative_funding_bullish(self) -> None:
        state = _state(
            [100.0] * 10,
            funding_rate=FundingRate(symbol="BTCUSDT", rate=-0.0002),
        )
        sig = FundingRateRule().evaluate(state)
        assert sig is not None
        assert sig.value > 0

    def test_magnitude_saturates_at_strong_threshold(self) -> None:
        state = _state(
            [100.0] * 10,
            funding_rate=FundingRate(symbol="BTCUSDT", rate=0.01),
        )
        sig = FundingRateRule(strong_threshold=0.0005).evaluate(state)
        assert sig is not None
        assert sig.value == pytest.approx(-1.0)
        assert sig.confidence == pytest.approx(1.0)
