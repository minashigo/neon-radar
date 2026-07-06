"""Tests for the IndicatorPipeline orchestrator."""

from __future__ import annotations

import math

import pytest

from neon_radar.application.services.indicator_pipeline import (
    IndicatorSpec,
    available_indicators,
    compute_indicators,
)
from neon_radar.domain.indicators import (
    EMA,
)
from tests.conftest import make_series


class TestIndicatorSpec:
    def test_basic(self) -> None:
        spec = IndicatorSpec(name="ema", params={"period": 50})
        assert spec.name == "ema"
        assert spec.params == {"period": 50}

    def test_default_params(self) -> None:
        spec = IndicatorSpec(name="sma")
        assert spec.params == {}

    def test_build_creates_instance(self) -> None:
        spec = IndicatorSpec(name="ema", params={"period": 30})
        instance = spec.build()
        assert isinstance(instance, EMA)
        assert instance.period == 30

    def test_unknown_name_raises(self) -> None:
        spec = IndicatorSpec(name="not_a_real_indicator")
        with pytest.raises(ValueError, match="not registered"):
            spec.build()


class TestComputeIndicators:
    def test_empty_specs_returns_empty(self) -> None:
        series = make_series([100.0] * 30)
        assert compute_indicators(series, []) == []

    def test_single_indicator(self) -> None:
        series = make_series([100.0 + i for i in range(30)])
        results = compute_indicators(series, [IndicatorSpec("sma", {"period": 10})])
        assert len(results) == 1
        assert results[0].name == "sma"
        assert len(results[0]) == len(series)

    def test_multiple_indicators_in_order(self) -> None:
        series = make_series([100.0 + i for i in range(50)])
        specs = [
            IndicatorSpec("ema", {"period": 10}),
            IndicatorSpec("rsi", {"period": 14}),
            IndicatorSpec("macd"),
        ]
        results = compute_indicators(series, specs)
        assert [r.name for r in results] == ["ema", "rsi", "macd"]
        assert all(len(r) == len(series) for r in results)

    def test_specs_can_be_reused(self) -> None:
        """Building the same spec twice produces independent instances."""
        series = make_series([100.0] * 30)
        spec = IndicatorSpec("sma", {"period": 5})
        r1 = compute_indicators(series, [spec])
        r2 = compute_indicators(series, [spec])
        # Different instances, same values.
        assert r1[0].snapshots is not r2[0].snapshots

    def test_all_builtins_compute(self) -> None:
        """Every registered built-in must compute without errors."""
        series = make_series([100.0 + i for i in range(60)])
        specs = [
            IndicatorSpec(name) for name in available_indicators()
        ]
        results = compute_indicators(series, specs)
        assert len(results) == len(available_indicators())


class TestAvailableIndicators:
    def test_returns_sorted(self) -> None:
        names = available_indicators()
        assert names == tuple(sorted(names))

    def test_includes_all_builtins(self) -> None:
        names = set(available_indicators())
        expected = {"atr", "bollinger", "ema", "macd", "rsi", "sma", "volume_ma"}
        assert expected.issubset(names)


class TestIntegrationWithDomain:
    """End-to-end: pipeline result can populate a MarketState."""

    def test_indicator_series_aligns_with_klines(self) -> None:
        from neon_radar.config.models import TimeFrame
        from neon_radar.domain.market_state import MarketState
        from neon_radar.domain.models import Symbol

        series = make_series([100.0 + i for i in range(40)], timeframe=TimeFrame.H4)
        results = compute_indicators(
            series,
            [IndicatorSpec("ema", {"period": 10}), IndicatorSpec("rsi", {"period": 14})],
        )
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=0,
            primary_series=series,
            indicator_series=tuple(results),
        )
        # Latest values via the convenience helper.
        ema_latest = state.get_indicator_value("ema")
        rsi_latest = state.get_indicator_value("rsi")
        assert ema_latest is not None
        assert rsi_latest is not None
        assert not math.isnan(ema_latest)
        assert not math.isnan(rsi_latest)
        # Warm-up: the first EMA snapshot must be NaN.
        ema_first_snap = state.get_indicator("ema").snapshots[0]
        assert math.isnan(ema_first_snap.get("ema"))
