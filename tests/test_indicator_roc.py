"""Tests for the ROC (Rate of Change) indicator."""

from __future__ import annotations

import math

import pytest

from neon_radar.domain.indicators import ROC, IndicatorRegistry
from tests.conftest import make_series


class TestROC:
    def test_registered(self) -> None:
        assert IndicatorRegistry.is_registered("roc")
        assert IndicatorRegistry.get("roc") is ROC

    def test_kind_is_oscillator(self) -> None:
        assert ROC.KIND.value == "oscillator"

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            ROC(period=0)
        with pytest.raises(ValueError):
            ROC(period=-5)

    def test_period_stored(self) -> None:
        assert ROC(period=14).period == 14
        assert ROC(period=5).period == 5

    def test_warmup_is_nan(self) -> None:
        """For period=3 the first 3 outputs must be NaN."""
        series = make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = ROC(period=3).compute(series)
        assert math.isnan(out.snapshots[0].get("roc"))
        assert math.isnan(out.snapshots[1].get("roc"))
        assert math.isnan(out.snapshots[2].get("roc"))
        assert not math.isnan(out.snapshots[3].get("roc"))

    def test_known_values(self) -> None:
        """ROC(2) on [10, 10, 15, 12].
        i=0 (10): NaN
        i=1 (10): NaN
        i=2 (15): (15-10)/10*100 = 50.0
        i=3 (12): (12-10)/10*100 = 20.0
        """
        series = make_series([10.0, 10.0, 15.0, 12.0])
        out = ROC(period=2).compute(series)
        vals = [s.get("roc") for s in out.snapshots]
        assert math.isnan(vals[0])
        assert math.isnan(vals[1])
        assert vals[2] == pytest.approx(50.0)
        assert vals[3] == pytest.approx(20.0)

    def test_short_series_all_nan(self) -> None:
        series = make_series([1.0, 2.0, 3.0])
        out = ROC(period=3).compute(series)
        for snap in out.snapshots:
            assert math.isnan(snap.get("roc"))

    def test_constant_series(self) -> None:
        """ROC on a flat series is 0.0."""
        series = make_series([100.0] * 5)
        out = ROC(period=2).compute(series)
        for snap in out.snapshots[2:]:
            assert snap.get("roc") == pytest.approx(0.0)

    def test_division_by_zero(self) -> None:
        """ROC handles past value == 0 smoothly (inf)."""
        from neon_radar.config.models import TimeFrame
        from neon_radar.domain.models import OHLCV, KlineSeries, Symbol

        candles = [
            OHLCV(open_time=0, open=0.0, high=0.0, low=0.0, close=0.0, volume=0.0),
            OHLCV(open_time=1000, open=0.0, high=0.0, low=0.0, close=0.0, volume=0.0),
            OHLCV(open_time=2000, open=0.0, high=10.0, low=0.0, close=10.0, volume=0.0),
        ]
        series = KlineSeries(
            symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.D1, candles=tuple(candles)
        )
        out = ROC(period=2).compute(series)
        assert math.isinf(out.snapshots[2].get("roc"))
        assert out.snapshots[2].get("roc") > 0

    def test_aligned_with_input(self) -> None:
        """Output length equals input length; timestamps match."""
        series = make_series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = ROC(period=2).compute(series)
        assert len(out) == len(series)
        for s_snap, candle in zip(out.snapshots, series, strict=True):
            assert s_snap.timestamp == candle.open_time
