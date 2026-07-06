"""Tests for the SMA (Simple Moving Average) indicator."""

from __future__ import annotations

import math

import pytest

from neon_radar.domain.indicators import SMA, IndicatorRegistry
from tests.conftest import make_series


class TestSMA:
    def test_registered(self) -> None:
        assert IndicatorRegistry.is_registered("sma")
        assert IndicatorRegistry.get("sma") is SMA

    def test_kind_is_overlay(self) -> None:
        assert SMA.KIND.value == "overlay"

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            SMA(period=0)
        with pytest.raises(ValueError):
            SMA(period=-5)

    def test_period_stored(self) -> None:
        assert SMA(period=20).period == 20
        assert SMA(period=5).period == 5

    def test_warmup_is_nan(self) -> None:
        """For period=3 the first 2 outputs must be NaN."""
        series = make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = SMA(period=3).compute(series)
        assert math.isnan(out.snapshots[0].get("sma"))
        assert math.isnan(out.snapshots[1].get("sma"))
        assert not math.isnan(out.snapshots[2].get("sma"))

    def test_known_values(self) -> None:
        """SMA(3) on [1,2,3,4,5]: [NaN, NaN, 2, 3, 4]."""
        series = make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = SMA(period=3).compute(series)
        vals = [s.get("sma") for s in out.snapshots]
        assert vals[0] != vals[0]  # NaN
        assert vals[1] != vals[1]  # NaN
        assert vals[2] == pytest.approx(2.0)
        assert vals[3] == pytest.approx(3.0)
        assert vals[4] == pytest.approx(4.0)

    def test_short_series_all_nan(self) -> None:
        series = make_series([1.0, 2.0])
        out = SMA(period=5).compute(series)
        for snap in out.snapshots:
            assert math.isnan(snap.get("sma"))

    def test_constant_series(self) -> None:
        """SMA on a flat series equals the constant (after warm-up)."""
        series = make_series([100.0] * 10)
        out = SMA(period=3).compute(series)
        for snap in out.snapshots[2:]:
            assert snap.get("sma") == pytest.approx(100.0)

    def test_aligned_with_input(self) -> None:
        """Output length equals input length; timestamps match."""
        series = make_series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = SMA(period=2).compute(series)
        assert len(out) == len(series)
        for s_snap, candle in zip(out.snapshots, series, strict=True):
            assert s_snap.timestamp == candle.open_time
