"""Tests for the MACD indicator."""

from __future__ import annotations

import math

import pytest

from neon_radar.domain.indicators import MACD, IndicatorRegistry
from tests.conftest import make_series


class TestMACD:
    def test_registered(self) -> None:
        assert IndicatorRegistry.is_registered("macd")
        assert IndicatorRegistry.get("macd") is MACD

    def test_kind_is_oscillator(self) -> None:
        assert MACD.KIND.value == "oscillator"

    def test_rejects_invalid_periods(self) -> None:
        with pytest.raises(ValueError):
            MACD(fast_period=0)
        with pytest.raises(ValueError):
            MACD(fast_period=10, slow_period=5)  # slow <= fast
        with pytest.raises(ValueError):
            MACD(signal_period=0)

    def test_three_outputs_per_snapshot(self) -> None:
        series = make_series([100.0 + i for i in range(40)])
        out = MACD().compute(series)
        for snap in out.snapshots:
            keys = {v.name for v in snap.values}
            assert keys == {"macd", "signal", "histogram"}

    def test_histogram_equals_macd_minus_signal(self) -> None:
        """histogram = macd - signal (definitionally)."""
        closes = [100.0 + i * 0.5 for i in range(50)]
        series = make_series(closes)
        out = MACD().compute(series)
        for snap in out.snapshots:
            macd = snap.get("macd")
            signal = snap.get("signal")
            hist = snap.get("histogram")
            if not math.isnan(macd) and not math.isnan(signal):
                assert hist == pytest.approx(macd - signal, rel=1e-9)

    def test_constant_series_macd_zero(self) -> None:
        """Flat prices -> fast EMA == slow EMA -> MACD line == 0."""
        closes = [100.0] * 50
        series = make_series(closes)
        out = MACD().compute(series)
        for snap in out.snapshots:
            macd = snap.get("macd")
            if not math.isnan(macd):
                assert macd == pytest.approx(0.0, abs=1e-9)

    def test_warmup_nan(self) -> None:
        """First slow_period-1 outputs are NaN for macd line."""
        series = make_series([100.0 + i for i in range(50)])
        out = MACD().compute(series)
        # slow_period default is 26.
        for i in range(25):
            assert math.isnan(out.snapshots[i].get("macd"))

    def test_aligned_with_input(self) -> None:
        series = make_series([100.0 + i for i in range(40)])
        out = MACD().compute(series)
        assert len(out) == len(series)
        for s_snap, candle in zip(out.snapshots, series, strict=True):
            assert s_snap.timestamp == candle.open_time
