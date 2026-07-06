"""Tests for the RSI (Relative Strength Index) indicator."""

from __future__ import annotations

import math

import pytest

from neon_radar.domain.indicators import RSI, IndicatorRegistry
from tests.conftest import make_series


class TestRSI:
    def test_registered(self) -> None:
        assert IndicatorRegistry.is_registered("rsi")
        assert IndicatorRegistry.get("rsi") is RSI

    def test_kind_is_oscillator(self) -> None:
        assert RSI.KIND.value == "oscillator"

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            RSI(period=0)

    def test_constant_series_returns_50(self) -> None:
        """No gains, no losses -> RS undefined -> RSI conventionally 50."""
        closes = [100.0] * 30
        series = make_series(closes)
        out = RSI(period=14).compute(series)
        # All RSI values after warm-up should be exactly 50.
        for snap in out.snapshots[14:]:
            assert snap.get("rsi") == pytest.approx(50.0)

    def test_strict_uptrend_high_rsi(self) -> None:
        """Monotonically rising closes -> RSI near 100."""
        closes = [100.0 + i for i in range(30)]
        series = make_series(closes)
        out = RSI(period=14).compute(series)
        # The last RSI must be very high (each gain > 0, no losses).
        assert out.snapshots[-1].get("rsi") > 90

    def test_strict_downtrend_low_rsi(self) -> None:
        closes = [200.0 - i for i in range(30)]
        series = make_series(closes)
        out = RSI(period=14).compute(series)
        assert out.snapshots[-1].get("rsi") < 10

    def test_warmup_period_is_period(self) -> None:
        """RSI needs period deltas -> first `period` outputs are NaN."""
        series = make_series([100.0 + i for i in range(30)])
        out = RSI(period=14).compute(series)
        # First 14 outputs should be NaN (need 14 deltas + 1 close).
        for i in range(14):
            assert math.isnan(out.snapshots[i].get("rsi"))
        # Output 14 onward is valid.
        assert not math.isnan(out.snapshots[14].get("rsi"))

    def test_rsi_bounded(self) -> None:
        """RSI must always be in [0, 100] when defined."""
        closes = [100.0 + (i % 7) - 3 for i in range(50)]
        series = make_series(closes)
        out = RSI(period=14).compute(series)
        for snap in out.snapshots[14:]:
            v = snap.get("rsi")
            assert 0.0 <= v <= 100.0
