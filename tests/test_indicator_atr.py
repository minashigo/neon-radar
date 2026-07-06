"""Tests for the ATR (Average True Range) indicator."""

from __future__ import annotations

import math

import pytest

from neon_radar.domain.indicators import ATR, IndicatorRegistry
from tests.conftest import make_series


class TestATR:
    def test_registered(self) -> None:
        assert IndicatorRegistry.is_registered("atr")
        assert IndicatorRegistry.get("atr") is ATR

    def test_kind_is_meta(self) -> None:
        assert ATR.KIND.value == "meta"

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            ATR(period=0)

    def test_constant_close_atr_equals_range(self) -> None:
        """If close is constant but high-low = 2, ATR = 2."""
        # We construct a series with constant close but our make_candles
        # forces high=close+1, low=close-1, so range = 2.
        closes = [100.0] * 30
        series = make_series(closes)
        out = ATR(period=14).compute(series)
        for snap in out.snapshots[13:]:
            assert snap.get("atr") == pytest.approx(2.0)

    def test_atr_non_negative(self) -> None:
        closes = [100.0 + (i % 5) - 2 for i in range(40)]
        series = make_series(closes)
        out = ATR(period=14).compute(series)
        for snap in out.snapshots[13:]:
            v = snap.get("atr")
            assert v >= 0

    def test_atr_warmup(self) -> None:
        series = make_series([100.0 + i for i in range(30)])
        out = ATR(period=14).compute(series)
        for i in range(13):
            assert math.isnan(out.snapshots[i].get("atr"))

    def test_single_candle(self) -> None:
        """One candle: no previous close, TR = high - low."""
        series = make_series([100.0])
        out = ATR(period=1).compute(series)
        assert out.snapshots[0].get("atr") == pytest.approx(2.0)
