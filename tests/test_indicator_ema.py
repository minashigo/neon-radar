"""Tests for the EMA (Exponential Moving Average) indicator."""

from __future__ import annotations

import math

import pytest

from neon_radar.domain.indicators import EMA, SMA, IndicatorRegistry
from tests.conftest import make_series


class TestEMA:
    def test_registered(self) -> None:
        assert IndicatorRegistry.is_registered("ema")
        assert IndicatorRegistry.get("ema") is EMA

    def test_kind_is_overlay(self) -> None:
        assert EMA.KIND.value == "overlay"

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            EMA(period=0)

    def test_warmup_is_nan(self) -> None:
        series = make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = EMA(period=3).compute(series)
        assert math.isnan(out.snapshots[0].get("ema"))
        assert math.isnan(out.snapshots[1].get("ema"))
        assert not math.isnan(out.snapshots[2].get("ema"))

    def test_seed_is_sma(self) -> None:
        """The first valid EMA value must equal SMA of first `period` closes."""
        # Closes: 1, 2, 3, 4, 5. SMA(3) of first 3 = 2.
        series = make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = EMA(period=3).compute(series)
        assert out.snapshots[2].get("ema") == pytest.approx(2.0)

    def test_recursive_step(self) -> None:
        """EMA(3) on [1,2,3,4,5] after seed: alpha=0.5.
        EMA[3] = 0.5*4 + 0.5*2 = 3
        EMA[4] = 0.5*5 + 0.5*3 = 4
        """
        series = make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = EMA(period=3).compute(series)
        assert out.snapshots[3].get("ema") == pytest.approx(3.0)
        assert out.snapshots[4].get("ema") == pytest.approx(4.0)

    def test_short_series(self) -> None:
        series = make_series([1.0, 2.0])
        out = EMA(period=5).compute(series)
        for snap in out.snapshots:
            assert math.isnan(snap.get("ema"))

    def test_constant_series(self) -> None:
        """EMA on a flat series equals the constant."""
        series = make_series([50.0] * 10)
        out = EMA(period=3).compute(series)
        for snap in out.snapshots[2:]:
            assert snap.get("ema") == pytest.approx(50.0)

    def test_uptrend_rises_faster_than_sma(self) -> None:
        """EMA reacts faster than SMA immediately after a price jump."""
        # Closes: 10, 10, 10, 10, 10, 100, 100, 100, 100, 100.
        # At index 5 (right after the jump), SMA(5) is still
        # ``mean(10, 10, 10, 10, 100) = 28`` while EMA(5) is
        # ``0.333*100 + 0.667*10 = 40``. EMA reacts faster.
        closes = [10.0] * 5 + [100.0] * 5
        series = make_series(closes)
        ema_out = EMA(period=5).compute(series)
        sma_out = SMA(period=5).compute(series)
        # Immediately after the jump, EMA > SMA.
        assert ema_out.snapshots[5].get("ema") > sma_out.snapshots[5].get("sma")
        # The crossover point: by the time enough jumps have happened,
        # SMA catches up and exceeds EMA. Verify EMA < SMA at the end.
        assert ema_out.snapshots[-1].get("ema") < sma_out.snapshots[-1].get("sma")
