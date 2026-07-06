"""Tests for the Bollinger Bands indicator."""

from __future__ import annotations

import math

import pytest

from neon_radar.domain.indicators import BollingerBands, IndicatorRegistry
from tests.conftest import make_series


class TestBollingerBands:
    def test_registered(self) -> None:
        assert IndicatorRegistry.is_registered("bollinger")
        assert IndicatorRegistry.get("bollinger") is BollingerBands

    def test_kind_is_overlay(self) -> None:
        assert BollingerBands.KIND.value == "overlay"

    def test_rejects_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            BollingerBands(period=0)
        with pytest.raises(ValueError):
            BollingerBands(std_multiplier=0)
        with pytest.raises(ValueError):
            BollingerBands(std_multiplier=-1)

    def test_three_outputs(self) -> None:
        series = make_series([100.0 + i for i in range(30)])
        out = BollingerBands().compute(series)
        for snap in out.snapshots:
            keys = {v.name for v in snap.values}
            assert keys == {"upper", "middle", "lower"}

    def test_upper_above_middle_above_lower(self) -> None:
        """upper > middle > lower (when std > 0)."""
        closes = [100.0 + (i % 5) for i in range(30)]
        series = make_series(closes)
        out = BollingerBands(period=20, std_multiplier=2.0).compute(series)
        for snap in out.snapshots[19:]:
            upper = snap.get("upper")
            middle = snap.get("middle")
            lower = snap.get("lower")
            if not math.isnan(upper):
                assert upper > middle > lower

    def test_constant_series_collapses(self) -> None:
        """Flat prices -> std = 0 -> upper = middle = lower."""
        closes = [100.0] * 30
        series = make_series(closes)
        out = BollingerBands(period=10, std_multiplier=2.0).compute(series)
        for snap in out.snapshots[9:]:
            assert snap.get("upper") == pytest.approx(100.0)
            assert snap.get("middle") == pytest.approx(100.0)
            assert snap.get("lower") == pytest.approx(100.0)

    def test_warmup(self) -> None:
        series = make_series([100.0 + i for i in range(30)])
        out = BollingerBands(period=20).compute(series)
        for i in range(19):
            assert math.isnan(out.snapshots[i].get("upper"))
            assert math.isnan(out.snapshots[i].get("middle"))
            assert math.isnan(out.snapshots[i].get("lower"))

    def test_std_multiplier_scales_bands(self) -> None:
        """Wider multiplier -> wider bands."""
        closes = [100.0 + (i % 7) for i in range(30)]
        series = make_series(closes)
        out_narrow = BollingerBands(period=20, std_multiplier=1.0).compute(series)
        out_wide = BollingerBands(period=20, std_multiplier=3.0).compute(series)
        # Last snapshot's band width.
        u_n = out_narrow.snapshots[-1].get("upper")
        l_n = out_narrow.snapshots[-1].get("lower")
        u_w = out_wide.snapshots[-1].get("upper")
        l_w = out_wide.snapshots[-1].get("lower")
        assert (u_w - l_w) > (u_n - l_n)
