"""Tests for the VolumeMA (Volume Moving Average) indicator."""

from __future__ import annotations

import math

import pytest

from neon_radar.domain.indicators import IndicatorRegistry, VolumeMA
from tests.conftest import make_series


class TestVolumeMA:
    def test_registered(self) -> None:
        assert IndicatorRegistry.is_registered("volume_ma")
        assert IndicatorRegistry.get("volume_ma") is VolumeMA

    def test_kind_is_meta(self) -> None:
        assert VolumeMA.KIND.value == "meta"

    def test_rejects_invalid_period(self) -> None:
        with pytest.raises(ValueError):
            VolumeMA(period=0)

    def test_warmup(self) -> None:
        series = make_series([100.0] * 30)
        out = VolumeMA(period=10).compute(series)
        for i in range(9):
            assert math.isnan(out.snapshots[i].get("volume_ma"))

    def test_volume_ma_known_values(self) -> None:
        """Volumes in our helper are base + index = 1000 + i.
        VolumeMA(3) at index 2 = mean(1000, 1001, 1002) = 1001.
        VolumeMA(3) at index 9 = mean(1007, 1008, 1009) = 1008.
        """
        series = make_series([100.0] * 10)
        out = VolumeMA(period=3).compute(series)
        assert out.snapshots[2].get("volume_ma") == pytest.approx(1001.0)
        assert out.snapshots[3].get("volume_ma") == pytest.approx(1002.0)
        assert out.snapshots[9].get("volume_ma") == pytest.approx(1008.0)

    def test_short_series(self) -> None:
        series = make_series([100.0] * 3)
        out = VolumeMA(period=10).compute(series)
        for snap in out.snapshots:
            assert math.isnan(snap.get("volume_ma"))
