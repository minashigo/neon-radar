"""Volume Moving Average — average trading volume over a window.

Used by volume-confirmation rules: a breakout above resistance is more
credible if it occurs on volume above its moving average.

Like :class:`ATR`, this is META — consumed by scoring rules rather
than rendered on a chart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neon_radar.domain.indicators._numpy_helpers import sma_array
from neon_radar.domain.indicators.base import (
    Indicator,
    IndicatorKind,
    IndicatorRegistry,
    IndicatorSeries,
)
from neon_radar.domain.indicators.sma import _build_snapshots

if TYPE_CHECKING:
    from neon_radar.domain.models import KlineSeries


@IndicatorRegistry.register("volume_ma")
class VolumeMA(Indicator):
    """Simple Moving Average of trading volume."""

    KIND = IndicatorKind.META

    def __init__(self, period: int = 20) -> None:
        if period < 1:
            raise ValueError(f"period must be positive, got {period}")
        self.period = period

    def compute(
        self,
        series: KlineSeries,
        *,
        name: str | None = None,
    ) -> IndicatorSeries:
        volumes = np.fromiter((c.volume for c in series), dtype=float, count=len(series))
        values = sma_array(volumes, self.period)
        snapshots = _build_snapshots(series, [("volume_ma", values)])
        return IndicatorSeries(name=name or self.NAME, kind=self.KIND, snapshots=snapshots)
