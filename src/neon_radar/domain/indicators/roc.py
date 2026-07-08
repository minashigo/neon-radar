"""Rate of Change (ROC) indicator.

Measures the percentage change in price between the current period and
a past period.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.indicators._numpy_helpers import roc_array
from neon_radar.domain.indicators.base import (
    Indicator,
    IndicatorKind,
    IndicatorRegistry,
    IndicatorSeries,
)
from neon_radar.domain.indicators.sma import _build_snapshots, _closes_array

if TYPE_CHECKING:
    from neon_radar.domain.models import KlineSeries


@IndicatorRegistry.register("roc")
class ROC(Indicator):
    """Rate of Change (ROC) of close prices."""

    KIND = IndicatorKind.OSCILLATOR

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError(f"period must be positive, got {period}")
        self.period = period

    def compute(
        self,
        series: KlineSeries,
        *,
        name: str | None = None,
    ) -> IndicatorSeries:
        closes = _closes_array(series)
        values = roc_array(closes, self.period)
        snapshots = _build_snapshots(series, [("roc", values)])
        return IndicatorSeries(name=name or self.NAME, kind=self.KIND, snapshots=snapshots)
