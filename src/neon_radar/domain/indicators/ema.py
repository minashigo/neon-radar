"""Exponential Moving Average (EMA) indicator.

EMA gives more weight to recent prices. The first valid value is
seeded with SMA of the first ``period`` candles; subsequent values
are computed recursively with ``alpha = 2 / (period + 1)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.indicators._numpy_helpers import ema_array
from neon_radar.domain.indicators.base import (
    Indicator,
    IndicatorKind,
    IndicatorRegistry,
    IndicatorSeries,
)
from neon_radar.domain.indicators.sma import _build_snapshots, _closes_array

if TYPE_CHECKING:
    import numpy as np

    from neon_radar.domain.models import KlineSeries


@IndicatorRegistry.register("ema")
class EMA(Indicator):
    """Exponential Moving Average of close prices."""

    KIND = IndicatorKind.OVERLAY

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
        closes = _closes_array(series)
        values = ema_array(closes, self.period)
        snapshots = _build_snapshots(series, [("ema", values)])
        return IndicatorSeries(name=name or self.NAME, kind=self.KIND, snapshots=snapshots)


def ema_difference(
    series: KlineSeries,
    *,
    fast_period: int,
    slow_period: int,
) -> np.ndarray:
    """EMA(fast) - EMA(slow). Used by :class:`MACD` and crossover rules."""
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
        )
    closes = _closes_array(series)
    return ema_array(closes, fast_period) - ema_array(closes, slow_period)
