"""Simple Moving Average (SMA) indicator.

Computes the arithmetic mean of the last ``period`` close prices.
NaN for the first ``period - 1`` candles (warm-up).
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
    IndicatorSnapshot,
    IndicatorValue,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neon_radar.domain.models import OHLCV, KlineSeries


@IndicatorRegistry.register("sma")
class SMA(Indicator):
    """Simple Moving Average of close prices."""

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
        values = sma_array(closes, self.period)
        snapshots = _build_snapshots(series, [("sma", values)])
        return IndicatorSeries(name=name or self.NAME, kind=self.KIND, snapshots=snapshots)


def _closes_array(series: KlineSeries) -> np.ndarray:
    """Extract close prices as a 1-D float array."""
    return np.fromiter((c.close for c in series), dtype=float, count=len(series))


def _build_snapshots(
    series: KlineSeries,
    named_arrays: Sequence[tuple[str, np.ndarray]],
) -> tuple[IndicatorSnapshot, ...]:
    """Build snapshots pairing each candle with one value per named array.

    Any NaN in the input arrays is preserved as ``float('nan')`` so
    downstream code can detect the warm-up period via
    :func:`math.isnan`.
    """
    snapshots: list[IndicatorSnapshot] = []
    for i, candle in enumerate(series):
        values = tuple(IndicatorValue(name, float(arr[i])) for name, arr in named_arrays)
        snapshots.append(IndicatorSnapshot(timestamp=candle.open_time, values=values))
    return tuple(snapshots)


def _ensure_ohlcv(series: KlineSeries) -> tuple[OHLCV, ...]:
    """Helper kept for future indicators that need full OHLCV data."""
    return series.candles
