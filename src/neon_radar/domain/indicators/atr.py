"""Average True Range (ATR) — Wilder's volatility measure.

True Range for each candle is the maximum of:

* ``high - low``
* ``|high - prev_close|``
* ``|low - prev_close|``

ATR smooths True Range with Wilder's method (the same as RSI).
ATR is META because it is rarely drawn on a chart directly — most
often it is used by other indicators (e.g. position-sizing rules,
Keltner channels, trailing stops).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neon_radar.domain.indicators._numpy_helpers import wilder_array
from neon_radar.domain.indicators.base import (
    Indicator,
    IndicatorKind,
    IndicatorRegistry,
    IndicatorSeries,
)
from neon_radar.domain.indicators.sma import _build_snapshots

if TYPE_CHECKING:
    from neon_radar.domain.models import KlineSeries


@IndicatorRegistry.register("atr")
class ATR(Indicator):
    """Average True Range with Wilder's smoothing."""

    KIND = IndicatorKind.META

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
        high = np.fromiter((c.high for c in series), dtype=float, count=len(series))
        low = np.fromiter((c.low for c in series), dtype=float, count=len(series))
        close = np.fromiter((c.close for c in series), dtype=float, count=len(series))
        atr = _atr_array(high, low, close, self.period)
        snapshots = _build_snapshots(series, [("atr", atr)])
        return IndicatorSeries(name=name or self.NAME, kind=self.KIND, snapshots=snapshots)


def _atr_array(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    n = closes.size
    out = np.full(n, np.nan, dtype=float)
    if n < 1 or period < 1:
        return out
    true_range = np.empty(n, dtype=float)
    # First candle has no previous close; TR = high - low.
    true_range[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        true_range[i] = max(hl, hc, lc)
    out = wilder_array(true_range, period)
    return out
