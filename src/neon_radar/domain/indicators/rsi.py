"""Relative Strength Index (RSI) — Wilder's classic oscillator.

RSI oscillates between 0 and 100:

* **RSI > 70** is conventionally "overbought" (potential short setup).
* **RSI < 30** is conventionally "oversold" (potential long setup).

The first ``period`` values of the close-price deltas are needed
before any RSI can be computed, so the warm-up is ``period`` candles
(not ``period - 1`` as for moving averages).
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
from neon_radar.domain.indicators.sma import _build_snapshots, _closes_array

if TYPE_CHECKING:
    from neon_radar.domain.models import KlineSeries


@IndicatorRegistry.register("rsi")
class RSI(Indicator):
    """Relative Strength Index (Wilder's smoothing)."""

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
        values = _rsi_array(closes, self.period)
        snapshots = _build_snapshots(series, [("rsi", values)])
        return IndicatorSeries(name=name or self.NAME, kind=self.KIND, snapshots=snapshots)


def _rsi_array(closes: np.ndarray, period: int) -> np.ndarray:
    n = closes.size
    out = np.full(n, np.nan, dtype=float)
    # Need at least one delta plus `period` averaged values.
    if n < period + 1 or period < 1:
        return out

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # ``avg_gain`` and ``avg_loss`` have length ``n - 1`` (one per delta).
    avg_gain = wilder_array(gains, period)
    avg_loss = wilder_array(losses, period)

    # Three edge cases handled explicitly so the formula never divides
    # by zero and never produces NaN where it shouldn't:
    #   * both zero       -> RS is undefined; conventional RSI = 50
    #   * loss == 0 only  -> only gains; RSI = 100
    #   * gain == 0 only  -> only losses; RSI = 0
    #   * otherwise       -> standard ``100 - 100 / (1 + RS)``
    rs = np.where(
        (avg_gain == 0) & (avg_loss == 0),
        1.0,  # -> RSI = 50 (see below)
        np.where(avg_loss == 0, np.inf, avg_gain / np.where(avg_loss == 0, 1.0, avg_loss)),
    )
    rsi = np.where(
        np.isinf(rs),
        100.0,
        100.0 - 100.0 / (1.0 + rs),
    )
    # ``deltas[i]`` corresponds to ``closes[i+1] - closes[i]``, so we
    # place the result for delta ``i`` at output index ``i + 1``.
    out[1:] = rsi
    return out
