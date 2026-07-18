"""Average Directional Index (ADX) indicator."""

from __future__ import annotations

import numpy as np

from neon_radar.domain.indicators._numpy_helpers import wilder_array
from neon_radar.domain.indicators.base import (
    Indicator,
    IndicatorKind,
    IndicatorRegistry,
    IndicatorSeries,
    IndicatorSnapshot,
    IndicatorValue,
)
from neon_radar.domain.models import KlineSeries


@IndicatorRegistry.register("adx")
class ADXIndicator(Indicator):
    """Average Directional Index (ADX) indicator."""

    NAME = "adx"
    KIND = IndicatorKind.OSCILLATOR

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("ADX period must be >= 1")
        self.period = period

    def compute(self, series: KlineSeries, *, name: str | None = None) -> IndicatorSeries:
        """Compute the ADX values over the entire series."""
        out_name = name or self.NAME
        n = len(series.candles)
        if n == 0:
            return IndicatorSeries(name=out_name, kind=self.KIND, snapshots=())

        high = np.array([c.high for c in series.candles], dtype=float)
        low = np.array([c.low for c in series.candles], dtype=float)
        close = np.array([c.close for c in series.candles], dtype=float)

        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]

        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low

        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        pos_dm[0] = np.nan
        neg_dm[0] = np.nan
        tr[0] = np.nan

        # Smoothed TR, +DM, -DM using Wilder's smoothing
        smoothed_tr = wilder_array(tr, self.period)
        smoothed_pos_dm = wilder_array(pos_dm, self.period)
        smoothed_neg_dm = wilder_array(neg_dm, self.period)

        # Directional Indicators
        with np.errstate(divide="ignore", invalid="ignore"):
            pos_di = 100.0 * smoothed_pos_dm / smoothed_tr
            neg_di = 100.0 * smoothed_neg_dm / smoothed_tr

            # Directional Index
            dx = 100.0 * np.abs(pos_di - neg_di) / (pos_di + neg_di)

        # ADX is Wilder's smoothing of DX
        adx = wilder_array(dx, self.period)

        out = []
        for i in range(n):
            ts = series.candles[i].open_time
            if np.isnan(adx[i]):
                out.append(IndicatorSnapshot(ts, ()))
            else:
                out.append(
                    IndicatorSnapshot(
                        ts, (IndicatorValue("adx", float(adx[i])),)
                    )
                )
        return IndicatorSeries(name=out_name, kind=self.KIND, snapshots=tuple(out))
