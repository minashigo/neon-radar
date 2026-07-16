"""MACD — Moving Average Convergence Divergence.

Three outputs per candle:

* **macd** — ``EMA(fast) - EMA(slow)``
* **signal** — EMA of the MACD line
* **histogram** — ``macd - signal`` (positive = bullish momentum)

Default periods ``(12, 26, 9)`` are the convention introduced by
Gerald Appel in the 1970s.
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


@IndicatorRegistry.register("macd")
class MACD(Indicator):
    """MACD with three sub-outputs: macd, signal, histogram."""

    KIND = IndicatorKind.OSCILLATOR

    def __init__(
        self,
        *,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        if fast_period < 1:
            raise ValueError(f"fast_period must be positive, got {fast_period}")
        if slow_period <= fast_period:
            raise ValueError(
                f"slow_period ({slow_period}) must be greater than fast_period ({fast_period})"
            )
        if signal_period < 1:
            raise ValueError(f"signal_period must be positive, got {signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def compute(
        self,
        series: KlineSeries,
        *,
        name: str | None = None,
    ) -> IndicatorSeries:
        closes = _closes_array(series)
        macd_line, signal, histogram = _macd_arrays(
            closes,
            fast=self.fast_period,
            slow=self.slow_period,
            signal_period=self.signal_period,
        )
        snapshots = _build_snapshots(
            series,
            [("macd", macd_line), ("signal", signal), ("histogram", histogram)],
        )
        return IndicatorSeries(name=name or self.NAME, kind=self.KIND, snapshots=snapshots)


def _macd_arrays(
    closes: np.ndarray,
    *,
    fast: int,
    slow: int,
    signal_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the three MACD arrays from closes."""
    macd_line = ema_array(closes, fast) - ema_array(closes, slow)
    # ``signal`` is EMA of the MACD line. NaN values in macd_line make
    # ema_array return NaN until the EMA has enough valid inputs.
    signal = ema_array(macd_line, signal_period)
    histogram = macd_line - signal
    return macd_line, signal, histogram
