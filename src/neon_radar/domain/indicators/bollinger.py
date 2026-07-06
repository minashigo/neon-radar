"""Bollinger Bands — volatility envelope around a moving average.

Three outputs:

* **upper** — middle + ``std_multiplier`` x rolling std
* **middle** — SMA(close, period)
* **lower** — middle - ``std_multiplier`` x rolling std

Default ``(20, 2.0)`` is John Bollinger's original recommendation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.indicators._numpy_helpers import rolling_std, sma_array
from neon_radar.domain.indicators.base import (
    Indicator,
    IndicatorKind,
    IndicatorRegistry,
    IndicatorSeries,
)
from neon_radar.domain.indicators.sma import _build_snapshots, _closes_array

if TYPE_CHECKING:
    from neon_radar.domain.models import KlineSeries


@IndicatorRegistry.register("bollinger")
class BollingerBands(Indicator):
    """Bollinger Bands with three sub-outputs: upper, middle, lower."""

    KIND = IndicatorKind.OVERLAY

    def __init__(
        self,
        *,
        period: int = 20,
        std_multiplier: float = 2.0,
    ) -> None:
        if period < 1:
            raise ValueError(f"period must be positive, got {period}")
        if std_multiplier <= 0:
            raise ValueError(f"std_multiplier must be positive, got {std_multiplier}")
        self.period = period
        self.std_multiplier = std_multiplier

    def compute(
        self,
        series: KlineSeries,
        *,
        name: str | None = None,
    ) -> IndicatorSeries:
        closes = _closes_array(series)
        middle = sma_array(closes, self.period)
        std = rolling_std(closes, self.period)
        upper = middle + self.std_multiplier * std
        lower = middle - self.std_multiplier * std
        snapshots = _build_snapshots(
            series,
            [("upper", upper), ("middle", middle), ("lower", lower)],
        )
        return IndicatorSeries(name=name or self.NAME, kind=self.KIND, snapshots=snapshots)
