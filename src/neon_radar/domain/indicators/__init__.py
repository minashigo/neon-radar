"""Technical indicators.

This package contains:

* :mod:`neon_radar.domain.indicators.base` — abstract ``Indicator``
  class + ``IndicatorRegistry``
* Built-in indicators (EMA, SMA, RSI, MACD, BollingerBands, ATR,
  VolumeMA) — registered automatically on import of this package.

Adding a new indicator is a **single new file**:

1. Create ``domain/indicators/<name>.py``.
2. Subclass :class:`Indicator`.
3. Decorate with :meth:`IndicatorRegistry.register`.
4. Add the import in this file so it auto-registers.

No other file needs to change. The pipeline picks it up
automatically.
"""

# Base abstractions (always available)
# Built-in indicators — imported for side-effect registration.
# Each decorator call below populates IndicatorRegistry._items.
from neon_radar.domain.indicators.adx import ADXIndicator
from neon_radar.domain.indicators.atr import ATR
from neon_radar.domain.indicators.base import (
    Indicator,
    IndicatorKind,
    IndicatorRegistry,
    IndicatorSeries,
    IndicatorSnapshot,
    IndicatorValue,
)
from neon_radar.domain.indicators.bollinger import BollingerBands
from neon_radar.domain.indicators.ema import EMA, ema_difference
from neon_radar.domain.indicators.macd import MACD
from neon_radar.domain.indicators.roc import ROC
from neon_radar.domain.indicators.rsi import RSI
from neon_radar.domain.indicators.sma import SMA
from neon_radar.domain.indicators.volume_ma import VolumeMA

__all__ = [
    "ATR",
    "EMA",
    "MACD",
    "ROC",
    "RSI",
    "SMA",
    "BollingerBands",
    "Indicator",
    "IndicatorKind",
    "IndicatorRegistry",
    "IndicatorSeries",
    "IndicatorSnapshot",
    "IndicatorValue",
    "VolumeMA",
    "ema_difference",
]
