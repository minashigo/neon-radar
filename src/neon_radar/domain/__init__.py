"""Domain layer.

Pure business models, enums and exceptions. **No I/O, no Qt, no logging
configuration here.** The domain must be importable from any layer without
pulling in heavy dependencies.

This is the heart of the application: the rules of trading analysis live
here, and they must not depend on how data is fetched or how the UI
renders.

Sub-modules
-----------
* :mod:`neon_radar.domain.models` — candles, series, symbols, tickers
* :mod:`neon_radar.domain.funding` — funding rate, open interest
* :mod:`neon_radar.domain.indicators` — indicator abstractions
* :mod:`neon_radar.domain.market_state` — assembled market view
* :mod:`neon_radar.domain.scoring` — score, signal, analysis result
* :mod:`neon_radar.domain.enums` — shared enums
* :mod:`neon_radar.domain.exceptions` — error hierarchy
"""

from neon_radar.domain.enums import Bias, MarketRegime, Side, TrendDirection
from neon_radar.domain.exceptions import (
    ApiError,
    ConfigError,
    DataError,
    DataValidationError,
    IndicatorError,
    NeonRadarError,
    NetworkError,
    ParseError,
    RateLimitError,
    ServerError,
)
from neon_radar.domain.funding import FundingRate, OpenInterest
from neon_radar.domain.indicators import (
    Indicator,
    IndicatorKind,
    IndicatorRegistry,
    IndicatorSeries,
    IndicatorSnapshot,
    IndicatorValue,
)
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import (
    OHLCV,
    KlineSeries,
    Symbol,
    TickerStats,
)
from neon_radar.domain.scoring import (
    AnalysisResult,
    EvidenceItem,
    FactorRule,
    Score,
    Signal,
)

__all__ = [
    "OHLCV",
    "AnalysisResult",
    "ApiError",
    "Bias",
    "ConfigError",
    "DataError",
    "DataValidationError",
    "EvidenceItem",
    "FactorRule",
    "FundingRate",
    "Indicator",
    "IndicatorError",
    "IndicatorKind",
    "IndicatorRegistry",
    "IndicatorSeries",
    "IndicatorSnapshot",
    "IndicatorValue",
    "KlineSeries",
    "MarketRegime",
    "MarketState",
    "NeonRadarError",
    "NetworkError",
    "OpenInterest",
    "ParseError",
    "RateLimitError",
    "Score",
    "ServerError",
    "Side",
    "Signal",
    "Symbol",
    "TickerStats",
    "TrendDirection",
]
