"""Application layer.

Use-case orchestration services that combine ``infrastructure`` and
``domain`` to fulfil the application's needs.

The presentation layer talks to services here, never directly to
``infrastructure``. This keeps the UI free of API-specific code.
"""

from neon_radar.application.services.analysis import (
    analyze_series,
    collect_indicator_specs,
)
from neon_radar.application.services.indicator_pipeline import (
    IndicatorSpec,
    available_indicators,
    compute_indicators,
)
from neon_radar.application.services.market_data import MarketDataService

__all__ = [
    "IndicatorSpec",
    "MarketDataService",
    "analyze_series",
    "available_indicators",
    "collect_indicator_specs",
    "compute_indicators",
]
