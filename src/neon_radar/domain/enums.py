"""Domain enumerations.

These types are deliberately kept separate from ``models.py`` so they can
be imported without paying the cost of the (heavier) data classes.

We use :class:`enum.StrEnum` (Python 3.11+) so each member ``is`` a plain
string. That means ``Side.LONG == "Long"`` is true without any explicit
cast, which keeps UI bindings and Pydantic interop trivial.
"""

from __future__ import annotations

from enum import StrEnum


class Side(StrEnum):
    """Trade direction considered by the analysis.

    Used by the future scoring engine. The string value is what gets
    displayed in the UI, so it is human-readable rather than numeric.
    """

    LONG = "Long"
    SHORT = "Short"
    NEUTRAL = "Neutral"


class Bias(StrEnum):
    """Overall directional bias computed by the analysis engine.

    Distinct from :class:`Side` because bias is the *aggregated* opinion
    of multiple indicators, while ``Side`` is what a single rule
    suggests.
    """

    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


class TrendDirection(StrEnum):
    """Trend classification, used by filters and chart overlays."""

    UP = "Up"
    DOWN = "Down"
    FLAT = "Flat"


class MarketRegime(StrEnum):
    """Market regime classification (placeholder for future use)."""

    TRENDING = "Trending"
    RANGING = "Ranging"
    VOLATILE = "Volatile"
    UNKNOWN = "Unknown"
