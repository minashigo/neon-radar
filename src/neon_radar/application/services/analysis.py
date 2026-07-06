"""Analysis pipeline: indicators -> rules -> aggregate score.

This is the small application-level use case shared by CLI, UI and
backtests. It keeps the "first complete analysis cycle" in one place:

1. collect every indicator required by the active rules;
2. compute those indicators over the primary candle series;
3. assemble a :class:`MarketState`;
4. evaluate all rules and aggregate their signals into a final score.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from neon_radar.application.services.indicator_pipeline import (
    IndicatorSpec,
    compute_indicators,
)
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.scoring import AnalysisResult, RuleBasedEngine

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neon_radar.domain.funding import FundingRate, OpenInterest
    from neon_radar.domain.models import KlineSeries, TickerStats
    from neon_radar.domain.scoring.factor_rule import FactorRule


def collect_indicator_specs(rules: Iterable[FactorRule]) -> tuple[IndicatorSpec, ...]:
    """Return unique indicator specs required by ``rules``.

    Uniqueness is keyed by ``IndicatorSpec.series_name``. This matches the
    lookup contract used by scoring rules (e.g. ``ema_20`` vs ``ema_50``).
    If two rules ask for the same series name, the later duplicate is
    ignored because computing it twice would produce the same state entry.
    """
    spec_map: dict[str, IndicatorSpec] = {}
    for rule in rules:
        for spec in rule.required_indicators():
            spec_map.setdefault(spec.series_name, spec)
    return tuple(spec_map.values())


def analyze_series(
    series: KlineSeries,
    rules: Iterable[FactorRule],
    *,
    min_confidence: float = 0.0,
    timestamp: int | None = None,
    higher_tf_series: KlineSeries | None = None,
    ticker: TickerStats | None = None,
    funding_rate: FundingRate | None = None,
    open_interest: OpenInterest | None = None,
) -> AnalysisResult:
    """Run the complete analysis cycle for one candle series."""
    rules_tuple = tuple(rules)
    specs = collect_indicator_specs(rules_tuple)
    indicators = compute_indicators(series, specs)

    state = MarketState(
        symbol=series.symbol,
        timestamp=timestamp if timestamp is not None else _default_timestamp(series),
        primary_series=series,
        higher_tf_series=higher_tf_series,
        indicator_series=tuple(indicators),
        ticker=ticker,
        funding_rate=funding_rate,
        open_interest=open_interest,
    )
    engine = RuleBasedEngine(rules=rules_tuple, min_confidence=min_confidence)
    return engine.evaluate(state)


def _default_timestamp(series: KlineSeries) -> int:
    latest = series.latest()
    if latest is not None:
        return latest.open_time
    return int(time.time() * 1000)
