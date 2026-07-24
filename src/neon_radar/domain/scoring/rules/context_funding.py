"""Funding Rate context rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("funding_trend")
class FundingTrendRule(FactorRule):
    """Analyzes the trend of the funding rate over a historical window."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.15,
        description: str | None = None,
        window_size: int = 5,
        trend_threshold: float = 0.0001,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        self.window_size = window_size
        self.trend_threshold = trend_threshold

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="funding_trend",
            display_name="Funding Trend",
            summary="Analyzes the trend of funding rates over time to detect crowded positioning",
            default_params={
                "window_size": 5,
                "trend_threshold": 0.0001,
            },
        )

    def required_indicators(self) -> tuple:
        return ()

    def evaluate(self, state: MarketState) -> Signal | None:
        hmc = state.historical_context
        if not hmc or not hmc.funding_history or hmc.funding_history.is_empty:
            return None

        series = hmc.funding_history.window(self.window_size)
        if len(series) < 2:
            return None

        first_val = series[0].funding_8h_equiv
        last_val = series[-1].funding_8h_equiv
        delta = last_val - first_val

        if abs(delta) < self.trend_threshold:
            return None

        # Positive trend (funding increasing) -> longs are crowded -> bearish signal
        direction = -1.0 if delta > 0 else 1.0
        magnitude = min(1.0, abs(delta) / (self.trend_threshold * 3))

        return Signal(
            name=self.name,
            weight=self.weight,
            value=direction * magnitude,
            confidence=magnitude,
            description=f"Funding rate shifted by {delta*10000:+.2f} bps over {len(series)} periods",
            evidence=(
                EvidenceItem("delta_bps", f"{delta*10000:.2f}"),
                EvidenceItem("first_val", f"{first_val:.6f}"),
                EvidenceItem("last_val", f"{last_val:.6f}"),
            ),
        )


@RuleRegistry.register("funding_extreme")
class FundingExtremeRule(FactorRule):
    """Detects extreme funding rates representing over-leveraged markets."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.20,
        description: str | None = None,
        extreme_threshold: float = 0.0005,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        self.extreme_threshold = extreme_threshold

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="funding_extreme",
            display_name="Funding Extreme",
            summary="Contrarian signal when funding hits extreme levels",
            default_params={
                "extreme_threshold": 0.0005,
            },
        )

    def required_indicators(self) -> tuple:
        return ()

    def evaluate(self, state: MarketState) -> Signal | None:
        hmc = state.historical_context
        if not hmc or not hmc.funding_history or hmc.funding_history.is_empty:
            return None

        latest = hmc.funding_history.latest()
        if not latest:
            return None

        rate = latest.funding_8h_equiv
        if abs(rate) < self.extreme_threshold:
            return None

        direction = -1.0 if rate > 0 else 1.0
        magnitude = min(1.0, abs(rate) / (self.extreme_threshold * 2))

        return Signal(
            name=self.name,
            weight=self.weight,
            value=direction * magnitude,
            confidence=min(1.0, magnitude * 1.2),
            description=f"Extreme funding rate ({rate*10000:+.2f} bps)",
            evidence=(
                EvidenceItem("funding_8h", f"{rate:.6f}"),
                EvidenceItem("annualized_apr", f"{latest.annualized_apr*100:.1f}%"),
            ),
        )
