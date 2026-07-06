"""RSI momentum rule.

Maps RSI levels to a directional signal:

* **RSI > bull_high** (default 70): ``0`` (overbought - neutral on
  direction; the trend may exhaust). Lower confidence.
* **bull_low < RSI <= bull_high** (default 50-70): ``+1`` (bullish
  momentum zone).
* **bear_low < RSI <= bear_high** (default 30-50): ``-1`` (bearish
  momentum zone).
* **RSI <= bear_low** (default 30): ``0`` (oversold - may bounce).

The "neutral on extremes" behaviour prevents the rule from chasing
overbought moves, which historically underperform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("rsi_momentum")
class RSIMomentumRule(FactorRule):
    """Map RSI levels to a directional bias."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.25,
        description: str | None = None,
        period: int = 14,
        bull_low: float = 51.0,
        bull_high: float = 70.0,
        bear_low: float = 30.0,
        bear_high: float = 49.0,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        if not 0 < bear_low < bear_high < bull_low < bull_high <= 100:
            raise ValueError(
                f"Invalid RSI thresholds: bear_low({bear_low}) < bear_high({bear_high}) "
                f"< bull_low({bull_low}) < bull_high({bull_high}) required"
            )
        self.period = period
        self.bull_low = bull_low
        self.bull_high = bull_high
        self.bear_low = bear_low
        self.bear_high = bear_high

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="rsi_momentum",
            display_name="RSI Momentum",
            summary="Bullish momentum when RSI is in [51, 70]",
            default_params={
                "period": 14,
                "bull_low": 51.0,
                "bull_high": 70.0,
                "bear_low": 30.0,
                "bear_high": 49.0,
            },
        )

    def required_indicators(self) -> tuple:
        # Lazy import to avoid circular dependency.
        from neon_radar.application.services.indicator_pipeline import IndicatorSpec

        return (
            IndicatorSpec(
                name="rsi",
                params={"period": self.period},
                tag=str(self.period),
            ),
        )

    def evaluate(self, state: MarketState) -> Signal | None:
        rsi = state.get_indicator_value(f"rsi_{self.period}")
        if rsi is None or rsi != rsi:  # NaN guard
            return None

        if rsi > self.bull_high:
            # Overbought - neutral on direction.
            return Signal(
                name=self.name,
                weight=self.weight,
                value=0.0,
                confidence=0.4,
                description=f"RSI overbought ({rsi:.1f} > {self.bull_high:.0f})",
                evidence=(EvidenceItem("rsi", f"{rsi:.2f}"),),
            )
        if rsi < self.bear_low:
            return Signal(
                name=self.name,
                weight=self.weight,
                value=0.0,
                confidence=0.4,
                description=f"RSI oversold ({rsi:.1f} < {self.bear_low:.0f})",
                evidence=(EvidenceItem("rsi", f"{rsi:.2f}"),),
            )

        if rsi >= self.bull_low:
            # Bull zone: stronger signal as RSI climbs toward overbought.
            magnitude = min(1.0, (rsi - self.bull_low) / (self.bull_high - self.bull_low))
            confidence = 0.7
            arrow = "↑"
            return Signal(
                name=self.name,
                weight=self.weight,
                value=magnitude,
                confidence=confidence,
                description=f"RSI bullish zone {arrow} ({rsi:.1f})",
                evidence=(EvidenceItem("rsi", f"{rsi:.2f}"),),
            )

        if rsi <= self.bear_high:
            # Bear zone: stronger signal as RSI falls toward oversold.
            magnitude = -min(
                1.0, (self.bear_high - rsi) / (self.bear_high - self.bear_low)
            )
            confidence = 0.7
            return Signal(
                name=self.name,
                weight=self.weight,
                value=magnitude,
                confidence=confidence,
                description=f"RSI bearish zone ({rsi:.1f})",
                evidence=(EvidenceItem("rsi", f"{rsi:.2f}"),),
            )

        # Gap zone: bear_high < rsi < bull_low - neutral, low confidence.
        return Signal(
            name=self.name,
            weight=self.weight,
            value=0.0,
            confidence=0.3,
            description=f"RSI neutral ({rsi:.1f})",
            evidence=(EvidenceItem("rsi", f"{rsi:.2f}"),),
        )
