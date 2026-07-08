"""SMA trend rule.

Compares a fast SMA with a slow SMA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("sma_trend")
class SMATrendRule(FactorRule):
    """Bullish if fast SMA is above slow SMA by more than a threshold."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.30,
        description: str | None = None,
        fast_period: int = 20,
        slow_period: int = 50,
        threshold_pct: float = 0.005,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
            )
        if threshold_pct <= 0:
            raise ValueError(f"threshold_pct must be positive, got {threshold_pct}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.threshold_pct = threshold_pct

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="sma_trend",
            display_name="SMA Trend",
            summary="Bullish when fast SMA is above slow SMA",
            default_params={"fast_period": 20, "slow_period": 50, "threshold_pct": 0.005},
        )

    def required_indicators(self) -> tuple:
        from neon_radar.application.services.indicator_pipeline import IndicatorSpec

        return (
            IndicatorSpec(
                name="sma",
                params={"period": self.fast_period},
                tag=str(self.fast_period),
            ),
            IndicatorSpec(
                name="sma",
                params={"period": self.slow_period},
                tag=str(self.slow_period),
            ),
        )

    def evaluate(self, state: MarketState) -> Signal | None:
        fast_name = f"sma_{self.fast_period}"
        slow_name = f"sma_{self.slow_period}"
        fast = state.get_indicator_value(fast_name)
        slow = state.get_indicator_value(slow_name)
        if fast is None or slow is None:
            return None
        if slow == 0 or fast != fast or slow != slow:
            return None

        gap_pct = (fast - slow) / slow
        if abs(gap_pct) < self.threshold_pct:
            return None

        direction = 1.0 if gap_pct > 0 else -1.0
        magnitude = min(1.0, abs(gap_pct) / 0.05)
        value = direction * magnitude
        confidence = min(1.0, abs(gap_pct) / 0.02)
        arrow = "↑" if direction > 0 else "↓"
        return Signal(
            name=self.name,
            weight=self.weight,
            value=value,
            confidence=confidence,
            description=(
                f"SMA({self.fast_period}) {arrow} SMA({self.slow_period}) "
                f"(gap {gap_pct * 100:+.2f}%)"
            ),
            evidence=(
                EvidenceItem("fast_sma", f"{fast:.4f}"),
                EvidenceItem("slow_sma", f"{slow:.4f}"),
                EvidenceItem("gap_pct", f"{gap_pct * 100:.3f}"),
            ),
        )
