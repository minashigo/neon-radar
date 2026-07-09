"""Higher-timeframe trend rule.

Evaluates the macro trend by comparing two exponential moving averages
on the higher timeframe. Requires ``higher_tf_series`` to be present
in the ``MarketState``.

* Fast HTF EMA above slow HTF EMA → bullish (+1).
* Fast HTF EMA below slow HTF EMA → bearish (-1).
* Small gap (within ``threshold_pct``) → neutral (returns ``None``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("higher_tf_trend")
class HigherTimeframeTrendRule(FactorRule):
    """Bullish if HTF fast EMA > HTF slow EMA by more than a threshold."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.40,
        description: str | None = None,
        fast_period: int = 20,
        slow_period: int = 50,
        threshold_pct: float = 0.005,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        if fast_period >= slow_period:
            raise ValueError(f"fast_period ({fast_period}) must be < slow_period ({slow_period})")
        if threshold_pct <= 0:
            raise ValueError(f"threshold_pct must be positive, got {threshold_pct}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.threshold_pct = threshold_pct

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="higher_tf_trend",
            display_name="Higher Timeframe Trend",
            summary="Evaluates the macro trend using HTF EMAs",
            default_params={"fast_period": 20, "slow_period": 50, "threshold_pct": 0.005},
        )

    def required_indicators(self) -> tuple:
        from neon_radar.application.services.indicator_pipeline import IndicatorSpec

        return (
            IndicatorSpec(
                name="ema",
                params={"period": self.fast_period},
                tag=str(self.fast_period),
                target="higher_tf",
            ),
            IndicatorSpec(
                name="ema",
                params={"period": self.slow_period},
                tag=str(self.slow_period),
                target="higher_tf",
            ),
        )

    def evaluate(self, state: MarketState) -> Signal | None:
        if state.higher_tf_series is None:
            return None

        fast_name = f"htf_ema_{self.fast_period}"
        slow_name = f"htf_ema_{self.slow_period}"
        fast = state.get_indicator_value(fast_name)
        slow = state.get_indicator_value(slow_name)

        if fast is None or slow is None:
            return None
        if slow == 0 or fast != fast or slow != slow:  # NaN guard
            return None

        gap_pct = (fast - slow) / slow
        if abs(gap_pct) < self.threshold_pct:
            return None  # No clear trend

        direction = 1.0 if gap_pct > 0 else -1.0
        magnitude = min(1.0, abs(gap_pct) / 0.05)  # saturate at 5% gap
        value = direction * magnitude
        confidence = min(1.0, abs(gap_pct) / 0.02)  # high confidence at 2%+

        arrow = "↑" if direction > 0 else "↓"
        return Signal(
            name=self.name,
            weight=self.weight,
            value=value,
            confidence=confidence,
            description=(
                f"HTF EMA({self.fast_period}) {arrow} HTF EMA({self.slow_period}) "
                f"(gap {gap_pct * 100:+.2f}%)"
            ),
            evidence=(
                EvidenceItem("htf_fast_ema", f"{fast:.4f}"),
                EvidenceItem("htf_slow_ema", f"{slow:.4f}"),
                EvidenceItem("htf_gap_pct", f"{gap_pct * 100:.3f}"),
            ),
        )
