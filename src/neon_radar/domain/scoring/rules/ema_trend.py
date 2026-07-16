"""EMA trend rule.

Compares two exponential moving averages on the primary timeframe.

* Fast EMA above slow EMA → bullish (+1, scaled by gap size).
* Fast EMA below slow EMA → bearish (-1, scaled by gap size).
* Small gap (within ``threshold_pct``) → neutral (returns ``None``).

The rule looks up indicators by the names ``"ema_{fast_period}"`` and
``"ema_{slow_period}"`` in the :class:`MarketState`. The orchestrator
must compute them with these exact names — see
:meth:`required_indicators`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("ema_trend")
class EMATrendRule(FactorRule):
    """Bullish if fast EMA > slow EMA by more than a threshold."""

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
            raise ValueError(f"fast_period ({fast_period}) must be < slow_period ({slow_period})")
        if threshold_pct <= 0:
            raise ValueError(f"threshold_pct must be positive, got {threshold_pct}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.threshold_pct = threshold_pct

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="ema_trend",
            display_name="EMA Trend",
            summary="Bullish when fast EMA is above slow EMA",
            default_params={"fast_period": 20, "slow_period": 50, "threshold_pct": 0.005},
        )

    def required_indicators(self) -> tuple:
        # Lazy import to avoid circular dependency at module load.
        from neon_radar.application.services.indicator_pipeline import IndicatorSpec

        return (
            IndicatorSpec(
                name="ema",
                params={"period": self.fast_period},
                tag=str(self.fast_period),
            ),
            IndicatorSpec(
                name="ema",
                params={"period": self.slow_period},
                tag=str(self.slow_period),
            ),
        )

    def evaluate(self, state: MarketState) -> Signal | None:
        fast_name = f"ema_{self.fast_period}"
        slow_name = f"ema_{self.slow_period}"
        fast = state.get_indicator_value(fast_name)
        slow = state.get_indicator_value(slow_name)
        if fast is None or slow is None:
            return None
        if slow == 0 or fast != fast or slow != slow:  # NaN guard
            return None

        gap_pct = (fast - slow) / slow
        if abs(gap_pct) < self.threshold_pct:
            return None  # No clear trend

        # Direction: +1 bullish (fast > slow), -1 bearish.
        # Magnitude scales with gap size (capped at 1.0).
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
                f"EMA({self.fast_period}) {arrow} EMA({self.slow_period}) "
                f"(gap {gap_pct * 100:+.2f}%)"
            ),
            evidence=(
                EvidenceItem("fast_ema", f"{fast:.4f}"),
                EvidenceItem("slow_ema", f"{slow:.4f}"),
                EvidenceItem("gap_pct", f"{gap_pct * 100:.3f}"),
            ),
        )
