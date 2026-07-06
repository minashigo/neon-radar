"""Volatility filter rule.

Examines ATR as a percentage of the current price:

* ATR very low → market is sleepy, signals less reliable
  → low confidence, direction neutral.
* ATR very high → market is chaotic, signals less reliable
  → low confidence, direction neutral.
* ATR in the sweet spot → signals are more trustworthy
  → direction-neutral but confidence boost.

This rule does not contribute to direction. Its sole purpose is
to **lower the overall confidence** of the score when volatility
is outside the configured bounds. The :func:`aggregate` function
weights confidence by rule weight, so a low-confidence volatility
signal pulls the total confidence down — exactly what we want.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("volatility_filter")
class VolatilityFilterRule(FactorRule):
    """Reduce confidence when ATR is outside the comfort zone."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.20,
        description: str | None = None,
        period: int = 14,
        min_atr_pct: float = 0.003,  # 0.3% of price
        max_atr_pct: float = 0.05,  # 5% of price
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        if not 0 < min_atr_pct < max_atr_pct:
            raise ValueError(
                f"Need 0 < min_atr_pct ({min_atr_pct}) < max_atr_pct ({max_atr_pct})"
            )
        self.period = period
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="volatility_filter",
            display_name="Volatility Filter",
            summary="Penalises confidence when ATR is outside comfort zone",
            default_params={"period": 14, "min_atr_pct": 0.003, "max_atr_pct": 0.05},
        )

    def required_indicators(self) -> tuple:
        # Lazy import to avoid circular dependency.
        from neon_radar.application.services.indicator_pipeline import IndicatorSpec

        return (
            IndicatorSpec(
                name="atr",
                params={"period": self.period},
                tag=str(self.period),
            ),
        )

    def evaluate(self, state: MarketState) -> Signal | None:
        atr = state.get_indicator_value(f"atr_{self.period}")
        if atr is None or atr != atr:  # NaN guard
            return None

        latest = state.primary_series.latest()
        if latest is None or latest.close <= 0:
            return None
        atr_pct = atr / latest.close

        if atr_pct < self.min_atr_pct:
            return Signal(
                name=self.name,
                weight=self.weight,
                value=0.0,
                confidence=0.3,
                description=(
                    f"Low volatility ({atr_pct * 100:.2f}% < "
                    f"{self.min_atr_pct * 100:.1f}%) — signals unreliable"
                ),
                evidence=(
                    EvidenceItem("atr_pct", f"{atr_pct * 100:.3f}%"),
                    EvidenceItem("atr", f"{atr:.4f}"),
                    EvidenceItem("price", f"{latest.close:.4f}"),
                ),
            )

        if atr_pct > self.max_atr_pct:
            return Signal(
                name=self.name,
                weight=self.weight,
                value=0.0,
                confidence=0.3,
                description=(
                    f"High volatility ({atr_pct * 100:.2f}% > "
                    f"{self.max_atr_pct * 100:.1f}%) — signals unreliable"
                ),
                evidence=(
                    EvidenceItem("atr_pct", f"{atr_pct * 100:.3f}%"),
                    EvidenceItem("atr", f"{atr:.4f}"),
                ),
            )

        # In the sweet spot — confidence boost.
        return Signal(
            name=self.name,
            weight=self.weight,
            value=0.0,
            confidence=0.9,
            description=(
                f"Volatility in comfort zone ({atr_pct * 100:.2f}%)"
            ),
            evidence=(
                EvidenceItem("atr_pct", f"{atr_pct * 100:.3f}%"),
                EvidenceItem("atr", f"{atr:.4f}"),
            ),
        )
