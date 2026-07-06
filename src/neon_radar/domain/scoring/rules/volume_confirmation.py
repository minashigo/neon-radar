"""Volume confirmation rule.

Compares the latest bar's volume to its moving average.

* Volume > ``strong_multiplier`` x VolumeMA → strong confirmation
  (``+1`` for trends, ``-1`` against - this rule is **direction-agnostic**,
  but we tie-break to the recent candle direction so a high-volume
  bullish candle still tilts bullish).
* Volume < ``weak_multiplier`` x VolumeMA → weak / no participation.
* Otherwise neutral.

The rule looks up the volume moving average by ``"volume_ma_{period}"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("volume_confirmation")
class VolumeConfirmationRule(FactorRule):
    """Strong volume vs weak volume as a directional modifier."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.25,
        description: str | None = None,
        period: int = 20,
        strong_multiplier: float = 1.5,
        weak_multiplier: float = 0.5,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        if period < 1:
            raise ValueError(f"period must be positive, got {period}")
        if strong_multiplier <= 1:
            raise ValueError(
                f"strong_multiplier must be > 1, got {strong_multiplier}"
            )
        if weak_multiplier >= 1 or weak_multiplier <= 0:
            raise ValueError(
                f"weak_multiplier must be in (0, 1), got {weak_multiplier}"
            )
        self.period = period
        self.strong_multiplier = strong_multiplier
        self.weak_multiplier = weak_multiplier

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="volume_confirmation",
            display_name="Volume Confirmation",
            summary="Direction-agnostic; tilts with the latest candle's sign",
            default_params={"period": 20, "strong_multiplier": 1.5, "weak_multiplier": 0.5},
        )

    def required_indicators(self) -> tuple:
        # Lazy import to avoid circular dependency.
        from neon_radar.application.services.indicator_pipeline import IndicatorSpec

        return (
            IndicatorSpec(
                name="volume_ma",
                params={"period": self.period},
                tag=str(self.period),
            ),
        )

    def evaluate(self, state: MarketState) -> Signal | None:
        ma = state.get_indicator_value(f"volume_ma_{self.period}")
        if ma is None or ma != ma or ma <= 0:  # NaN guard
            return None

        latest = state.primary_series.latest()
        if latest is None:
            return None
        ratio = latest.volume / ma

        if ratio >= self.strong_multiplier:
            # Strong volume - follow the candle direction.
            candle_dir = 1.0 if latest.is_bullish else -1.0
            return Signal(
                name=self.name,
                weight=self.weight,
                value=candle_dir * min(1.0, (ratio - 1.0) / 1.0),
                confidence=0.8,
                description=(
                    f"Strong volume ({ratio:.2f}x avg), "
                    f"{'bullish' if candle_dir > 0 else 'bearish'} candle"
                ),
                evidence=(
                    EvidenceItem("volume_ratio", f"{ratio:.2f}"),
                    EvidenceItem("avg_volume", f"{ma:.0f}"),
                ),
            )

        if ratio <= self.weak_multiplier:
            # Low volume - neutral on direction.
            return Signal(
                name=self.name,
                weight=self.weight,
                value=0.0,
                confidence=0.5,
                description=f"Low volume ({ratio:.2f}x avg) - no conviction",
                evidence=(EvidenceItem("volume_ratio", f"{ratio:.2f}"),),
            )

        return None  # Normal volume - no opinion
