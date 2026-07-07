"""Bollinger Bands rule.

This first version is intentionally simple:
* bullish when the latest close is above the upper band;
* bearish when the latest close is below the lower band;
* returns ``None`` otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("bollinger_bands")
class BollingerBandsRule(FactorRule):
    """Simple breakout-style rule based on Bollinger Bands."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.20,
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="bollinger_bands",
            display_name="Bollinger Bands",
            summary="Bullish when price closes above the upper band; bearish below the lower band",
            default_params={},
        )

    def evaluate(self, state: MarketState) -> Signal | None:
        bands = state.get_indicator("bollinger")
        if bands is None:
            return None

        latest = bands.latest()
        if latest is None:
            return None

        upper = latest.get("upper")
        lower = latest.get("lower")
        close = state.primary_series.candles[-1].close if state.primary_series.candles else None

        if upper is None or lower is None or close is None:
            return None

        if close > upper:
            return Signal(
                name=self.name,
                weight=self.weight,
                value=1.0,
                confidence=0.8,
                description="Price closed above the upper Bollinger Band",
                evidence=(
                    EvidenceItem("upper", f"{upper:.4f}"),
                    EvidenceItem("close", f"{close:.4f}"),
                ),
            )

        if close < lower:
            return Signal(
                name=self.name,
                weight=self.weight,
                value=-1.0,
                confidence=0.8,
                description="Price closed below the lower Bollinger Band",
                evidence=(
                    EvidenceItem("lower", f"{lower:.4f}"),
                    EvidenceItem("close", f"{close:.4f}"),
                ),
            )

        return None
