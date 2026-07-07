"""Candle breakout rule.

Bullish when the latest candle closes above the previous candle's high.
Bearish when the latest candle closes below the previous candle's low.
Returns ``None`` otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("candle_breakout")
class CandleBreakoutRule(FactorRule):
    """Simple breakout rule based on the latest candle vs. the previous one."""

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
            name="candle_breakout",
            display_name="Candle Breakout",
            summary="Bullish when the latest candle closes above the previous high; bearish below the previous low",
            default_params={},
        )

    def evaluate(self, state: MarketState) -> Signal | None:
        candles = state.primary_series.candles
        if len(candles) < 2:
            return None

        prev = candles[-2]
        latest = candles[-1]

        if latest.close > prev.high:
            return Signal(
                name=self.name,
                weight=self.weight,
                value=1.0,
                confidence=0.8,
                description="Latest candle closed above the previous candle's high",
                evidence=(
                    EvidenceItem("prev_high", f"{prev.high:.4f}"),
                    EvidenceItem("latest_close", f"{latest.close:.4f}"),
                ),
            )

        if latest.close < prev.low:
            return Signal(
                name=self.name,
                weight=self.weight,
                value=-1.0,
                confidence=0.8,
                description="Latest candle closed below the previous candle's low",
                evidence=(
                    EvidenceItem("prev_low", f"{prev.low:.4f}"),
                    EvidenceItem("latest_close", f"{latest.close:.4f}"),
                ),
            )

        return None
