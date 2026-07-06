"""Open interest confirmation rule.

This confidence-only rule uses the latest open interest snapshot and
the recent price-volume context to decide whether market participation
confirms or diverges from recent price movement.

It intentionally returns `value=0.0` (does not pick direction) and only
affects overall `Score.confidence` through its `confidence` output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("open_interest_confirmation")
class OpenInterestConfirmationRule(FactorRule):
    """Confidence-only rule based on Open Interest magnitude vs recent volume.

    Parameters
    ----------
    period
        Number of candles to use for average quote-volume (default 20).
    low_ratio
        If `oi_quote / avg_quote_volume < low_ratio` → low confidence.
    high_ratio
        If ratio >= high_ratio → strong participation (higher confidence
        when aligned with price movement).
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.10,
        description: str | None = None,
        period: int = 20,
        low_ratio: float = 0.5,
        high_ratio: float = 3.0,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        if period < 1:
            raise ValueError("period must be positive")
        if not (0.0 < low_ratio < high_ratio):
            raise ValueError("need 0 < low_ratio < high_ratio")
        self.period = period
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="open_interest_confirmation",
            display_name="Open Interest Confirmation",
            summary=(
                "Confidence-only factor: high open interest vs recent volume "
                "increases confidence when aligned with price movement"
            ),
            default_params={"period": 20, "low_ratio": 0.5, "high_ratio": 3.0},
        )

    def required_indicators(self) -> tuple:
        return ()

    def evaluate(self, state: MarketState) -> Signal | None:
        oi = state.open_interest
        if oi is None:
            return None

        latest = state.primary_series.latest()
        # Need at least two candles to compute recent price move and average
        if latest is None or len(state.primary_series) < 2:
            return None

        # Compute previous close to get simple price direction
        prev = state.primary_series.candles[-2]
        price_dir = 0
        if latest.close > prev.close:
            price_dir = 1
        elif latest.close < prev.close:
            price_dir = -1

        # Compute approximate open interest in quote currency
        oi_quote = oi.value_quote if oi.value_quote is not None else oi.value * latest.close

        # Average recent quote-volume
        n = min(self.period, len(state.primary_series))
        recent = state.primary_series.candles[-n:]
        avg_quote_vol = 0.0
        for c in recent:
            avg_quote_vol += c.volume * c.close
        avg_quote_vol /= n

        if avg_quote_vol <= 0.0:
            return None

        ratio = oi_quote / avg_quote_vol

        # Map ratio and alignment to confidence
        if ratio < self.low_ratio:
            confidence = 0.25
            desc = f"Low OI vs volume (ratio {ratio:.2f}) — lower confidence"
        elif ratio >= self.high_ratio:
            # Strong participation. If aligned with price movement → boost
            if price_dir > 0:
                confidence = 0.9
                desc = f"High OI confirms bullish move (ratio {ratio:.2f})"
            elif price_dir < 0:
                confidence = 0.9
                desc = f"High OI confirms bearish move (ratio {ratio:.2f})"
            else:
                confidence = 0.7
                desc = f"High OI (ratio {ratio:.2f}) — neutral price"
        else:
            # Medium participation — moderate confidence; penalize divergence
            if (price_dir > 0 and ratio < 1.0) or (price_dir < 0 and ratio < 1.0):
                confidence = 0.45
                desc = f"Weak participation vs recent volume (ratio {ratio:.2f})"
            else:
                confidence = 0.6
                desc = f"Moderate OI (ratio {ratio:.2f})"

        return Signal(
            name=self.name,
            weight=self.weight,
            value=0.0,
            confidence=confidence,
            description=desc,
            evidence=(
                EvidenceItem("oi_quote", f"{oi_quote:.2f}"),
                EvidenceItem("avg_quote_vol", f"{avg_quote_vol:.2f}"),
                EvidenceItem("ratio", f"{ratio:.3f}"),
            ),
        )
