"""Funding rate rule — contrarian crowd positioning signal.

Positive funding means longs pay shorts: the market is crowded long.
Negative funding means shorts pay longs: the market is crowded short.

This rule applies a **contrarian** tilt:

* Funding above ``neutral_band`` → bearish (fade crowded longs).
* Funding below ``-neutral_band`` → bullish (fade crowded shorts).
* Inside the band → no opinion (returns ``None``).

Magnitude and confidence scale with ``|rate|`` up to
``strong_threshold``. No indicators are required — the rule reads
``state.funding_rate`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("funding_rate")
class FundingRateRule(FactorRule):
    """Contrarian signal from perpetual funding rate."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.15,
        description: str | None = None,
        neutral_band: float = 0.00005,
        strong_threshold: float = 0.0005,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        if neutral_band <= 0:
            raise ValueError(f"neutral_band must be positive, got {neutral_band}")
        if strong_threshold <= neutral_band:
            raise ValueError(
                f"strong_threshold ({strong_threshold}) must be > "
                f"neutral_band ({neutral_band})"
            )
        self.neutral_band = neutral_band
        self.strong_threshold = strong_threshold

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="funding_rate",
            display_name="Funding Rate",
            summary="Contrarian tilt: fade crowded longs/shorts via funding",
            default_params={
                "neutral_band": 0.00005,
                "strong_threshold": 0.0005,
            },
        )

    def required_indicators(self) -> tuple:
        return ()

    def evaluate(self, state: MarketState) -> Signal | None:
        fr = state.funding_rate
        if fr is None:
            return None

        rate = fr.rate
        if rate != rate:  # NaN guard
            return None

        if abs(rate) < self.neutral_band:
            return None

        # Contrarian: positive funding → bearish, negative → bullish.
        direction = -1.0 if rate > 0 else 1.0
        magnitude = min(1.0, abs(rate) / self.strong_threshold)
        value = direction * magnitude
        confidence = min(1.0, abs(rate) / (self.strong_threshold * 0.5))

        rate_bps = rate * 10_000
        crowd = "longs pay shorts" if rate > 0 else "shorts pay longs"
        tilt = "bearish fade" if direction < 0 else "bullish fade"
        return Signal(
            name=self.name,
            weight=self.weight,
            value=value,
            confidence=confidence,
            description=(
                f"Funding {rate_bps:+.2f} bps ({crowd}) → {tilt}"
            ),
            evidence=(
                EvidenceItem("rate", f"{rate:.6f}"),
                EvidenceItem("rate_bps", f"{rate_bps:.3f}"),
                EvidenceItem("annualized_pct", f"{fr.annualized_pct:.2f}%"),
            ),
        )
