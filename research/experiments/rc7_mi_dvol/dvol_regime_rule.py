"""DVOL Regime-Conditioned rule.

Examines the 30-day Z-Score of the Deribit Volatility Index (DVOL)
during a specific market regime (e.g., BEAR_TREND) to identify
potential capitulation bottoms or mean-reverting conditions.

Supports multiple configurable candidate behaviors for Walk-Forward Analysis:
1. "directional": Contributes a bullish signal with confidence.
2. "confidence": Does not contribute directional value, only boosts confidence.
3. "filter": Lowers confidence drastically if DVOL is not elevated (acts as a veto).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("dvol_regime")
class DvolRegimeFactorRule(FactorRule):
    """Evaluates DVOL Z-Score during a specific market regime."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 1.0,
        description: str | None = None,
        behavior: str = "directional",  # 'directional', 'confidence', 'filter'
        z_score_threshold: float = 1.5,
        confidence_boost: float = 0.8,
        confidence_penalty: float = 0.1,  # Used by 'filter' behavior
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        if behavior not in ("directional", "confidence", "filter"):
            raise ValueError(f"Unknown behavior {behavior}")

        self.behavior = behavior
        self.z_score_threshold = z_score_threshold
        self.confidence_boost = confidence_boost
        self.confidence_penalty = confidence_penalty

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="dvol_regime",
            display_name="DVOL Regime Conditioned",
            summary="Evaluates DVOL Z-Score during Bear Trends",
            default_params={
                "behavior": "directional",
                "z_score_threshold": 1.5,
                "confidence_boost": 0.8,
                "confidence_penalty": 0.1,
            },
        )

    def evaluate(self, state: MarketState) -> Signal | None:
        from neon_radar.domain.trading.regime import MarketRegime

        # Rule only applies during BEAR_TREND
        if state.regime != MarketRegime.BEAR_TREND:
            return None

        # Check if DVOL is available
        if state.intelligence is None or state.intelligence.dvol_z_score_30d is None:
            return None

        z_score = state.intelligence.dvol_z_score_30d

        is_elevated = z_score > self.z_score_threshold

        evidence = (
            EvidenceItem("dvol_z_score_30d", f"{z_score:.2f}"),
            EvidenceItem("threshold", f"{self.z_score_threshold:.2f}"),
            EvidenceItem("regime", state.regime.value),
        )

        if not is_elevated:
            if self.behavior == "filter":
                # Act as a veto (lower overall confidence) when conditions are not met
                return Signal(
                    name=self.name,
                    weight=self.weight,
                    value=0.0,
                    confidence=self.confidence_penalty,
                    description=f"DVOL Z-Score ({z_score:.2f}) not elevated in Bear Trend - unreliable.",
                    evidence=evidence,
                )
            # For 'directional' and 'confidence', we simply don't contribute if it's not elevated.
            return None

        # DVOL is elevated
        if self.behavior == "directional":
            return Signal(
                name=self.name,
                weight=self.weight,
                value=1.0,  # Bullish contribution
                confidence=self.confidence_boost,
                description=f"Capitulation signal: Elevated DVOL ({z_score:.2f}) in Bear Trend.",
                evidence=evidence,
            )
        elif self.behavior == "confidence":
            return Signal(
                name=self.name,
                weight=self.weight,
                value=0.0,  # No directional bias
                confidence=self.confidence_boost,
                description=f"Confidence boost: Elevated DVOL ({z_score:.2f}) in Bear Trend.",
                evidence=evidence,
            )
        elif self.behavior == "filter":
            # Pass the filter (provide high confidence, neutral direction)
            return Signal(
                name=self.name,
                weight=self.weight,
                value=0.0,
                confidence=self.confidence_boost,
                description=f"Filter passed: Elevated DVOL ({z_score:.2f}) in Bear Trend.",
                evidence=evidence,
            )

        return None
