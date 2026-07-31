"""Narrative engine logic for Market Intelligence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType, NarrativeType
from neon_radar.domain.market_intelligence.models import MarketNarrative, SignalEvidence

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class NarrativeEngine:
    """Identifies active market narratives from intelligence signals."""

    def __init__(self, min_strength_threshold: float = 0.4, min_evidence_count: int = 2) -> None:
        """Initialize the narrative engine.

        Args:
            min_strength_threshold: Minimum calculated strength to activate a narrative.
            min_evidence_count: Minimum number of supporting signals to consider a narrative valid.
        """
        self.min_strength_threshold = min_strength_threshold
        self.min_evidence_count = min_evidence_count

    def compute_narratives(
        self, signals: Iterable[SignalEvidence], current_timestamp: int
    ) -> tuple[MarketNarrative, ...]:
        """Compute the active narratives based on the provided signals."""
        signals_list = tuple(signals)
        if not signals_list:
            return ()

        narratives: list[MarketNarrative] = []

        # Helper to extract supporting signals for a specific condition
        def _evaluate_narrative(
            n_type: NarrativeType, condition: Callable[[SignalEvidence], bool]
        ) -> MarketNarrative | None:
            supporting = [s for s in signals_list if condition(s)]
            if len(supporting) < self.min_evidence_count:
                return None

            # Calculate aggregate strength using independent probability (1 - prod(1 - p))
            p_none = 1.0
            for s in supporting:
                power = s.strength * s.source.weight
                p_none *= 1.0 - power
            strength = 1.0 - p_none

            # Duration based on the oldest supporting signal
            oldest_ts = min(s.timestamp for s in supporting)
            duration = max(0, current_timestamp - oldest_ts)

            if strength >= self.min_strength_threshold:
                return MarketNarrative(
                    type=n_type,
                    strength=min(1.0, strength),
                    duration=duration,
                    evidence_count=len(supporting),
                )
            return None

        # 1. ETF Accumulation
        etf_acc = _evaluate_narrative(
            NarrativeType.ETF_ACCUMULATION,
            lambda s: s.type == IntelligenceSignalType.ETF_FLOW and s.direction > 0,
        )
        if etf_acc:
            narratives.append(etf_acc)

        # 2. Risk-On (Macro bullish + Tech/Market bullish)
        risk_on = _evaluate_narrative(
            NarrativeType.RISK_ON,
            lambda s: (
                (
                    s.type in (IntelligenceSignalType.CPI, IntelligenceSignalType.FOMC)
                    and s.direction > 0
                )
                or (s.type == IntelligenceSignalType.SOCIAL_SENTIMENT and s.direction > 0)
            ),
        )
        if risk_on:
            narratives.append(risk_on)

        # 3. Risk-Off
        risk_off = _evaluate_narrative(
            NarrativeType.RISK_OFF,
            lambda s: (
                (
                    s.type in (IntelligenceSignalType.CPI, IntelligenceSignalType.FOMC)
                    and s.direction < 0
                )
                or (s.type == IntelligenceSignalType.DXY and s.direction > 0)
            ),
        )
        if risk_off:
            narratives.append(risk_off)

        # 4. Short Squeeze (High funding/OI bearish + liquidations bullish + sudden price up)
        short_sq = _evaluate_narrative(
            NarrativeType.SHORT_SQUEEZE,
            lambda s: s.type == IntelligenceSignalType.LIQUIDATIONS and s.direction > 0,
        )
        if short_sq:
            narratives.append(short_sq)

        # 5. Long Squeeze
        long_sq = _evaluate_narrative(
            NarrativeType.LONG_SQUEEZE,
            lambda s: s.type == IntelligenceSignalType.LIQUIDATIONS and s.direction < 0,
        )
        if long_sq:
            narratives.append(long_sq)

        # 6. Alt Season (BTC Dominance falling)
        alt_season = _evaluate_narrative(
            NarrativeType.ALT_SEASON,
            lambda s: s.type == IntelligenceSignalType.BTC_DOMINANCE and s.direction < 0,
        )
        if alt_season:
            narratives.append(alt_season)

        # 7. BTC Dominance Expansion
        btc_dom = _evaluate_narrative(
            NarrativeType.BTC_DOMINANCE_EXPANSION,
            lambda s: s.type == IntelligenceSignalType.BTC_DOMINANCE and s.direction > 0,
        )
        if btc_dom:
            narratives.append(btc_dom)

        # 8. Stablecoin Expansion
        stable_exp = _evaluate_narrative(
            NarrativeType.STABLECOIN_EXPANSION,
            lambda s: s.type == IntelligenceSignalType.STABLECOIN_FLOW and s.direction > 0,
        )
        if stable_exp:
            narratives.append(stable_exp)

        return tuple(sorted(narratives, key=lambda n: n.strength, reverse=True))
