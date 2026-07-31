"""Consensus engine logic for Market Intelligence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.market_intelligence.enums import ConsensusDirection
from neon_radar.domain.market_intelligence.models import MarketConsensus, SignalEvidence

if TYPE_CHECKING:
    from collections.abc import Iterable


class ConsensusEngine:
    """Determines the market consensus by weighing filtered intelligence signals."""

    def __init__(
        self,
        bullish_threshold: float = 0.3,
        bearish_threshold: float = -0.3,
        conflict_threshold: float = 0.4,
    ) -> None:
        """Initialize the consensus engine.

        Args:
            bullish_threshold: Score above this is bullish consensus.
            bearish_threshold: Score below this is bearish consensus.
            conflict_threshold: Ratio of opposing forces that triggers a conflict state.
        """
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.conflict_threshold = conflict_threshold

    def compute_consensus(self, signals: Iterable[SignalEvidence]) -> MarketConsensus:
        """Compute the market consensus from a set of signals."""
        signals_list = tuple(signals)

        if not signals_list:
            return MarketConsensus(
                direction=ConsensusDirection.NEUTRAL,
                confidence=0.0,
                conflict_level=0.0,
            )

        total_weight = 0.0
        bullish_power = 0.0
        bearish_power = 0.0

        for s in signals_list:
            # Power = strength * source_weight
            power = s.strength * s.source.weight
            total_weight += power

            if s.direction > 0:
                bullish_power += power * s.direction
            elif s.direction < 0:
                bearish_power += power * abs(s.direction)

        if total_weight == 0.0:
            return MarketConsensus(
                direction=ConsensusDirection.NEUTRAL,
                confidence=0.0,
                conflict_level=0.0,
            )

        net_score = (bullish_power - bearish_power) / total_weight

        # Calculate conflict level (ratio of minority power to majority power)
        max_power = max(bullish_power, bearish_power)
        min_power = min(bullish_power, bearish_power)
        conflict_level = (min_power / max_power) if max_power > 0 else 0.0

        # Determine confidence (how much total evidence we have, up to a cap, scaled by conflict)
        # Cap total weight at a reasonable arbitrary value (e.g. 5.0) to represent 100% confidence baseline
        base_confidence = min(1.0, total_weight / 5.0)
        # Confidence drops as conflict rises
        confidence = base_confidence * (1.0 - conflict_level * 0.5)

        if conflict_level >= self.conflict_threshold and abs(net_score) < max(
            self.bullish_threshold, abs(self.bearish_threshold)
        ):
            direction = ConsensusDirection.CONFLICTING
        elif net_score >= self.bullish_threshold:
            direction = ConsensusDirection.BULLISH
        elif net_score <= self.bearish_threshold:
            direction = ConsensusDirection.BEARISH
        else:
            direction = ConsensusDirection.NEUTRAL

        return MarketConsensus(
            direction=direction,
            confidence=max(0.0, min(1.0, confidence)),
            conflict_level=max(0.0, min(1.0, conflict_level)),
        )
