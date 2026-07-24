"""Tests for Confluence logic in aggregator."""

from __future__ import annotations

import pytest

from neon_radar.domain.scoring.aggregator import aggregate
from neon_radar.domain.scoring.value_objects import ConfluenceState, Signal, SignalCategory


def _sig(name: str, value: float, category: SignalCategory, weight: float = 0.5, confidence: float = 0.5) -> Signal:
    return Signal(
        name=name,
        weight=weight,
        value=value,
        confidence=confidence,
        description="",
        category=category,
    )


class TestConfluence:
    def test_unaligned_single_category(self) -> None:
        """If only one category is present, it's UNALIGNED."""
        signals = (
            _sig("tech1", value=1.0, category=SignalCategory.TECHNICAL),
        )
        score = aggregate(signals)
        assert score.confluence_state == ConfluenceState.UNALIGNED
        assert score.confidence == 0.5  # base confidence

    def test_unaligned_neutral_direction(self) -> None:
        """If overall direction is neutral, it's UNALIGNED."""
        signals = (
            _sig("tech1", value=1.0, category=SignalCategory.TECHNICAL),
            _sig("mic1", value=-1.0, category=SignalCategory.MICROSTRUCTURE),
        )
        score = aggregate(signals)
        assert score.confluence_state == ConfluenceState.UNALIGNED
        assert score.value == 0.0

    def test_confirmed_bonus(self) -> None:
        """Multiple categories agreeing should yield CONFIRMED and a bonus."""
        signals = (
            _sig("tech1", value=1.0, category=SignalCategory.TECHNICAL),
            _sig("mic1", value=1.0, category=SignalCategory.MICROSTRUCTURE),
        )
        # Base conf is 0.5, bonus is 0.20 -> final conf = 0.70
        score = aggregate(signals, confluence_bonus=0.20)
        assert score.confluence_state == ConfluenceState.CONFIRMED
        assert score.confidence == pytest.approx(0.70)
        assert "Confirmed by" in score.confluence_details[0]

    def test_confirmed_bonus_cap(self) -> None:
        """Bonus should not exceed max_confidence_boost."""
        signals = (
            _sig("tech1", value=1.0, category=SignalCategory.TECHNICAL),
            _sig("mic1", value=1.0, category=SignalCategory.MICROSTRUCTURE),
            _sig("onc1", value=1.0, category=SignalCategory.ONCHAIN),
        )
        # 3 categories -> 2 bonuses = 0.40, but max is 0.30
        score = aggregate(signals, confluence_bonus=0.20, max_confidence_boost=0.30)
        assert score.confluence_state == ConfluenceState.CONFIRMED
        assert score.confidence == pytest.approx(0.80)  # 0.5 base + 0.30 max boost

    def test_conflicting_penalty(self) -> None:
        """Categories disagreeing with the primary direction yield CONFLICTING."""
        signals = (
            _sig("tech1", value=1.0, category=SignalCategory.TECHNICAL, weight=0.6), # 0.6
            _sig("tech2", value=1.0, category=SignalCategory.TECHNICAL, weight=0.6), # 0.6 => total 1.2 long
            _sig("mic1", value=-1.0, category=SignalCategory.MICROSTRUCTURE, weight=0.4), # total 0.4 short
        )
        # Primary direction is LONG (value > 0). MICROSTRUCTURE is SHORT.
        # Conflicting categories = 1 (MICROSTRUCTURE). Penalty = 0.15.
        score = aggregate(signals, confluence_penalty=0.15)
        assert score.confluence_state == ConfluenceState.CONFLICTING
        assert score.confidence == pytest.approx(0.5 - 0.15)
        assert "Conflicting" in score.confluence_details[1]

    def test_conflicting_penalty_floor(self) -> None:
        """Penalty cannot drive confidence below 0.0."""
        signals = (
            _sig("tech1", value=1.0, category=SignalCategory.TECHNICAL, weight=0.8, confidence=0.1),
            _sig("mic1", value=-1.0, category=SignalCategory.MICROSTRUCTURE, weight=0.2, confidence=0.1),
        )
        # Base conf = 0.1, penalty = 0.15 -> should floor to 0.0
        score = aggregate(signals, confluence_penalty=0.15)
        assert score.confluence_state == ConfluenceState.CONFLICTING
        assert score.confidence == 0.0
