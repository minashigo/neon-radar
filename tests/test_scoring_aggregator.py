"""Tests for the score aggregator."""

from __future__ import annotations

import pytest

from neon_radar.domain.scoring.aggregator import aggregate
from neon_radar.domain.scoring.value_objects import Signal


def _sig(name: str, value: float, weight: float = 0.25, confidence: float = 1.0) -> Signal:
    return Signal(
        name=name,
        weight=weight,
        value=value,
        confidence=confidence,
        description="",
    )


class TestAggregate:
    def test_empty_signals(self) -> None:
        s = aggregate(())
        assert s.value == 0.0
        assert s.confidence == 0.0
        assert s.long_score == 0.0
        assert s.short_score == 0.0
        assert s.contributing_signals == 0

    def test_single_bullish(self) -> None:
        s = aggregate((_sig("r", value=1.0, weight=1.0),))
        assert s.value == pytest.approx(1.0)
        assert s.long_score == pytest.approx(1.0)
        assert s.short_score == 0.0

    def test_single_bearish(self) -> None:
        s = aggregate((_sig("r", value=-1.0, weight=1.0),))
        assert s.value == pytest.approx(-1.0)
        assert s.long_score == 0.0
        assert s.short_score == pytest.approx(1.0)

    def test_mixed_signals(self) -> None:
        """Long and short contributions cancel."""
        s = aggregate(
            (
                _sig("a", value=+0.6, weight=0.5),
                _sig("b", value=-0.4, weight=0.5),
            )
        )
        # Long: 0.6 * 0.5 = 0.30. Short: 0.4 * 0.5 = 0.20. Value = 0.10.
        assert s.value == pytest.approx(0.10)
        assert s.long_score == pytest.approx(0.30)
        assert s.short_score == pytest.approx(0.20)

    def test_confidence_weighted(self) -> None:
        """Confidence is a weighted average of signal confidences."""
        s = aggregate(
            (
                _sig("a", value=0.5, weight=0.5, confidence=1.0),
                _sig("b", value=0.5, weight=0.5, confidence=0.0),
            )
        )
        assert s.confidence == pytest.approx(0.5)

    def test_low_confidence_pulls_score_down(self) -> None:
        """After Stage 5A decoupling: low confidence does NOT scale value.

        This test was renamed in spirit. The behaviour is now expressed
        by ``test_low_confidence_does_not_scale_value`` in
        ``test_scoring_aggregator_decoupled.py``. We keep this entry as
        a marker that the old formula was retired.
        """
        pytest.skip(
            "Behaviour verified in test_scoring_aggregator_decoupled.py "
            "(decoupled formula — low confidence no longer scales value)."
        )

    def test_neutral_signals_only(self) -> None:
        s = aggregate(
            (
                _sig("a", value=0.0, weight=0.5),
                _sig("b", value=0.0, weight=0.5),
            )
        )
        assert s.value == 0.0
        assert s.long_score == 0.0
        assert s.short_score == 0.0
        assert s.contributing_signals == 2
