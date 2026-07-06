"""Tests for the decoupled aggregator (Score and Confidence independent).

After Stage 5A: ``value`` is derived from ``signal.value * weight``
only; ``confidence`` is derived from ``signal.confidence`` only.
"""

from __future__ import annotations

import pytest

from neon_radar.domain.scoring.aggregator import aggregate
from neon_radar.domain.scoring.value_objects import Signal


def _sig(name: str, value: float, weight: float = 0.5, confidence: float = 1.0) -> Signal:
    return Signal(
        name=name,
        weight=weight,
        value=value,
        confidence=confidence,
        description="",
    )


class TestDecoupledAggregation:
    def test_empty_returns_zero(self) -> None:
        s = aggregate(())
        assert s.value == 0.0
        assert s.confidence == 0.0
        assert s.long_score == 0.0
        assert s.short_score == 0.0
        assert s.contributing_signals == 0

    def test_value_does_not_use_confidence(self) -> None:
        """Even with low confidence, value reflects direction and magnitude."""
        s = aggregate(
            (
                _sig("a", value=1.0, weight=1.0, confidence=0.1),
            )
        )
        # OLD: would be 1.0 * 1.0 * 0.1 = 0.1
        # NEW: 1.0 * 1.0 / 1.0 = 1.0 (confidence is separate)
        assert s.value == pytest.approx(1.0)
        # Confidence is reported separately.
        assert s.confidence == pytest.approx(0.1)

    def test_confidence_independent_of_value(self) -> None:
        """High value with zero confidence gives zero confidence score."""
        s = aggregate(
            (
                _sig("a", value=1.0, weight=1.0, confidence=0.0),
            )
        )
        assert s.value == pytest.approx(1.0)
        assert s.confidence == pytest.approx(0.0)

    def test_balanced_bull_and_bear_cancel(self) -> None:
        s = aggregate(
            (
                _sig("a", value=+0.6, weight=0.5),
                _sig("b", value=-0.4, weight=0.5),
            )
        )
        # (0.6*0.5 - 0.4*0.5) / 1.0 = 0.10
        assert s.value == pytest.approx(0.10)
        assert s.long_score == pytest.approx(0.30)
        assert s.short_score == pytest.approx(0.20)

    def test_confidence_weighted_average(self) -> None:
        s = aggregate(
            (
                _sig("a", value=0.5, weight=0.5, confidence=1.0),
                _sig("b", value=0.5, weight=0.5, confidence=0.0),
            )
        )
        assert s.confidence == pytest.approx(0.5)


class TestMinConfidenceFilter:
    def test_filters_low_confidence(self) -> None:
        s = aggregate(
            (
                _sig("low", value=0.9, weight=0.5, confidence=0.1),
                _sig("high", value=0.5, weight=0.5, confidence=0.8),
            ),
            min_confidence=0.3,
        )
        # Only "high" contributes.
        assert s.contributing_signals == 1
        assert s.value == pytest.approx(0.5)
        assert s.confidence == pytest.approx(0.8)

    def test_all_filtered_returns_zero(self) -> None:
        s = aggregate(
            (
                _sig("a", value=0.9, weight=1.0, confidence=0.1),
                _sig("b", value=-0.5, weight=1.0, confidence=0.2),
            ),
            min_confidence=0.3,
        )
        assert s.contributing_signals == 0
        assert s.value == 0.0
        assert s.confidence == 0.0

    def test_default_zero_keeps_all(self) -> None:
        s = aggregate((_sig("a", value=0.5, weight=1.0, confidence=0.1),))
        assert s.contributing_signals == 1

    def test_rejects_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            aggregate((_sig("a", value=0.5)), min_confidence=-0.1)
        with pytest.raises(ValueError):
            aggregate((_sig("a", value=0.5)), min_confidence=1.5)
