import pytest

from neon_radar.domain.scoring.aggregator import aggregate
from neon_radar.domain.scoring.value_objects import Signal


def test_only_directional_signals():
    a = Signal(name="A", weight=0.6, value=1.0, confidence=0.9, description="A")
    b = Signal(name="B", weight=0.4, value=-1.0, confidence=0.8, description="B")
    score = aggregate((a, b), min_confidence=0.0)

    assert score.long_score == pytest.approx(0.6)
    assert score.short_score == pytest.approx(0.4)
    assert score.value == pytest.approx(0.2)
    assert score.confidence == pytest.approx((0.6 * 0.9 + 0.4 * 0.8) / 1.0)


def test_directional_plus_confidence_only():
    a = Signal(name="A", weight=0.5, value=1.0, confidence=0.9, description="A")
    b = Signal(name="B", weight=0.3, value=-1.0, confidence=0.8, description="B")
    c = Signal(name="C", weight=0.1, value=0.0, confidence=0.5, description="C")
    d = Signal(name="D", weight=0.1, value=0.0, confidence=0.6, description="D")

    score = aggregate((a, b, c, d), min_confidence=0.0)

    # Directional normalization uses only A and B (weight 0.5 + 0.3 = 0.8)
    assert score.long_score == pytest.approx(0.5 / 0.8)
    assert score.short_score == pytest.approx(0.3 / 0.8)
    assert score.value == pytest.approx((0.5 / 0.8) - (0.3 / 0.8))
    # Confidence remains weighted average across all signals
    expected_conf = (0.5 * 0.9 + 0.3 * 0.8 + 0.1 * 0.5 + 0.1 * 0.6) / 1.0
    assert score.confidence == pytest.approx(expected_conf)


def test_only_confidence_only_signals():
    c = Signal(name="C", weight=0.4, value=0.0, confidence=0.5, description="C")
    d = Signal(name="D", weight=0.6, value=0.0, confidence=0.7, description="D")

    score = aggregate((c, d), min_confidence=0.0)

    assert score.long_score == pytest.approx(0.0)
    assert score.short_score == pytest.approx(0.0)
    assert score.value == pytest.approx(0.0)
    expected_conf = (0.4 * 0.5 + 0.6 * 0.7) / 1.0
    assert score.confidence == pytest.approx(expected_conf)


def test_multiple_confidence_modifiers_with_directional():
    a = Signal(name="A", weight=0.4, value=1.0, confidence=0.9, description="A")
    b = Signal(name="B", weight=0.2, value=-1.0, confidence=0.8, description="B")
    c = Signal(name="C", weight=0.1, value=0.0, confidence=0.2, description="C")
    d = Signal(name="D", weight=0.15, value=0.0, confidence=0.6, description="D")
    e = Signal(name="E", weight=0.15, value=0.0, confidence=0.7, description="E")

    score = aggregate((a, b, c, d, e), min_confidence=0.0)

    directional_weight = 0.4 + 0.2
    assert score.long_score == pytest.approx(0.4 / directional_weight)
    assert score.short_score == pytest.approx(0.2 / directional_weight)
    assert score.value == pytest.approx((0.4 / directional_weight) - (0.2 / directional_weight))
    expected_conf = (0.4 * 0.9 + 0.2 * 0.8 + 0.1 * 0.2 + 0.15 * 0.6 + 0.15 * 0.7) / 1.0
    assert score.confidence == pytest.approx(expected_conf)
