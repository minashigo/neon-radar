"""Tests for scoring value objects (Score, Signal, AnalysisResult, FactorRule)."""

from __future__ import annotations

import pytest

from neon_radar.config.models import TimeFrame
from neon_radar.domain.enums import Bias
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.scoring import (
    AnalysisResult,
    EvidenceItem,
    FactorRule,
    Score,
    Signal,
)


class TestEvidenceItem:
    def test_basic(self) -> None:
        e = EvidenceItem(key="rsi", value="72.3")
        assert e.key == "rsi"
        assert e.value == "72.3"

    def test_rejects_empty_key(self) -> None:
        with pytest.raises(ValueError):
            EvidenceItem(key="", value="x")


class TestSignal:
    def test_basic(self) -> None:
        s = Signal(
            name="ema_cross",
            weight=0.3,
            value=0.8,
            confidence=0.7,
            description="EMA(20) crossed above EMA(50)",
        )
        assert s.is_bullish
        assert not s.is_bearish

    def test_bearish(self) -> None:
        s = Signal(name="rsi_ob", weight=0.2, value=-0.5, confidence=0.6, description="RSI > 70")
        assert s.is_bearish

    def test_neutral(self) -> None:
        s = Signal(name="noise", weight=0.1, value=0.0, confidence=0.5, description="")
        assert not s.is_bullish
        assert not s.is_bearish

    def test_rejects_weight_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            Signal(name="x", weight=1.5, value=0.0, confidence=0.5, description="")
        with pytest.raises(ValueError):
            Signal(name="x", weight=-0.1, value=0.0, confidence=0.5, description="")

    def test_rejects_value_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            Signal(name="x", weight=0.1, value=2.0, confidence=0.5, description="")
        with pytest.raises(ValueError):
            Signal(name="x", weight=0.1, value=-2.0, confidence=0.5, description="")

    def test_rejects_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            Signal(name="x", weight=0.1, value=0.0, confidence=1.5, description="")

    def test_with_evidence(self) -> None:
        s = Signal(
            name="rsi",
            weight=0.2,
            value=0.5,
            confidence=0.8,
            description="RSI in bull zone",
            evidence=(
                EvidenceItem("rsi", "65.4"),
                EvidenceItem("period", "14"),
            ),
        )
        assert len(s.evidence) == 2
        assert s.evidence[0].key == "rsi"


class TestScore:
    def test_bullish(self) -> None:
        s = Score(
            value=0.7,
            confidence=0.8,
            long_score=0.8,
            short_score=0.1,
            contributing_signals=4,
        )
        assert s.bias is Bias.BULLISH
        assert s.magnitude == pytest.approx(0.7)

    def test_bearish(self) -> None:
        s = Score(
            value=-0.6,
            confidence=0.7,
            long_score=0.2,
            short_score=0.8,
            contributing_signals=3,
        )
        assert s.bias is Bias.BEARISH

    def test_neutral_zone(self) -> None:
        s = Score(
            value=0.1,
            confidence=0.5,
            long_score=0.3,
            short_score=0.2,
            contributing_signals=2,
        )
        assert s.bias is Bias.NEUTRAL

    def test_at_boundary_bullish(self) -> None:
        s = Score(
            value=0.21,
            confidence=0.5,
            long_score=0.21,
            short_score=0.0,
            contributing_signals=1,
        )
        assert s.bias is Bias.BULLISH

    def test_rejects_value_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            Score(value=1.5, confidence=0.5, long_score=0.0, short_score=0.0, contributing_signals=0)

    def test_rejects_negative_components(self) -> None:
        with pytest.raises(ValueError):
            Score(value=0.0, confidence=0.5, long_score=-0.1, short_score=0.0, contributing_signals=0)


class TestAnalysisResult:
    def test_basic(self) -> None:
        score = Score(value=0.5, confidence=0.8, long_score=0.7, short_score=0.2, contributing_signals=3)
        signals = (
            Signal(name="a", weight=0.4, value=0.6, confidence=0.7, description=""),
            Signal(name="b", weight=0.3, value=0.4, confidence=0.9, description=""),
        )
        result = AnalysisResult(score=score, signals=signals, summary="bullish bias", computed_at=1)
        assert result.bias is Bias.BULLISH
        assert result.signal_count == 2

    def test_signals_by_name(self) -> None:
        signals = (
            Signal(name="a", weight=0.3, value=0.5, confidence=0.7, description=""),
            Signal(name="b", weight=0.3, value=0.4, confidence=0.7, description=""),
            Signal(name="a", weight=0.3, value=0.6, confidence=0.7, description=""),  # dup
        )
        result = AnalysisResult(
            score=Score(value=0.5, confidence=0.7, long_score=0.5, short_score=0.0, contributing_signals=2),
            signals=signals,
            summary="",
            computed_at=1,
        )
        grouped = result.signals_by_name()
        assert set(grouped) == {"a", "b"}
        # First signal with a given name wins.
        assert grouped["a"].value == 0.5


# ---------------------------------------------------------------------------
# FactorRule ABC
# ---------------------------------------------------------------------------


class _StubRule(FactorRule):
    """Returns a constant bullish signal. Used for testing."""

    @classmethod
    def description(cls):  # type: ignore[override]
        from neon_radar.domain.scoring.factor_rule import RuleDescription
        return RuleDescription(
            name="stub", display_name="Stub", summary="stub rule", default_params={}
        )

    def evaluate(self, state: MarketState) -> Signal:
        return Signal(
            name=self.name,
            weight=self.weight,
            value=0.5,
            confidence=0.6,
            description=self.description_text,
        )


class TestFactorRule:
    def test_basic(self) -> None:
        rule = _StubRule(name="stub", weight=0.3)
        assert rule.name == "stub"
        assert rule.weight == 0.3

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            _StubRule(name="", weight=0.1)

    def test_rejects_bad_weight(self) -> None:
        with pytest.raises(ValueError):
            _StubRule(name="x", weight=1.5)
        with pytest.raises(ValueError):
            _StubRule(name="x", weight=-0.1)

    def test_evaluate_called(self) -> None:
        rule = _StubRule(name="stub", weight=0.4)
        # A minimal MarketState — the stub rule ignores it.
        candle = OHLCV(open_time=1, open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0)
        series = KlineSeries(
            symbol=Symbol("BTCUSDT"),
            timeframe=TimeFrame.H4,
            candles=(candle,),
        )
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=1,
            primary_series=series,
        )
        sig = rule.evaluate(state)
        assert sig.name == "stub"
        assert sig.weight == 0.4
