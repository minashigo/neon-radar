"""Tests for the rule-based scoring engine."""

from __future__ import annotations

import pytest

from neon_radar.application.services.indicator_pipeline import (
    IndicatorSpec,
    compute_indicators,
)
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import Symbol
from neon_radar.domain.scoring import (
    EMATrendRule,
    FactorRule,
    RuleBasedEngine,
    Signal,
    VolatilityFilterRule,
)
from tests.conftest import make_series


class _AlwaysBullishRule(FactorRule):
    """Test fixture: returns +1 with confidence 1.0."""

    def __init__(self, *, name="always_bull", weight=0.5, **kwargs):
        super().__init__(name=name, weight=weight)

    @classmethod
    def description(cls):  # type: ignore[override]
        from neon_radar.domain.scoring.factor_rule import RuleDescription

        return RuleDescription(
            name="always_bull", display_name="AB", summary="always bull", default_params={}
        )

    def evaluate(self, state):
        return Signal(
            name=self.name,
            weight=self.weight,
            value=1.0,
            confidence=1.0,
            description="always bullish",
        )


class _AlwaysBearishRule(FactorRule):
    def __init__(self, *, name="always_bear", weight=0.5, **kwargs):
        super().__init__(name=name, weight=weight)

    @classmethod
    def description(cls):  # type: ignore[override]
        from neon_radar.domain.scoring.factor_rule import RuleDescription

        return RuleDescription(
            name="always_bear", display_name="AB", summary="always bear", default_params={}
        )

    def evaluate(self, state):
        return Signal(
            name=self.name,
            weight=self.weight,
            value=-1.0,
            confidence=1.0,
            description="always bearish",
        )


class _CrashyRule(FactorRule):
    """Always raises — tests engine robustness."""

    def __init__(self, *, name="crash", weight=0.1, **kwargs):
        super().__init__(name=name, weight=weight)

    @classmethod
    def description(cls):  # type: ignore[override]
        from neon_radar.domain.scoring.factor_rule import RuleDescription

        return RuleDescription(name="crash", display_name="C", summary="crashes", default_params={})

    def evaluate(self, state):
        raise RuntimeError("intentional crash for testing")


class TestRuleBasedEngine:
    def test_empty_rules_returns_neutral(self) -> None:
        series = make_series([100.0 + i for i in range(30)])
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=0,
            primary_series=series,
        )
        engine = RuleBasedEngine(rules=())
        result = engine.evaluate(state)
        assert result.bias.value == "Neutral"
        assert result.signal_count == 0

    def test_two_opposite_rules_cancel(self) -> None:
        series = make_series([100.0 + i for i in range(30)])
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=0,
            primary_series=series,
        )
        engine = RuleBasedEngine(
            rules=(_AlwaysBullishRule(), _AlwaysBearishRule()),
        )
        result = engine.evaluate(state)
        assert result.score.value == pytest.approx(0.0, abs=1e-9)

    def test_buggy_rule_is_skipped(self) -> None:
        series = make_series([100.0 + i for i in range(30)])
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=0,
            primary_series=series,
        )
        engine = RuleBasedEngine(
            rules=(_AlwaysBullishRule(), _CrashyRule()),
        )
        result = engine.evaluate(state)
        # Buggy rule contributes nothing; bullish rule gives +1.
        assert result.signal_count == 1
        assert result.score.value > 0

    def test_volatility_low_confidence_dominates(self) -> None:
        """A low-confidence volatility rule pulls overall confidence down."""

        series = make_series([100.0 + i for i in range(30)])
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=0,
            primary_series=series,
        )

        # Always-bullish (conf 1.0, weight 0.5) + low-confidence rule.
        class _LowConf(FactorRule):
            @classmethod
            def description(cls):  # type: ignore[override]
                from neon_radar.domain.scoring.factor_rule import RuleDescription

                return RuleDescription(
                    name="lowconf", display_name="L", summary="low", default_params={}
                )

            def evaluate(self, state):
                return Signal(
                    name="lowconf", weight=0.5, value=0.0, confidence=0.1, description="low"
                )

        engine = RuleBasedEngine(
            rules=(_AlwaysBullishRule(), _LowConf()),
        )
        result = engine.evaluate(state)
        # Confidence = (0.5 * 1.0 + 0.5 * 0.1) / 1.0 = 0.55
        assert result.score.confidence == pytest.approx(0.55, abs=1e-9)

    def test_integration_with_real_rules(self) -> None:
        """End-to-end: a real MarketState with real rules produces a result."""
        closes = [100.0 + i for i in range(60)]
        series = make_series(closes)
        specs = [
            IndicatorSpec(name="ema", params={"period": 20}, tag="20"),
            IndicatorSpec(name="ema", params={"period": 50}, tag="50"),
            IndicatorSpec(name="rsi", params={"period": 14}, tag="14"),
            IndicatorSpec(name="atr", params={"period": 14}, tag="14"),
            IndicatorSpec(name="volume_ma", params={"period": 20}, tag="20"),
        ]
        indicators = compute_indicators(series, specs)
        state = MarketState(
            symbol=Symbol("BTCUSDT"),
            timestamp=0,
            primary_series=series,
            indicator_series=tuple(indicators),
        )
        engine = RuleBasedEngine(
            rules=(EMATrendRule(), VolatilityFilterRule()),
        )
        result = engine.evaluate(state)
        assert result.signal_count >= 1
        # Strong uptrend → EMA bullish, volatility neutral-on-direction
        # but high confidence (in comfort zone).
        assert result.score.value > 0
