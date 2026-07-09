"""Tests for the application-level analysis pipeline."""

from __future__ import annotations

import pytest

from neon_radar.application.services.analysis import (
    analyze_series,
    collect_indicator_specs,
)
from neon_radar.domain.funding import FundingRate
from neon_radar.domain.scoring import EMATrendRule, FundingRateRule
from tests.conftest import make_series


class TestCollectIndicatorSpecs:
    def test_uses_rule_instance_parameters(self) -> None:
        specs = collect_indicator_specs((EMATrendRule(fast_period=12, slow_period=26),))

        assert {s.series_name for s in specs} == {"ema_12", "ema_26"}
        assert {s.params["period"] for s in specs} == {12, 26}


class TestAnalyzeSeries:
    def test_full_cycle_with_custom_indicator_periods(self) -> None:
        series = make_series([100.0 + i for i in range(80)])
        result = analyze_series(
            series,
            (EMATrendRule(fast_period=12, slow_period=26),),
            min_confidence=0.0,
        )

        assert result.market_state is not None
        assert {i.name for i in result.market_state.indicator_series} == {
            "ema_12",
            "ema_26",
        }
        assert result.signal_count == 1
        assert result.signals[0].name == "ema_trend"
        assert result.score.value > 0

    def test_rules_without_indicators_still_evaluate(self) -> None:
        series = make_series([100.0] * 10)
        result = analyze_series(
            series,
            (FundingRateRule(),),
            funding_rate=FundingRate(symbol="BTCUSDT", rate=-0.0002),
        )

        assert result.market_state is not None
        assert result.market_state.indicator_series == ()
        assert result.signal_count == 1
        assert result.signals[0].name == "funding_rate"
        assert result.score.value == pytest.approx(result.signals[0].value)

    def test_higher_tf_indicators(self) -> None:
        from neon_radar.application.services.indicator_pipeline import IndicatorSpec
        from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
        from neon_radar.domain.scoring.value_objects import Signal

        class HTFDummyRule(FactorRule):
            NAME = "htf_dummy"

            @classmethod
            def description(cls) -> RuleDescription:
                return RuleDescription(name="htf_dummy", display_name="Dummy", summary="Dummy")

            def required_indicators(self):
                return [
                    IndicatorSpec("sma", {"period": 5}, tag="5"),
                    IndicatorSpec("sma", {"period": 5}, tag="5", target="higher_tf"),
                ]

            def evaluate(self, state):
                return Signal(name="dummy", value=0.5, confidence=1.0)

        from neon_radar.config.models import TimeFrame
        series = make_series([100.0] * 10, timeframe=TimeFrame.H4)
        higher_tf_series = make_series([100.0] * 10, timeframe=TimeFrame.D1)

        result = analyze_series(
            series,
            (HTFDummyRule(),),
            higher_tf_series=higher_tf_series,
        )

        assert result.market_state is not None
        ind_names = {i.name for i in result.market_state.indicator_series}
        assert "sma_5" in ind_names
        assert "htf_sma_5" in ind_names

