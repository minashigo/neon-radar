"""Tests for FeatureImportanceAnalyzer."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from neon_radar.application.services.feature_analyzer import FeatureImportanceAnalyzer
from neon_radar.application.services.trade_analyzer import TradeAnalyzer
from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.domain.models import Symbol
from neon_radar.domain.scoring.factor_rule import FactorRule
from neon_radar.domain.trading.backtest import BacktestReport, StatisticalValidationReport


@pytest.fixture
def mock_rule_a():
    rule = MagicMock(spec=FactorRule)
    rule.name = "RuleA"
    return rule

@pytest.fixture
def mock_rule_b():
    rule = MagicMock(spec=FactorRule)
    rule.name = "RuleB"
    return rule

def _make_report(pf, exp, sharpe, wr, prob_loss, pval):
    return BacktestReport(
        total_trades=100,
        win_rate=wr,
        wins=int(100 * wr),
        losses=100 - int(100 * wr),
        avg_win_pct=0.05,
        avg_loss_pct=0.05,
        profit_factor=pf,
        expectancy=exp,
        sharpe_ratio=sharpe,
        max_consecutive_wins=5,
        max_consecutive_losses=5,
        avg_holding_time_ms=1000.0,
        validation=StatisticalValidationReport(True, pval, 2.0, 0.01, 0.05, prob_loss),
        trades=(),
    )

@pytest.mark.asyncio
async def test_feature_importance_analyzer(mock_rule_a, mock_rule_b):
    # 1. Setup mocks
    analyzer_mock = MagicMock(spec=TradeAnalyzer)

    baseline_tester = MagicMock(spec=TradeBacktester)
    baseline_tester._exchange = MagicMock()
    baseline_tester._scoring_config = MagicMock()
    baseline_tester._rules = (mock_rule_a, mock_rule_b)
    baseline_tester.cache = {}

    # We will mock TradeBacktester's run method to return dummy trades.
    # The actual reports will be returned by analyzer.analyze.
    # We can patch TradeBacktester inside the service or just make the analyzer return specific reports
    # based on the number of calls.

    # baseline call, ablated Rule A call, ablated Rule B call
    analyzer_mock.analyze.side_effect = [
        _make_report(pf=1.5, exp=0.05, sharpe=1.0, wr=0.5, prob_loss=0.1, pval=0.05),  # Baseline
        _make_report(pf=1.0, exp=0.00, sharpe=0.0, wr=0.4, prob_loss=0.4, pval=0.5),   # Without Rule A (Worse -> A is good)
        _make_report(pf=2.0, exp=0.10, sharpe=2.0, wr=0.6, prob_loss=0.0, pval=0.01),  # Without Rule B (Better -> B is bad)
    ]

    # Mock TradeBacktester constructor to return a mock tester
    with pytest.MonkeyPatch.context() as m:
        tester_instance_mock = MagicMock(spec=TradeBacktester)
        tester_instance_mock.run = AsyncMock(return_value=())
        m.setattr("neon_radar.application.services.feature_analyzer.TradeBacktester", lambda **kwargs: tester_instance_mock)

        baseline_tester.run = AsyncMock(return_value=())

        feature_analyzer = FeatureImportanceAnalyzer(analyzer_mock)
        report = await feature_analyzer.analyze(
            baseline_tester=baseline_tester,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            symbols=[Symbol("BTCUSDT")],
        )

        assert report.baseline.profit_factor == 1.5
        assert len(report.features) == 2

        # Rule A was good. Removing it made PF drop from 1.5 to 1.0. Delta = 1.5 - 1.0 = +0.5
        # Exp: 0.05 - 0.00 = +0.05
        # Sharpe: 1.0 - 0.0 = +1.0
        # WR: 0.5 - 0.4 = +0.1
        # Prob Loss: 0.4 - 0.1 = +0.3 (Wait, baseline is 0.1, ablated is 0.4. delta_prob_loss = a - b = 0.4 - 0.1 = 0.3)
        # Pval: a_pval - b_pval = 0.5 - 0.05 = +0.45

        # Check rule A metrics
        rule_a_metrics = next(f for f in report.features if f.rule_name == "RuleA")
        assert rule_a_metrics.delta_profit_factor == pytest.approx(0.5)
        assert rule_a_metrics.delta_expectancy == pytest.approx(0.05)
        assert rule_a_metrics.delta_sharpe_ratio == pytest.approx(1.0)
        assert rule_a_metrics.delta_win_rate == pytest.approx(0.1)
        assert rule_a_metrics.delta_probability_of_loss == pytest.approx(0.3)
        assert rule_a_metrics.feature_score > 0  # Should be highly positive

        # Check rule B metrics (removing it made things better, so delta is negative)
        rule_b_metrics = next(f for f in report.features if f.rule_name == "RuleB")
        assert rule_b_metrics.delta_profit_factor == pytest.approx(-0.5)
        assert rule_b_metrics.delta_expectancy == pytest.approx(-0.05)
        assert rule_b_metrics.delta_sharpe_ratio == pytest.approx(-1.0)
        assert rule_b_metrics.delta_win_rate == pytest.approx(-0.1)
        assert rule_b_metrics.delta_probability_of_loss == pytest.approx(-0.1)
        assert rule_b_metrics.feature_score < 0  # Should be negative

        # Ensure ranking is sorted correctly (A > B)
        assert report.features[0].rule_name == "RuleA"
        assert report.features[1].rule_name == "RuleB"
