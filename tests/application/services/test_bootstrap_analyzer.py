"""Tests for the Bootstrap Analyzer."""

from unittest.mock import MagicMock

import pytest

from neon_radar.application.services.bootstrap_analyzer import BootstrapAnalyzer
from neon_radar.domain.trading.backtest import BacktestReport, Trade
from neon_radar.domain.trading.bootstrap import BootstrapReport


def _create_mock_trade(pnl: float) -> Trade:
    mock_trade = MagicMock(spec=Trade)
    mock_trade.net_pnl_pct = pnl
    return mock_trade


def _mock_metric_calculator(trades):
    # Calculate simple stats for tests
    wins = [t for t in trades if t.net_pnl_pct > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    expectancy = sum(t.net_pnl_pct for t in trades) / len(trades) if trades else 0.0

    mock_report = MagicMock(spec=BacktestReport)
    mock_report.net_profit_factor = 1.5
    mock_report.net_expectancy = expectancy
    mock_report.win_rate = win_rate
    mock_report.net_sharpe_ratio = 1.0
    mock_report.max_drawdown_pct = 0.2
    return mock_report


def test_bootstrap_analyzer_returns_none_if_no_trades():
    analyzer = BootstrapAnalyzer(_mock_metric_calculator)
    assert analyzer.run([]) is None


def test_bootstrap_analyzer_seed_reproducibility():
    analyzer = BootstrapAnalyzer(_mock_metric_calculator)

    # Create 50 dummy trades
    trades = [_create_mock_trade((i % 5) - 2) for i in range(50)]

    report1 = analyzer.run(trades, block_size=10, iterations=10, random_seed=42)
    report2 = analyzer.run(trades, block_size=10, iterations=10, random_seed=42)

    # Both should be exactly the same due to identical seed
    assert report1.iterations == report2.iterations
    assert report1.metrics["Win Rate (%)"].mean == pytest.approx(report2.metrics["Win Rate (%)"].mean)
    assert report1.metrics["Expectancy (%)"].mean == pytest.approx(report2.metrics["Expectancy (%)"].mean)


def test_bootstrap_analyzer_structure():
    analyzer = BootstrapAnalyzer(_mock_metric_calculator)
    trades = [_create_mock_trade((i % 5) - 2) for i in range(50)]

    report = analyzer.run(trades, block_size=10, iterations=50, random_seed=42)

    assert isinstance(report, BootstrapReport)
    assert "Win Rate (%)" in report.metrics
    assert "Expectancy (%)" in report.metrics
    assert "Profit Factor" in report.metrics
    assert "Sharpe Ratio" in report.metrics
    assert "Max Drawdown (%)" in report.metrics

    dist = report.metrics["Max Drawdown (%)"]
    assert dist.mean == pytest.approx(20.0) # 0.2 * 100
    assert len(dist.values) == 50
    assert dist.ci_lower_95 <= dist.ci_upper_95
