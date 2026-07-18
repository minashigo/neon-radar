"""Tests for TradeAnalyzer and statistical validation."""

import pytest

from neon_radar.application.services.trade_analyzer import TradeAnalyzer
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import Symbol
from neon_radar.domain.trading.backtest import Trade, TradeExitReason, TradeStatus


@pytest.fixture
def base_trade():
    """Helper to generate a base trade for testing PnL."""
    return Trade(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BULLISH,
        entry_time=1000,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=110.0,
    )

def test_trade_analyzer_empty():
    analyzer = TradeAnalyzer()
    report = analyzer.analyze([])
    assert report.total_trades == 0
    assert report.gross_expectancy == 0.0
    assert report.net_expectancy == 0.0
    assert report.validation.is_valid is False

def test_trade_analyzer_edge(base_trade):
    """Test a scenario with a clear statistical edge."""
    # Generate 100 trades, 60 wins (10% each), 40 losses (5% each)
    trades = []
    for _ in range(60):
        t = Trade(
            symbol=base_trade.symbol,
            direction=base_trade.direction,
            entry_time=base_trade.entry_time,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            exit_time=1100,
            exit_price=110.0,
            status=TradeStatus.WIN,
            exit_reason=TradeExitReason.TAKE_PROFIT
        )
        trades.append(t)

    for _ in range(40):
        t = Trade(
            symbol=base_trade.symbol,
            direction=base_trade.direction,
            entry_time=base_trade.entry_time,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            exit_time=1100,
            exit_price=95.0,
            status=TradeStatus.LOSS,
            exit_reason=TradeExitReason.STOP_LOSS
        )
        trades.append(t)

    analyzer = TradeAnalyzer()
    report = analyzer.analyze(trades)

    assert report.total_trades == 100
    assert report.win_rate == 0.6
    assert report.wins == 60
    assert report.losses == 40
    assert report.net_expectancy > 0.03  # 0.6 * 0.1 - 0.4 * 0.05 = 0.06 - 0.02 = 0.04
    assert report.gross_expectancy > 0.03

    val = report.validation
    assert val.is_valid
    assert val.p_value < 0.05  # Should have a strong edge
    assert val.t_statistic > 2.0
    assert val.mc_expectancy_95_ci_lower > 0.0  # Lower bound of expectancy CI > 0
    assert val.mc_probability_of_loss < 0.05  # Unlikely to be unprofitable

def test_trade_analyzer_no_edge(base_trade):
    """Test a scenario with no statistical edge (random noise around 0)."""
    # 50 wins (+5%), 50 losses (-5%)
    trades = []
    for _ in range(50):
        t = Trade(
            symbol=base_trade.symbol,
            direction=base_trade.direction,
            entry_time=base_trade.entry_time,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=105.0,
            exit_time=1100,
            exit_price=105.0,
            status=TradeStatus.WIN,
            exit_reason=TradeExitReason.TAKE_PROFIT
        )
        trades.append(t)

    for _ in range(50):
        t = Trade(
            symbol=base_trade.symbol,
            direction=base_trade.direction,
            entry_time=base_trade.entry_time,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=105.0,
            exit_time=1100,
            exit_price=95.0,
            status=TradeStatus.LOSS,
            exit_reason=TradeExitReason.STOP_LOSS
        )
        trades.append(t)

    analyzer = TradeAnalyzer()
    report = analyzer.analyze(trades)

    val = report.validation
    assert val.is_valid
    assert val.p_value >= 0.05  # No edge, should fail reject null hypothesis
    assert val.mc_expectancy_95_ci_lower < 0.0  # CI crosses 0
    assert val.mc_expectancy_95_ci_upper > 0.0
