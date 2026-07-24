import pytest

from neon_radar.application.services.risk.manager import RiskManager, RiskManagerConfig
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import Symbol
from neon_radar.domain.risk import AccountState, DrawdownState, PortfolioState, PositionState
from neon_radar.domain.scoring.value_objects import AnalysisResult, Score


@pytest.fixture
def empty_portfolio():
    return PortfolioState(account=AccountState(10000.0, 10000.0))


@pytest.fixture
def analysis():
    return AnalysisResult(
        score=Score(
            value=1.0, confidence=0.8, long_score=1.0, short_score=0.0, contributing_signals=2
        ),
        signals=tuple(),
        computed_at=0,
        summary="test",
    )


def test_risk_manager_allows_trade(empty_portfolio, analysis):
    rm = RiskManager(RiskManagerConfig())
    decision = rm.evaluate(analysis, empty_portfolio)

    assert decision.is_allowed is True
    assert decision.max_risk_budget == 200.0  # 2% of 10k
    assert decision.max_position_size == 10000.0  # 100% of 10k
    assert decision.risk_penalty_factor == 1.0


def test_risk_manager_rejects_max_open_positions(analysis):
    config = RiskManagerConfig(max_open_positions=1)
    rm = RiskManager(config)

    pos = PositionState(Symbol("ETHUSDT"), Bias.BULLISH, 3000.0, 1.0)
    portfolio = PortfolioState(AccountState(10000.0, 7000.0), positions=(pos,))

    decision = rm.evaluate(analysis, portfolio)
    assert decision.is_allowed is False
    assert "Max open positions reached" in decision.rejection_reason


def test_risk_manager_rejects_duplicate_symbol(analysis):
    class MockSeries:
        symbol = Symbol("BTCUSDT")

    class MockMarketState:
        primary_series = MockSeries()

    analysis_with_symbol = AnalysisResult(
        score=analysis.score,
        signals=analysis.signals,
        computed_at=analysis.computed_at,
        summary=analysis.summary,
        market_state=MockMarketState(),
    )

    rm = RiskManager(RiskManagerConfig())
    pos = PositionState(Symbol("BTCUSDT"), Bias.BULLISH, 50000.0, 0.1)
    portfolio = PortfolioState(AccountState(10000.0, 5000.0), positions=(pos,))

    decision = rm.evaluate(analysis_with_symbol, portfolio)
    assert decision.is_allowed is False
    assert "Position already open" in decision.rejection_reason


def test_risk_manager_rejects_max_exposure(empty_portfolio, analysis):
    config = RiskManagerConfig(max_portfolio_exposure_pct=0.5)
    rm = RiskManager(config)

    pos = PositionState(Symbol("ETHUSDT"), Bias.BULLISH, 3000.0, 2.0)  # Quote size: 6000
    portfolio = PortfolioState(AccountState(10000.0, 4000.0), positions=(pos,))

    decision = rm.evaluate(analysis, portfolio)
    assert decision.is_allowed is False
    assert "Max portfolio exposure reached" in decision.rejection_reason


def test_risk_manager_applies_drawdown_penalty(empty_portfolio, analysis):
    config = RiskManagerConfig(
        max_risk_per_trade_pct=0.02,
        drawdown_penalty_threshold_pct=10.0,
        drawdown_penalty_factor=0.5,
    )
    rm = RiskManager(config)

    # 15% drawdown
    dd = DrawdownState(current_equity=8500.0, ath_equity=10000.0, max_drawdown_pct=15.0)

    decision = rm.evaluate(analysis, empty_portfolio, drawdown=dd)

    assert decision.is_allowed is True
    # Base risk budget is 2% of 10000 = 200, penalized by 0.5 = 100
    assert decision.max_risk_budget == 100.0
    assert decision.risk_penalty_factor == 0.5
