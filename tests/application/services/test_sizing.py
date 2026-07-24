import pytest

from neon_radar.application.services.risk.sizing import (
    ATRBasedStrategy,
    FixedRiskStrategy,
    FixedSizeStrategy,
    PositionSizingEngine,
    SizedTradeSetup,
)
from neon_radar.domain.enums import Bias
from neon_radar.domain.risk import RiskDecision
from neon_radar.domain.trading.setup import TradeDiagnostics, TradeSetup


@pytest.fixture
def base_setup():
    return TradeSetup(
        direction=Bias.BULLISH,
        entry_price=50000.0,
        stop_loss=45000.0,
        take_profit_1=55000.0,
        take_profit_2=60000.0,
        risk_reward=(1.0, 2.0),
        diagnostics=TradeDiagnostics(
            atr=5000.0,
            adx=25.0,
            rsi=60.0,
            ema_spread_pct=0.0,
            htf_trend=1.0,
            confidence=0.8,
            final_score=1.0,
            triggered_rules="",
            entry_reason="",
            regime="bullish",
            regime_reason="",
        ),
    )


@pytest.fixture
def decision_allow():
    return RiskDecision(
        is_allowed=True, max_risk_budget=500.0, max_position_size=10000.0, risk_penalty_factor=1.0
    )


@pytest.fixture
def decision_deny():
    return RiskDecision(
        is_allowed=False, rejection_reason="Test denial", max_risk_budget=0.0, max_position_size=0.0
    )


def test_fixed_size_strategy(base_setup, decision_allow):
    strategy = FixedSizeStrategy(fixed_quote_amount=2000.0)
    engine = PositionSizingEngine(strategy)

    sized = engine.build_sized_setup(base_setup, decision_allow)

    assert isinstance(sized, SizedTradeSetup)
    assert sized.quote_size == 2000.0
    assert sized.base_size == 2000.0 / 50000.0


def test_fixed_size_capped_by_manager(base_setup, decision_allow):
    # Setup decision with smaller max_position_size
    decision = RiskDecision(is_allowed=True, max_risk_budget=500, max_position_size=1000)

    strategy = FixedSizeStrategy(fixed_quote_amount=2000.0)
    engine = PositionSizingEngine(strategy)

    sized = engine.build_sized_setup(base_setup, decision)
    assert sized.quote_size == 1000.0


def test_fixed_risk_strategy(base_setup, decision_allow):
    strategy = FixedRiskStrategy()
    engine = PositionSizingEngine(strategy)

    sized = engine.build_sized_setup(base_setup, decision_allow)

    # Risk budget = 500. SL distance = 5000. Base size = 500/5000 = 0.1
    # Quote size = 0.1 * 50000 = 5000.0
    assert sized.quote_size == 5000.0
    assert sized.base_size == 0.1


def test_atr_based_strategy(base_setup, decision_allow):
    strategy = ATRBasedStrategy(atr_multiplier=2.0)
    engine = PositionSizingEngine(strategy)

    sized = engine.build_sized_setup(base_setup, decision_allow)

    # Risk budget = 500. ATR = 5000. Multiplier = 2.0 -> Risk distance = 10000.
    # Base size = 500 / 10000 = 0.05
    # Quote size = 0.05 * 50000 = 2500.0
    assert sized.quote_size == 2500.0
    assert sized.base_size == 0.05


def test_engine_rejects_denied_decision(base_setup, decision_deny):
    strategy = FixedSizeStrategy(2000.0)
    engine = PositionSizingEngine(strategy)

    assert engine.build_sized_setup(base_setup, decision_deny) is None
