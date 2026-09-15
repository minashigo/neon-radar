"""Unit tests for Portfolio Risk Management Core (Phase 2).

Tests policy configuration, drawdown circuit breaker, portfolio heat,
concurrent trades limits, short constraints, and position sizing integration.
"""

import pytest

from neon_radar.application.services.risk.manager import RiskManager, RiskManagerConfig
from neon_radar.application.services.risk.sizing import FixedRiskStrategy, PositionSizingEngine
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import Symbol
from neon_radar.domain.portfolio import AccountState, OpenPosition, PortfolioState
from neon_radar.domain.risk import (
    DrawdownState,
    DrawdownTier,
    PortfolioRiskPolicy,
    ShortRiskConstraints,
)
from neon_radar.domain.scoring.value_objects import AnalysisResult, Score
from neon_radar.domain.trading.setup import TradeSetup


class MockSeries:
    def __init__(self, symbol_str: str = "BTCUSDT") -> None:
        self.symbol = Symbol(symbol_str)


class MockMarketState:
    def __init__(self, symbol_str: str = "BTCUSDT") -> None:
        self.primary_series = MockSeries(symbol_str)


def make_analysis(
    bias: Bias = Bias.BULLISH,
    symbol_str: str = "BTCUSDT",
    confidence: float = 0.8,
) -> AnalysisResult:
    score_val = 1.0 if bias == Bias.BULLISH else -1.0
    return AnalysisResult(
        score=Score(
            value=score_val,
            confidence=confidence,
            long_score=1.0 if bias == Bias.BULLISH else 0.0,
            short_score=1.0 if bias == Bias.BEARISH else 0.0,
            contributing_signals=2,
        ),
        signals=(),
        computed_at=0,
        summary="test",
        market_state=MockMarketState(symbol_str),
    )


@pytest.fixture
def empty_portfolio() -> PortfolioState:
    return PortfolioState(account=AccountState(total_capital=10000.0, free_capital=10000.0))


# --------------------------------------------------------------------------
# 1. Backward Compatibility & Normal Conditions
# --------------------------------------------------------------------------


def test_default_policy_backward_compatibility(empty_portfolio):
    """Default RiskManager preserves exact legacy RC9.5 baseline behavior."""
    rm = RiskManager()
    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")

    decision = rm.evaluate(analysis, empty_portfolio)
    assert decision.is_allowed is True
    assert decision.max_risk_budget == 200.0  # 2% of 10,000
    assert decision.max_position_size == 10000.0
    assert decision.risk_penalty_factor == 1.0

    # Test legacy drawdown behavior (10% threshold -> 0.5 factor)
    dd_legacy = DrawdownState(current_equity=8500.0, ath_equity=10000.0, max_drawdown_pct=15.0)
    decision_dd = rm.evaluate(analysis, empty_portfolio, drawdown=dd_legacy)
    assert decision_dd.is_allowed is True
    assert decision_dd.risk_penalty_factor == 0.5
    assert decision_dd.max_risk_budget == 100.0


def test_risk_manager_config_adaptation(empty_portfolio):
    """Passing legacy RiskManagerConfig works transparently."""
    cfg = RiskManagerConfig(max_risk_per_trade_pct=0.01, max_open_positions=5)
    rm = RiskManager(cfg)
    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")

    decision = rm.evaluate(analysis, empty_portfolio)
    assert decision.is_allowed is True
    assert decision.max_risk_budget == 100.0
    assert rm.config.max_open_positions == 5


# --------------------------------------------------------------------------
# 2. Drawdown Circuit Breaker
# --------------------------------------------------------------------------


def test_drawdown_circuit_breaker_multi_tier(empty_portfolio):
    """Circuit breaker enforces tiered penalty factors and trading halt."""
    tiers = (
        DrawdownTier(threshold_pct=15.0, risk_multiplier=0.5),
        DrawdownTier(threshold_pct=25.0, risk_multiplier=0.25),
        DrawdownTier(threshold_pct=35.0, risk_multiplier=0.0, is_trading_halt=True),
    )
    policy = PortfolioRiskPolicy(
        base_risk_per_trade_pct=0.02,
        enable_circuit_breaker=True,
        drawdown_tiers=tiers,
    )
    rm = RiskManager(policy)
    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")

    # Tier 0: Normal (< 15%)
    dd_normal = DrawdownState(current_equity=9000.0, ath_equity=10000.0, max_drawdown_pct=10.0)
    d0 = rm.evaluate(analysis, empty_portfolio, drawdown=dd_normal)
    assert d0.is_allowed is True
    assert d0.risk_penalty_factor == 1.0
    assert d0.max_risk_budget == 200.0

    # Tier 1: Caution (15% <= DD < 25%) -> multiplier 0.5
    dd_t1 = DrawdownState(current_equity=8200.0, ath_equity=10000.0, max_drawdown_pct=18.0)
    d1 = rm.evaluate(analysis, empty_portfolio, drawdown=dd_t1)
    assert d1.is_allowed is True
    assert d1.risk_penalty_factor == 0.5
    assert d1.max_risk_budget == 100.0

    # Tier 2: Defensive (25% <= DD < 35%) -> multiplier 0.25
    dd_t2 = DrawdownState(current_equity=7200.0, ath_equity=10000.0, max_drawdown_pct=28.0)
    d2 = rm.evaluate(analysis, empty_portfolio, drawdown=dd_t2)
    assert d2.is_allowed is True
    assert d2.risk_penalty_factor == 0.25
    assert d2.max_risk_budget == 50.0

    # Tier 3: Trading Halt (DD >= 35%) -> is_allowed = False
    dd_halt = DrawdownState(current_equity=6000.0, ath_equity=10000.0, max_drawdown_pct=40.0)
    d3 = rm.evaluate(analysis, empty_portfolio, drawdown=dd_halt)
    assert d3.is_allowed is False
    assert d3.risk_penalty_factor == 0.0
    assert "Trading halted by drawdown circuit breaker" in d3.rejection_reason


def test_circuit_breaker_ath_reset(empty_portfolio):
    """When equity reaches new ATH, DD resets to 0 and breaker resets to 1.0."""
    tiers = (DrawdownTier(threshold_pct=15.0, risk_multiplier=0.5),)
    policy = PortfolioRiskPolicy(enable_circuit_breaker=True, drawdown_tiers=tiers)
    rm = RiskManager(policy)
    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")

    # New ATH reached
    dd_ath = DrawdownState(current_equity=12000.0, ath_equity=12000.0, max_drawdown_pct=15.0)
    decision = rm.evaluate(analysis, empty_portfolio, drawdown=dd_ath)
    assert decision.is_allowed is True
    assert decision.risk_penalty_factor == 1.0
    assert decision.max_risk_budget == 200.0


# --------------------------------------------------------------------------
# 3. Maximum Concurrent Trades & Duplicate Check
# --------------------------------------------------------------------------


def test_max_concurrent_trades_gate():
    """Reject candidate setup when concurrent trade limit is reached."""
    policy = PortfolioRiskPolicy(max_concurrent_trades=2)
    rm = RiskManager(policy)

    acc = AccountState(total_capital=10000.0, free_capital=6000.0)
    pos1 = OpenPosition(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BULLISH,
        entry_price=30000.0,
        quantity=0.1,
        position_size=3000.0,
        stop_loss=28000.0,
        take_profit=34000.0,
        opened_at=1000,
    )
    pos2 = OpenPosition(
        symbol=Symbol("ETHUSDT"),
        direction=Bias.BULLISH,
        entry_price=2000.0,
        quantity=1.0,
        position_size=2000.0,
        stop_loss=1900.0,
        take_profit=2200.0,
        opened_at=1000,
    )

    portfolio_full = PortfolioState(account=acc, positions=(pos1, pos2))
    analysis_sol = make_analysis(Bias.BULLISH, "SOLUSDT")

    decision = rm.evaluate(analysis_sol, portfolio_full)
    assert decision.is_allowed is False
    assert "Max open positions reached (2)" in decision.rejection_reason

    # One position closed -> trade allowed
    portfolio_open = PortfolioState(account=acc, positions=(pos1,))
    decision_ok = rm.evaluate(analysis_sol, portfolio_open)
    assert decision_ok.is_allowed is True


def test_duplicate_symbol_rejection():
    """Reject candidate trade if symbol already has an open position."""
    rm = RiskManager()
    acc = AccountState(total_capital=10000.0, free_capital=7000.0)
    pos = OpenPosition(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BULLISH,
        entry_price=30000.0,
        quantity=0.1,
        position_size=3000.0,
        stop_loss=28000.0,
        take_profit=34000.0,
        opened_at=1000,
    )
    portfolio = PortfolioState(account=acc, positions=(pos,))
    analysis_btc = make_analysis(Bias.BULLISH, "BTCUSDT")

    decision = rm.evaluate(analysis_btc, portfolio)
    assert decision.is_allowed is False
    assert "Position already open for BTCUSDT" in decision.rejection_reason


# --------------------------------------------------------------------------
# 4. Maximum Portfolio Heat
# --------------------------------------------------------------------------


def test_portfolio_heat_sufficient_room():
    """When remaining heat exceeds requested risk, trade is allowed with full budget."""
    policy = PortfolioRiskPolicy(
        base_risk_per_trade_pct=0.02,
        enable_portfolio_heat=True,
        max_portfolio_heat_pct=0.06,  # $600 max heat on $10,000
    )
    rm = RiskManager(policy)

    acc = AccountState(total_capital=10000.0, free_capital=8000.0)
    # Existing position risks: |30000 - 28000| * 0.1 = $200
    pos1 = OpenPosition(
        symbol=Symbol("ETHUSDT"),
        direction=Bias.BULLISH,
        entry_price=30000.0,
        quantity=0.1,
        position_size=3000.0,
        stop_loss=28000.0,
        take_profit=34000.0,
        opened_at=1000,
    )
    portfolio = PortfolioState(account=acc, positions=(pos1,))
    assert portfolio.total_risk == 200.0

    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")
    # Candidate risk budget = $200. Remaining heat = 600 - 200 = 400.
    decision = rm.evaluate(analysis, portfolio)
    assert decision.is_allowed is True
    assert decision.max_risk_budget == 200.0


def test_portfolio_heat_clipped_to_remaining_budget():
    """When candidate risk exceeds remaining heat, budget is clipped to available heat."""
    policy = PortfolioRiskPolicy(
        base_risk_per_trade_pct=0.02,
        enable_portfolio_heat=True,
        max_portfolio_heat_pct=0.06,  # $600 max heat on $10,000
        min_trade_risk_budget=1.0,
    )
    rm = RiskManager(policy)

    acc = AccountState(total_capital=10000.0, free_capital=8000.0)
    # Existing positions risk $500 total
    pos1 = OpenPosition(
        symbol=Symbol("ETHUSDT"),
        direction=Bias.BULLISH,
        entry_price=100.0,
        quantity=5.0,
        position_size=500.0,
        stop_loss=0.0,  # risk = 100 * 5 = 500
        take_profit=200.0,
        opened_at=1000,
    )
    portfolio = PortfolioState(account=acc, positions=(pos1,))
    assert portfolio.total_risk == 500.0

    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")
    # Base risk = $200, but remaining heat = 600 - 500 = 100.
    decision = rm.evaluate(analysis, portfolio)
    assert decision.is_allowed is True
    assert decision.max_risk_budget == 100.0  # Clipped to remaining heat!


def test_portfolio_heat_exceeded_rejection():
    """When portfolio heat is fully exhausted, candidate trade is rejected."""
    policy = PortfolioRiskPolicy(
        enable_portfolio_heat=True,
        max_portfolio_heat_pct=0.06,  # $600 max heat
    )
    rm = RiskManager(policy)

    acc = AccountState(total_capital=10000.0, free_capital=8000.0)
    pos1 = OpenPosition(
        symbol=Symbol("ETHUSDT"),
        direction=Bias.BULLISH,
        entry_price=100.0,
        quantity=6.0,
        position_size=600.0,
        stop_loss=0.0,  # risk = 600
        take_profit=200.0,
        opened_at=1000,
    )
    portfolio = PortfolioState(account=acc, positions=(pos1,))
    assert portfolio.total_risk == 600.0

    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")
    decision = rm.evaluate(analysis, portfolio)
    assert decision.is_allowed is False
    assert "Max portfolio heat reached" in decision.rejection_reason


def test_portfolio_heat_insufficient_for_min_trade():
    """When remaining heat is below min_trade_risk_budget, trade is rejected."""
    policy = PortfolioRiskPolicy(
        enable_portfolio_heat=True,
        max_portfolio_heat_pct=0.06,
        min_trade_risk_budget=5.0,  # require at least $5
    )
    rm = RiskManager(policy)

    acc = AccountState(total_capital=10000.0, free_capital=8000.0)
    pos1 = OpenPosition(
        symbol=Symbol("ETHUSDT"),
        direction=Bias.BULLISH,
        entry_price=100.0,
        quantity=5.98,
        position_size=598.0,
        stop_loss=0.0,  # risk = 598.0 -> remaining heat = 2.0 < 5.0
        take_profit=200.0,
        opened_at=1000,
    )
    portfolio = PortfolioState(account=acc, positions=(pos1,))
    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")

    decision = rm.evaluate(analysis, portfolio)
    assert decision.is_allowed is False
    assert "Insufficient portfolio heat budget" in decision.rejection_reason


def test_portfolio_heat_release_on_breakeven():
    """Positions moved to breakeven have max_risk = 0, releasing portfolio heat."""
    policy = PortfolioRiskPolicy(
        base_risk_per_trade_pct=0.02,
        enable_portfolio_heat=True,
        max_portfolio_heat_pct=0.02,  # exactly $200 max heat
    )
    rm = RiskManager(policy)

    acc = AccountState(total_capital=10000.0, free_capital=8000.0)
    # Stop loss moved to entry price (breakeven)
    pos1 = OpenPosition(
        symbol=Symbol("ETHUSDT"),
        direction=Bias.BULLISH,
        entry_price=2000.0,
        quantity=1.0,
        position_size=2000.0,
        stop_loss=2000.0,  # entry == SL -> max_risk = 0.0
        take_profit=2500.0,
        opened_at=1000,
    )
    portfolio = PortfolioState(account=acc, positions=(pos1,))
    assert portfolio.total_risk == 0.0

    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")
    decision = rm.evaluate(analysis, portfolio)
    assert decision.is_allowed is True
    assert decision.max_risk_budget == 200.0


# --------------------------------------------------------------------------
# 5. Short Constraints
# --------------------------------------------------------------------------


def test_short_constraints_when_enabled(empty_portfolio):
    """Short-specific limits and risk scaling take effect only when enabled."""
    short_cfg = ShortRiskConstraints(
        enabled=True,
        risk_multiplier=0.5,
        max_concurrent_shorts=1,
        allow_in_drawdown=False,
    )
    policy = PortfolioRiskPolicy(
        base_risk_per_trade_pct=0.02,
        short_constraints=short_cfg,
    )
    rm = RiskManager(policy)

    analysis_short = make_analysis(Bias.BEARISH, "BTCUSDT")
    # 1. Scaled budget: 200 * 0.5 = 100
    d_short = rm.evaluate(analysis_short, empty_portfolio)
    assert d_short.is_allowed is True
    assert d_short.max_risk_budget == 100.0

    # 2. Max concurrent shorts: if 1 short is already open, 2nd short rejected
    acc = AccountState(total_capital=10000.0, free_capital=8000.0)
    short_pos = OpenPosition(
        symbol=Symbol("ETHUSDT"),
        direction=Bias.BEARISH,
        entry_price=2000.0,
        quantity=1.0,
        position_size=2000.0,
        stop_loss=2100.0,
        take_profit=1800.0,
        opened_at=1000,
    )
    port_with_short = PortfolioState(account=acc, positions=(short_pos,))

    d_second_short = rm.evaluate(analysis_short, port_with_short)
    assert d_second_short.is_allowed is False
    assert "Max concurrent shorts reached (1)" in d_second_short.rejection_reason

    # Long trade is still allowed!
    analysis_long = make_analysis(Bias.BULLISH, "SOLUSDT")
    d_long = rm.evaluate(analysis_long, port_with_short)
    assert d_long.is_allowed is True
    assert d_long.max_risk_budget == 200.0

    # 3. Drawdown restriction: short rejected during active drawdown
    dd = DrawdownState(current_equity=8500.0, ath_equity=10000.0, max_drawdown_pct=15.0)
    d_dd_short = rm.evaluate(analysis_short, empty_portfolio, drawdown=dd)
    assert d_dd_short.is_allowed is False
    assert "Short positions disabled during drawdown" in d_dd_short.rejection_reason


# --------------------------------------------------------------------------
# 6. Invalid Portfolio State Handling
# --------------------------------------------------------------------------


def test_invalid_portfolio_state_handling():
    """Missing or corrupted portfolio states are safely rejected."""
    rm = RiskManager()
    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")

    # None portfolio
    d_none = rm.evaluate(analysis, None)
    assert d_none.is_allowed is False
    assert "Missing portfolio state" in d_none.rejection_reason

    # Non-positive capital
    class CorruptAccount:
        total_capital = 0.0
        free_capital = 0.0

    class CorruptPortfolio:
        account = CorruptAccount()
        positions = ()

    d_corrupt = rm.evaluate(analysis, CorruptPortfolio())
    assert d_corrupt.is_allowed is False
    assert "Invalid account capital" in d_corrupt.rejection_reason


# --------------------------------------------------------------------------
# 7. Position Sizing Engine Integration
# --------------------------------------------------------------------------


def test_position_sizing_engine_respects_circuit_breaker(empty_portfolio):
    """PositionSizingEngine shrinks position size when circuit breaker scales risk."""
    tiers = (DrawdownTier(threshold_pct=15.0, risk_multiplier=0.5),)
    policy = PortfolioRiskPolicy(
        base_risk_per_trade_pct=0.02,
        enable_circuit_breaker=True,
        drawdown_tiers=tiers,
    )
    rm = RiskManager(policy)
    sizing_engine = PositionSizingEngine(FixedRiskStrategy())

    setup = TradeSetup(
        direction=Bias.BULLISH,
        entry_price=100.0,
        stop_loss=95.0,  # sl_distance = 5.0
        take_profit_1=110.0,
        take_profit_2=120.0,
        risk_reward=(2.0, 4.0),
    )
    analysis = make_analysis(Bias.BULLISH, "BTCUSDT")

    # Normal conditions: Budget = $200. Base size = 200 / 5 = 40. Quote size = 40 * 100 = 4000.
    d_normal = rm.evaluate(analysis, empty_portfolio)
    sized_normal = sizing_engine.build_sized_setup(setup, d_normal)
    assert sized_normal is not None
    assert sized_normal.base_size == 40.0
    assert sized_normal.quote_size == 4000.0

    # Drawdown Tier 1 (18% DD): Budget = $100. Base size = 100 / 5 = 20. Quote size = 20 * 100 = 2000.
    dd_t1 = DrawdownState(current_equity=8200.0, ath_equity=10000.0, max_drawdown_pct=18.0)
    d_t1 = rm.evaluate(analysis, empty_portfolio, drawdown=dd_t1)
    sized_t1 = sizing_engine.build_sized_setup(setup, d_t1)
    assert sized_t1 is not None
    assert sized_t1.base_size == 20.0
    assert sized_t1.quote_size == 2000.0


# --------------------------------------------------------------------------
# 8. Configuration Models (Pydantic)
# --------------------------------------------------------------------------


def test_risk_policy_config_defaults():
    """Verify default values and domain mapping of RiskPolicyConfig."""
    from neon_radar.config.risk import RiskPolicyConfig

    cfg = RiskPolicyConfig()
    domain_policy = cfg.to_domain()

    assert domain_policy.base_risk_per_trade_pct == 0.02
    assert domain_policy.max_concurrent_trades == 3
    assert domain_policy.max_portfolio_exposure_pct == 1.0
    assert domain_policy.enable_circuit_breaker is False
    assert domain_policy.enable_portfolio_heat is False
    assert domain_policy.short_constraints.enabled is False


def test_risk_policy_config_custom_to_domain():
    """Verify custom config maps cleanly to domain models."""
    from neon_radar.config.risk import DrawdownTierConfig, RiskPolicyConfig, ShortRiskConfig

    cfg = RiskPolicyConfig(
        base_risk_per_trade_pct=0.015,
        max_concurrent_trades=4,
        enable_circuit_breaker=True,
        drawdown_tiers=[
            DrawdownTierConfig(threshold_pct=15.0, risk_multiplier=0.5),
            DrawdownTierConfig(threshold_pct=30.0, risk_multiplier=0.0, is_trading_halt=True),
        ],
        enable_portfolio_heat=True,
        max_portfolio_heat_pct=0.05,
        short_constraints=ShortRiskConfig(
            enabled=True,
            risk_multiplier=0.5,
            max_concurrent_shorts=1,
            allow_in_drawdown=False,
        ),
    )
    policy = cfg.to_domain()

    assert policy.base_risk_per_trade_pct == 0.015
    assert policy.max_concurrent_trades == 4
    assert policy.enable_circuit_breaker is True
    assert len(policy.drawdown_tiers) == 2
    assert policy.drawdown_tiers[0].threshold_pct == 15.0
    assert policy.drawdown_tiers[1].is_trading_halt is True
    assert policy.enable_portfolio_heat is True
    assert policy.max_portfolio_heat_pct == 0.05
    assert policy.short_constraints.enabled is True
    assert policy.short_constraints.max_concurrent_shorts == 1
    assert policy.short_constraints.allow_in_drawdown is False

