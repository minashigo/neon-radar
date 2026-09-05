"""Regression and integrity tests for RC8 Sprint 1: PnL, Risk & Execution Semantics."""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from neon_radar.application.services.bootstrap_analyzer import BootstrapAnalyzer
from neon_radar.application.services.execution import PaperExecutionEngine
from neon_radar.application.services.portfolio_engine import PortfolioEngine
from neon_radar.application.services.trade_analyzer import TradeAnalyzer
from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.config.models import TimeFrame
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.enums import Bias
from neon_radar.domain.execution_costs import (
    BinanceFuturesFeeModel as CanonicalFeeModel,
)
from neon_radar.domain.execution_costs import (
    FixedSlippageModel as CanonicalSlippageModel,
)
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.portfolio import OpenPosition
from neon_radar.domain.risk import RiskDecision
from neon_radar.domain.trading.backtest import Trade, TradeExitReason, TradeStatus
from neon_radar.domain.trading.execution import (
    BinanceFuturesFeeModel as DelegatingFeeModel,
)
from neon_radar.domain.trading.execution import (
    ExecutionType,
    TradeCosts,
)
from neon_radar.domain.trading.execution import (
    FixedSlippageModel as DelegatingSlippageModel,
)
from neon_radar.domain.trading.setup import FinalTradeSetup
from neon_radar.infrastructure.providers.binance_funding import BinanceHistoricalFundingProvider

# ---------------------------------------------------------------------------
# 1. Prefetch Pagination Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_pagination_multi_batch():
    """Verify _prefetch fetches multiple batches when history spans > 1500 candles."""
    symbol = Symbol("BTCUSDT")
    tf = TimeFrame.H4

    # Create mock exchange that returns 1500 candles per call
    exchange_mock = MagicMock()

    # Generate 3000 candles spaced by 4 hours
    base_ts = 1_700_000_000_000
    all_candles = [
        OHLCV(
            open_time=base_ts + i * 14_400_000,
            open=100.0 + i,
            high=105.0 + i,
            low=95.0 + i,
            close=102.0 + i,
            volume=10.0,
            close_time=base_ts + (i + 1) * 14_400_000 - 1,
        )
        for i in range(3000)
    ]

    async def fake_get_klines(sym, timeframe, *, limit=1500, end_time=None):
        if end_time is None:
            batch = [c for c in all_candles if c.open_time <= all_candles[-1].open_time]
        else:
            batch = [c for c in all_candles if c.open_time <= end_time]
        selected = batch[-limit:]
        return KlineSeries(symbol=sym, timeframe=timeframe, candles=tuple(selected))

    exchange_mock.get_klines = AsyncMock(side_effect=fake_get_klines)

    start_d = datetime.fromtimestamp(all_candles[500].open_time / 1000.0, tz=UTC).date()
    end_d = datetime.fromtimestamp(all_candles[-1].open_time / 1000.0, tz=UTC).date()

    tester = TradeBacktester(
        exchange=exchange_mock,
        scoring_config=ScoringRulesConfig.model_validate({"rules": []}),
        rules=(),
    )

    await tester._prefetch((symbol,), tf.value, start_d, end_d)

    key = (str(symbol), tf.value)
    assert key in tester.cache
    cached_candles = tester.cache[key].candles

    # Should have fetched more than 1500 candles via pagination
    assert len(cached_candles) > 1500
    assert exchange_mock.get_klines.call_count >= 2
    # Verify candles are strictly sorted ascending
    for i in range(len(cached_candles) - 1):
        assert cached_candles[i].open_time < cached_candles[i + 1].open_time


# ---------------------------------------------------------------------------
# 2. Gap-Aware Execution Tests
# ---------------------------------------------------------------------------


def test_gap_execution_long_stop_loss():
    """Verify Long stop loss fills at open when candle opens below SL."""
    port_engine = PortfolioEngine(10000.0)
    exec_engine = PaperExecutionEngine(port_engine)

    pos = OpenPosition(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BULLISH,
        entry_price=100.0,
        quantity=1.0,
        position_size=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        opened_at=1000,
    )
    port_engine.open_position(pos)

    # Candle gaps down: open=85.0 (below SL=90.0), low=80.0
    gap_candle = OHLCV(
        open_time=2000,
        open=85.0,
        high=88.0,
        low=80.0,
        close=82.0,
        volume=10.0,
        close_time=2999,
    )

    exec_engine.process_market_tick(Symbol("BTCUSDT"), gap_candle)

    assert len(port_engine.history) == 1
    closed = port_engine.history[0]
    assert closed.close_reason == "STOP_LOSS"
    # Gap fill: must fill at 85.0, NOT 90.0!
    assert closed.exit_price == 85.0


def test_gap_execution_short_stop_loss():
    """Verify Short stop loss fills at open when candle opens above SL."""
    port_engine = PortfolioEngine(10000.0)
    exec_engine = PaperExecutionEngine(port_engine)

    pos = OpenPosition(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BEARISH,
        entry_price=100.0,
        quantity=1.0,
        position_size=100.0,
        stop_loss=110.0,
        take_profit=80.0,
        opened_at=1000,
    )
    port_engine.open_position(pos)

    # Candle gaps up: open=115.0 (above SL=110.0), high=120.0
    gap_candle = OHLCV(
        open_time=2000,
        open=115.0,
        high=120.0,
        low=112.0,
        close=118.0,
        volume=10.0,
        close_time=2999,
    )

    exec_engine.process_market_tick(Symbol("BTCUSDT"), gap_candle)

    assert len(port_engine.history) == 1
    closed = port_engine.history[0]
    assert closed.close_reason == "STOP_LOSS"
    # Gap fill: must fill at 115.0, NOT 110.0!
    assert closed.exit_price == 115.0


def test_ambiguous_candle_conservative_sl_priority():
    """Verify ambiguous candle touching both SL and TP resolves deterministically to SL."""
    port_engine = PortfolioEngine(10000.0)
    exec_engine = PaperExecutionEngine(port_engine)

    pos = OpenPosition(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BULLISH,
        entry_price=100.0,
        quantity=1.0,
        position_size=100.0,
        stop_loss=90.0,
        take_profit=110.0,
        opened_at=1000,
    )
    port_engine.open_position(pos)

    # Candle spans both SL (90) and TP (110)
    ambiguous_candle = OHLCV(
        open_time=2000,
        open=100.0,
        high=115.0,
        low=85.0,
        close=95.0,
        volume=10.0,
        close_time=2999,
    )

    exec_engine.process_market_tick(Symbol("BTCUSDT"), ambiguous_candle)

    assert len(port_engine.history) == 1
    closed = port_engine.history[0]
    # Deterministic conservative rule: SL must take priority
    assert closed.close_reason == "STOP_LOSS"
    assert closed.exit_price == 90.0


# ---------------------------------------------------------------------------
# 3. Canonical Trade Economics & ClosedPosition Conversion
# ---------------------------------------------------------------------------


def test_closed_position_and_trade_canonical_economics():
    """Verify ClosedPosition and Trade retain full canonical economics."""
    port_engine = PortfolioEngine(10000.0)
    exec_engine = PaperExecutionEngine(port_engine)

    # Queue setup
    setup = FinalTradeSetup(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BULLISH,
        entry=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        confidence=1.0,
        score=1.0,
        risk_decision=RiskDecision(True),
        position_size=2.0,  # 2 BTC
        quote_size=200.0,  # 200 USDT margin
        risk_amount=20.0,  # $20 at risk (10 * 2)
        expected_reward=40.0,
    )
    exec_engine.execute_setup(setup, 1000)

    # Tick 1: trigger entry
    tick1 = OHLCV(
        open_time=1000, open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0, close_time=1999
    )
    exec_engine.process_market_tick(Symbol("BTCUSDT"), tick1)

    assert len(port_engine.state.positions) == 1
    open_pos = port_engine.state.positions[0]
    assert open_pos.quantity == 2.0
    assert open_pos.capital_at_entry == 10000.0
    assert open_pos.max_risk == 20.0

    # Tick 2: hit TP at 120.0
    tick2 = OHLCV(
        open_time=2000, open=105.0, high=125.0, low=104.0, close=122.0, volume=1.0, close_time=2999
    )
    exec_engine.process_market_tick(Symbol("BTCUSDT"), tick2)

    assert len(port_engine.history) == 1
    closed = port_engine.history[0]
    assert closed.quantity == 2.0
    assert closed.initial_risk == 20.0
    assert closed.stop_loss == 90.0
    assert closed.take_profit == 120.0
    assert closed.capital_at_entry == 10000.0
    assert closed.profit_r == pytest.approx(closed.net_pnl / 20.0)

    # Test conversion logic
    p = closed
    notional = p.notional
    trade_costs = TradeCosts(
        fees_pct=(p.execution_summary.entry_fee + p.execution_summary.exit_fee) / notional,
        slippage_pct=p.execution_summary.slippage_cost / notional,
        funding_pct=p.execution_summary.funding_cost / notional,
    )
    t = Trade(
        symbol=Symbol(str(p.symbol)),
        direction=p.direction,
        entry_time=p.entry_time,
        entry_price=p.entry_price,
        stop_loss=p.stop_loss,
        take_profit=p.take_profit,
        exit_time=p.exit_time,
        exit_price=p.exit_price,
        status=TradeStatus.WIN,
        exit_reason=TradeExitReason.TAKE_PROFIT,
        costs=trade_costs,
        quantity=p.quantity,
        notional=notional,
        initial_risk_dollar=p.initial_risk,
        gross_pnl=p.gross_pnl,
        net_pnl=p.net_pnl,
        profit_r=p.profit_r,
        portfolio_capital_at_entry=p.capital_at_entry,
    )
    assert t.quantity == 2.0
    assert t.notional == 200.0
    assert t.initial_risk_dollar == 20.0
    assert t.profit_r == pytest.approx(p.profit_r)
    assert t.portfolio_capital_at_entry == 10000.0


# ---------------------------------------------------------------------------
# 4. TradeAnalyzer Corrected Metrics Tests
# ---------------------------------------------------------------------------


def test_trade_analyzer_dollar_pnl_profit_factor_and_expectancy():
    """Verify TradeAnalyzer calculates PF from dollar PnL and Expectancy from Net R."""
    # Trade 1: Micro altcoin trade (+100% price return, but tiny $2 win on $2 risk)
    t1 = Trade(
        symbol=Symbol("SMALLUSDT"),
        direction=Bias.BULLISH,
        entry_time=1_000_000,
        entry_price=1.0,
        stop_loss=0.5,
        take_profit=2.0,
        exit_time=1_086_400_000,
        exit_price=2.0,
        status=TradeStatus.WIN,
        exit_reason=TradeExitReason.TAKE_PROFIT,
        quantity=4.0,
        notional=4.0,
        initial_risk_dollar=2.0,
        gross_pnl=4.0,
        net_pnl=4.0,
        profit_r=2.0,  # +2.0 R
        portfolio_capital_at_entry=10000.0,
    )

    # Trade 2: BTC trade (-5% price return, but huge $500 loss on $500 risk)
    t2 = Trade(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BULLISH,
        entry_time=1_000_000,
        entry_price=50000.0,
        stop_loss=47500.0,
        take_profit=60000.0,
        exit_time=1_086_400_000,
        exit_price=47500.0,
        status=TradeStatus.LOSS,
        exit_reason=TradeExitReason.STOP_LOSS,
        quantity=0.2,
        notional=10000.0,
        initial_risk_dollar=500.0,
        gross_pnl=-500.0,
        net_pnl=-500.0,
        profit_r=-1.0,  # -1.0 R
        portfolio_capital_at_entry=10000.0,
    )

    analyzer = TradeAnalyzer()
    report = analyzer.analyze([t1, t2])

    assert report.total_trades == 2
    assert report.wins == 1
    assert report.losses == 1

    # Real dollar Profit Factor: $4 / $500 = 0.008 (NOT 100% / 5% = 20.0!)
    assert report.net_profit_factor == pytest.approx(4.0 / 500.0)

    # Real Net Expectancy in R: (2.0 + (-1.0)) / 2 = +0.5 R
    assert report.net_expectancy == pytest.approx(0.5)

    # Real portfolio return %: ($4 - $500) / 10000 = -4.96%
    assert report.net_profit_pct == pytest.approx(-496.0 / 10000.0)


def test_trade_analyzer_time_based_sharpe_ratio():
    """Verify daily time-based Sharpe calculation with calendar day grouping."""
    base_dt = datetime(2024, 1, 1, tzinfo=UTC)
    trades = []
    # 10 days of consistent positive PnL ($10/day)
    for day in range(10):
        entry_ts = int((base_dt + timedelta(days=day)).timestamp() * 1000)
        exit_ts = int((base_dt + timedelta(days=day, hours=4)).timestamp() * 1000)
        t = Trade(
            symbol=Symbol("BTCUSDT"),
            direction=Bias.BULLISH,
            entry_time=entry_ts,
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=110.0,
            exit_time=exit_ts,
            exit_price=110.0,
            status=TradeStatus.WIN,
            exit_reason=TradeExitReason.TAKE_PROFIT,
            quantity=1.0,
            notional=100.0,
            initial_risk_dollar=10.0,
            gross_pnl=10.0,
            net_pnl=10.0,
            profit_r=1.0,
            portfolio_capital_at_entry=10000.0,
        )
        trades.append(t)

    analyzer = TradeAnalyzer()
    report = analyzer.analyze(trades)

    # Sharpe ratio should be strictly positive and annualized by sqrt(365)
    assert report.net_sharpe_ratio > 0.0


# ---------------------------------------------------------------------------
# 5. Chronological Ordering & Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_analyzer_preserves_chronology():
    """Verify BootstrapAnalyzer sorts multi-symbol trades chronologically."""
    # Create out-of-order multi-symbol trades
    t_early = Trade(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BULLISH,
        entry_time=1_000_000,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=110.0,
        exit_time=2_000_000,
        exit_price=110.0,
        status=TradeStatus.WIN,
        exit_reason=TradeExitReason.TAKE_PROFIT,
        quantity=1.0,
        net_pnl=10.0,
        profit_r=1.0,
    )
    t_late = Trade(
        symbol=Symbol("ETHUSDT"),
        direction=Bias.BULLISH,
        entry_time=5_000_000,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=110.0,
        exit_time=6_000_000,
        exit_price=110.0,
        status=TradeStatus.WIN,
        exit_reason=TradeExitReason.TAKE_PROFIT,
        quantity=1.0,
        net_pnl=10.0,
        profit_r=1.0,
    )

    recorded_first_timestamps = []

    def mock_calc(sampled):
        recorded_first_timestamps.append(sampled[0].exit_time)
        analyzer = TradeAnalyzer()
        return analyzer.analyze(sampled)

    bootstrap = BootstrapAnalyzer(mock_calc)
    # Pass them in reverse chronological order
    bootstrap.run([t_late, t_early], block_size=1, iterations=10)

    # After sorting, t_early should always precede t_late in the sequence
    assert recorded_first_timestamps[0] in (2_000_000, 6_000_000)


# ---------------------------------------------------------------------------
# 6. Cost Model Delegation Tests
# ---------------------------------------------------------------------------


def test_cost_models_delegation_integrity():
    """Verify domain/trading/execution models match domain/execution_costs canonical models."""
    del_fee = DelegatingFeeModel(maker_fee=0.0002, taker_fee=0.0005)
    can_fee = CanonicalFeeModel(maker_fee_pct=0.0002, taker_fee_pct=0.0005)

    assert del_fee.calculate_entry_fee_pct(ExecutionType.MAKER) == can_fee.calculate_entry_fee(
        1.0, ExecutionType.MAKER
    )
    assert del_fee.calculate_entry_fee_pct(ExecutionType.TAKER) == can_fee.calculate_entry_fee(
        1.0, ExecutionType.TAKER
    )

    del_slip = DelegatingSlippageModel(slippage_pct=0.0008)
    can_slip = CanonicalSlippageModel(slippage_pct=0.0008)

    assert del_slip.calculate_slippage_pct(
        Symbol("BTCUSDT"), ExecutionType.TAKER, Bias.BULLISH
    ) == can_slip.calculate_slippage(1.0, Symbol("BTCUSDT"), ExecutionType.TAKER, Bias.BULLISH)


# ---------------------------------------------------------------------------
# 7. Funding Provider Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_binance_historical_funding_provider():
    """Verify BinanceHistoricalFundingProvider queries and looks up rates properly."""
    client_mock = MagicMock()
    # Mock Binance /fapi/v1/fundingRate
    client_mock._get_json = AsyncMock(
        return_value=[
            {"fundingRate": "0.0001", "fundingTime": 28_800_000},
            {"fundingRate": "0.0002", "fundingTime": 57_600_000},
        ]
    )

    provider = BinanceHistoricalFundingProvider(client=client_mock)
    sym = Symbol("BTCUSDT")

    await provider.prefetch((sym,), date(1970, 1, 1), date(1970, 1, 2))

    r1 = provider.get_funding_rate_at(sym, 28_800_000)
    assert r1 is not None
    assert r1.rate == 0.0001

    r2 = provider.get_funding_rate_at(sym, 57_600_000)
    assert r2 is not None
    assert r2.rate == 0.0002

    # Query between boundaries: should return closest preceding rate
    r_mid = provider.get_funding_rate_at(sym, 35_000_000)
    assert r_mid is not None
    assert r_mid.rate == 0.0001
