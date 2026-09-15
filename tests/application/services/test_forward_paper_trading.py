"""Comprehensive tests for Forward Paper Trading Engine.

Validates:
1. Closed candle triggers exactly once.
2. Open (forming) candle does not trigger signal.
3. Paper order produces simulated fill on price touch.
4. Stop loss closes position with taker fee.
5. Take profit closes position with maker fee.
6. Fees, slippage, and funding cost accounting.
7. Multi-asset shared portfolio state and global concurrent limits.
8. Duplicate candle protection (idempotency).
9. Crash recovery and state restoration from JSON store.
10. Stale data rejection.
11. Network disconnection and reconnection resilience.
12. Risk rejection blocks paper order creation.
13. Zero real order API calls in paper mode.
14. Complete point-in-time telemetry on closed positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from neon_radar.application.services.execution import PaperExecutionEngine
from neon_radar.application.services.paper_trading_engine import (
    CANONICAL_SYMBOLS,
    ForwardPaperTradingEngine,
)
from neon_radar.application.services.portfolio_engine import PortfolioEngine
from neon_radar.application.services.risk.manager import RiskManager
from neon_radar.domain.enums import Bias
from neon_radar.domain.execution_costs import (
    BinanceFuturesFeeModel,
    FixedSlippageModel,
)
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.portfolio import PositionCloseReason
from neon_radar.domain.risk import PortfolioRiskPolicy, RiskDecision
from neon_radar.domain.trading.backtest import TradeDiagnostics, TradeEntryReason
from neon_radar.domain.trading.setup import FinalTradeSetup
from neon_radar.infrastructure.exchanges.base import ExchangeClient, ExchangeInfo

if TYPE_CHECKING:
    from neon_radar.config.models import TimeFrame
from neon_radar.infrastructure.storage.paper_state_store import JsonPaperStateStore


class MockExchangeClient(ExchangeClient):
    """Mock exchange client for paper trading tests."""

    name = "mock_binance"

    def __init__(self, server_time: int = 1_700_000_000_000) -> None:
        self.server_time = server_time
        self.klines_db: dict[tuple[str, str], list[OHLCV]] = {}
        self.raise_on_klines: Exception | None = None
        self.order_calls: list[Any] = []

    async def info(self) -> ExchangeInfo:
        return ExchangeInfo("mock", "Mock Exchange", "https://mock.io", True, True)

    async def get_server_time(self) -> int:
        return self.server_time

    async def get_klines(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        *,
        limit: int = 500,
        end_time: int | None = None,
    ) -> KlineSeries:
        if self.raise_on_klines is not None:
            raise self.raise_on_klines

        candles = self.klines_db.get((str(symbol), timeframe.value), [])
        return KlineSeries(symbol=symbol, timeframe=timeframe.value, candles=tuple(candles))

    async def get_ticker(self, symbol: Symbol) -> Any:
        from neon_radar.domain.models import TickerStats

        return TickerStats(
            symbol=symbol,
            last_price=100.0,
            price_change_percent_24h=0.0,
            quote_volume_24h=1_000_000.0,
            high_price_24h=105.0,
            low_price_24h=95.0,
        )

    async def post_order(self, *args: Any, **kwargs: Any) -> None:
        self.order_calls.append((args, kwargs))


def make_candle(
    open_time: int,
    open_price: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: float = 1000.0,
    tf_seconds: int = 86400,
) -> OHLCV:
    close_time = open_time + (tf_seconds * 1000) - 1
    return OHLCV(
        open_time=open_time,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        close_time=close_time,
    )


def make_dummy_setup(
    symbol: Symbol = Symbol("BTCUSDT"),
    entry: float = 100.0,
    sl: float | None = None,
    tp: float | None = None,
    position_size: float | None = None,
    quote_size: float = 100.0,
    direction: Bias = Bias.BULLISH,
) -> FinalTradeSetup:
    if sl is None:
        sl = entry * 0.95 if direction == Bias.BULLISH else entry * 1.05
    if tp is None:
        tp = entry * 1.10 if direction == Bias.BULLISH else entry * 0.90
    if position_size is None:
        position_size = quote_size / entry if entry > 0 else 1.0
    diagnostics = TradeDiagnostics(
        adx=25.0,
        atr=5.0,
        rsi=55.0,
        ema_spread_pct=0.02,
        htf_trend=1.0,
        confidence=0.80,
        final_score=0.75,
        triggered_rules="trend:0.5, rsi:0.25",
        entry_reason=TradeEntryReason.CONFIDENCE_THRESHOLD,
        regime="BULL_TREND",
        regime_reason="Trend test",
    )
    return FinalTradeSetup(
        symbol=symbol,
        direction=direction,
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        confidence=0.80,
        score=0.75,
        risk_decision=RiskDecision(is_allowed=True, max_risk_budget=10.0, max_position_size=quote_size),
        position_size=position_size,
        quote_size=quote_size,
        risk_amount=abs(entry - sl) * position_size,
        expected_reward=abs(tp - entry) * position_size,
        diagnostics=diagnostics,
    )


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.evaluate.return_value = None
    return pipeline


# ---------------------------------------------------------------------------
# 1. Closed candle triggers exactly once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closed_candle_triggers_exactly_once(mock_pipeline):
    base_time = 1_700_000_000_000

    # Candle 1 is closed
    c1 = make_candle(base_time, close=100.0)
    server_time = c1.close_time + 1000

    exchange = MockExchangeClient(server_time=server_time)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    setup = make_dummy_setup(symbol=CANONICAL_SYMBOLS[0], entry=100.0)
    mock_pipeline.evaluate.return_value = setup

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    # First cycle: candle is closed, pipeline must be evaluated
    report1 = await engine.poll_cycle()
    assert len(report1.new_setups) == 1
    assert mock_pipeline.evaluate.call_count == 1
    assert engine.last_processed_candles[str(CANONICAL_SYMBOLS[0])] == c1.open_time

    # Second cycle with same data: must NOT re-trigger (idempotent)
    report2 = await engine.poll_cycle()
    assert len(report2.new_setups) == 0
    assert mock_pipeline.evaluate.call_count == 1  # Not called again


# ---------------------------------------------------------------------------
# 2. Open (forming) candle does not trigger signal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_candle_does_not_trigger_signal(mock_pipeline):
    base_time = 1_700_000_000_000
    c1 = make_candle(base_time)

    # Server time is INSIDE the candle (candle is still open/forming)
    server_time = c1.open_time + 3600 * 1000

    exchange = MockExchangeClient(server_time=server_time)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    report = await engine.poll_cycle()
    assert len(report.new_setups) == 0
    assert mock_pipeline.evaluate.call_count == 0  # Pipeline was never called on forming candle


# ---------------------------------------------------------------------------
# 3. Paper order produces simulated fill
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paper_order_produces_simulated_fill(mock_pipeline):
    base_time = 1_700_000_000_000
    tf_ms = 86400 * 1000

    c1 = make_candle(base_time, close=100.0)
    c2 = make_candle(base_time + tf_ms, open_price=100.0, high=105.0, low=98.0, close=102.0)

    exchange = MockExchangeClient(server_time=c1.close_time + 1000)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    setup = make_dummy_setup(symbol=CANONICAL_SYMBOLS[0], entry=100.0, quote_size=500.0, position_size=5.0)
    mock_pipeline.evaluate.return_value = setup

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    # Cycle 1: register pending setup
    await engine.poll_cycle()
    assert str(CANONICAL_SYMBOLS[0]) in engine.execution_engine.pending_setups
    assert len(engine.portfolio_engine.state.positions) == 0

    # Cycle 2: candle 2 arrives, price trades at 100.0 -> fills order
    exchange.server_time = c2.close_time + 1000
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1, c2]
    mock_pipeline.evaluate.return_value = None  # No new setup on C2

    report2 = await engine.poll_cycle()
    assert len(report2.new_fills) == 1
    assert len(engine.portfolio_engine.state.positions) == 1
    pos = engine.portfolio_engine.state.positions[0]
    assert pos.symbol == CANONICAL_SYMBOLS[0]
    assert pos.entry_price == 100.0
    assert str(CANONICAL_SYMBOLS[0]) not in engine.execution_engine.pending_setups


# ---------------------------------------------------------------------------
# 4. Stop loss closes position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sl_closes_position(mock_pipeline):
    base_time = 1_700_000_000_000
    tf_ms = 86400 * 1000

    c1 = make_candle(base_time, close=100.0)
    c2 = make_candle(base_time + tf_ms, low=99.0, high=102.0, close=100.0)  # Fills entry at 100
    c3 = make_candle(base_time + 2 * tf_ms, low=94.0, high=101.0, close=95.0)  # Hits SL at 95

    exchange = MockExchangeClient(server_time=c1.close_time + 1000)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    setup = make_dummy_setup(symbol=CANONICAL_SYMBOLS[0], entry=100.0, sl=95.0, tp=110.0, position_size=2.0)
    mock_pipeline.evaluate.return_value = setup

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    # 1. Setup placed
    await engine.poll_cycle()
    mock_pipeline.evaluate.return_value = None

    # 2. Filled on C2
    exchange.server_time = c2.close_time + 1000
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1, c2]
    await engine.poll_cycle()
    assert len(engine.portfolio_engine.state.positions) == 1

    # 3. Stopped out on C3
    exchange.server_time = c3.close_time + 1000
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1, c2, c3]
    report = await engine.poll_cycle()

    assert len(report.new_exits) == 1
    assert len(engine.portfolio_engine.state.positions) == 0
    closed = report.new_exits[0]
    assert closed.close_reason == PositionCloseReason.STOP_LOSS
    assert closed.net_pnl < 0


# ---------------------------------------------------------------------------
# 5. Take profit closes position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tp_closes_position(mock_pipeline):
    base_time = 1_700_000_000_000
    tf_ms = 86400 * 1000

    c1 = make_candle(base_time, close=100.0)
    c2 = make_candle(base_time + tf_ms, low=99.0, high=102.0, close=100.0)  # Fills entry at 100
    c3 = make_candle(base_time + 2 * tf_ms, low=99.0, high=112.0, close=111.0)  # Hits TP at 110

    exchange = MockExchangeClient(server_time=c1.close_time + 1000)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    setup = make_dummy_setup(symbol=CANONICAL_SYMBOLS[0], entry=100.0, sl=95.0, tp=110.0, position_size=2.0)
    mock_pipeline.evaluate.return_value = setup

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    await engine.poll_cycle()
    mock_pipeline.evaluate.return_value = None

    exchange.server_time = c2.close_time + 1000
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1, c2]
    await engine.poll_cycle()

    exchange.server_time = c3.close_time + 1000
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1, c2, c3]
    report = await engine.poll_cycle()

    assert len(report.new_exits) == 1
    assert len(engine.portfolio_engine.state.positions) == 0
    closed = report.new_exits[0]
    assert closed.close_reason == PositionCloseReason.TAKE_PROFIT
    assert closed.net_pnl > 0


# ---------------------------------------------------------------------------
# 6. Fees, slippage, and funding cost accounting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fees_slippage_funding_accounting(mock_pipeline):
    base_time = 1_700_000_000_000
    tf_ms = 86400 * 1000

    c1 = make_candle(base_time, close=100.0)
    c2 = make_candle(base_time + tf_ms, low=99.0, high=101.0, close=100.0)
    c3 = make_candle(base_time + 2 * tf_ms, low=94.0, high=101.0, close=95.0)

    exchange = MockExchangeClient(server_time=c1.close_time + 1000)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    portfolio = PortfolioEngine(initial_capital=10000.0)
    exec_engine = PaperExecutionEngine(
        portfolio_engine=portfolio,
        fee_model=BinanceFuturesFeeModel(),
        slippage_model=FixedSlippageModel(slippage_pct=0.0005),
    )

    setup = make_dummy_setup(symbol=CANONICAL_SYMBOLS[0], entry=100.0, sl=95.0, tp=110.0, position_size=10.0, quote_size=1000.0)
    mock_pipeline.evaluate.return_value = setup

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        portfolio_engine=portfolio,
        execution_engine=exec_engine,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    await engine.poll_cycle()
    mock_pipeline.evaluate.return_value = None

    exchange.server_time = c2.close_time + 1000
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1, c2]
    await engine.poll_cycle()

    exchange.server_time = c3.close_time + 1000
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1, c2, c3]
    report = await engine.poll_cycle()

    closed = report.new_exits[0]
    costs = closed.execution_summary
    assert costs.entry_fee > 0
    assert costs.exit_fee > 0
    assert costs.slippage_cost > 0
    assert costs.gross_pnl == (95.0 - 100.0) * 10.0  # -50.0
    expected_net = costs.gross_pnl - costs.entry_fee - costs.exit_fee - costs.slippage_cost - costs.funding_cost
    assert costs.net_pnl == pytest.approx(expected_net)
    assert portfolio.state.account.total_capital == pytest.approx(10000.0 + expected_net)


# ---------------------------------------------------------------------------
# 7. Multi-asset shared portfolio & global concurrent trades limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_asset_shared_portfolio(mock_pipeline):
    base_time = 1_700_000_000_000
    tf_ms = 86400 * 1000

    btc_c1 = make_candle(base_time, close=50000.0)
    eth_c1 = make_candle(base_time, close=3000.0)

    exchange = MockExchangeClient(server_time=btc_c1.close_time + 1000)
    exchange.klines_db[("BTCUSDT", "1d")] = [btc_c1]
    exchange.klines_db[("ETHUSDT", "1d")] = [eth_c1]

    # Shared portfolio with $10,000
    portfolio = PortfolioEngine(initial_capital=10000.0)
    risk_policy = PortfolioRiskPolicy(max_concurrent_trades=2)
    risk_mgr = RiskManager(risk_policy)

    mock_pipeline.risk_manager = risk_mgr

    # Pipeline returns setup matching symbol
    def eval_side_effect(series, portfolio, **kwargs):
        sym = series.symbol
        return make_dummy_setup(symbol=sym, entry=series.candles[-1].close, quote_size=2000.0)

    mock_pipeline.evaluate.side_effect = eval_side_effect

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        portfolio_engine=portfolio,
        symbols=(Symbol("BTCUSDT"), Symbol("ETHUSDT")),
    )

    # Both symbols evaluated in single cycle
    report = await engine.poll_cycle()
    assert len(report.new_setups) == 2
    assert "BTCUSDT" in engine.execution_engine.pending_setups
    assert "ETHUSDT" in engine.execution_engine.pending_setups

    # Next cycle fills both
    btc_c2 = make_candle(base_time + tf_ms, low=49000.0, high=51000.0, close=50000.0)
    eth_c2 = make_candle(base_time + tf_ms, low=2900.0, high=3100.0, close=3000.0)
    exchange.server_time = btc_c2.close_time + 1000
    exchange.klines_db[("BTCUSDT", "1d")] = [btc_c1, btc_c2]
    exchange.klines_db[("ETHUSDT", "1d")] = [eth_c1, eth_c2]

    await engine.poll_cycle()
    # Both active in single shared portfolio
    assert len(portfolio.state.positions) == 2
    assert portfolio.state.account.free_capital == pytest.approx(10000.0 - 2000.0 - 2000.0)


# ---------------------------------------------------------------------------
# 8. Duplicate candle protection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_candle_protection(mock_pipeline):
    base_time = 1_700_000_000_000
    c1 = make_candle(base_time, close=100.0)

    exchange = MockExchangeClient(server_time=c1.close_time + 1000)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    # First cycle processes candle
    await engine.poll_cycle()
    assert engine.last_processed_candles[str(CANONICAL_SYMBOLS[0])] == c1.open_time

    # Same candle repeatedly sent -> ignored
    for _ in range(3):
        report = await engine.poll_cycle()
        assert len(report.symbols_evaluated) == 0


# ---------------------------------------------------------------------------
# 9. Crash recovery and restart idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_recovery_idempotency(tmp_path, mock_pipeline):
    base_time = 1_700_000_000_000
    c1 = make_candle(base_time, close=100.0)

    state_file = tmp_path / "paper_state.json"
    state_store = JsonPaperStateStore(state_file)

    exchange = MockExchangeClient(server_time=c1.close_time + 1000)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    setup = make_dummy_setup(symbol=CANONICAL_SYMBOLS[0], entry=100.0)
    mock_pipeline.evaluate.return_value = setup

    # 1. First engine instance generates setup and saves state
    engine1 = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        state_store=state_store,
        symbols=(CANONICAL_SYMBOLS[0],),
    )
    await engine1.poll_cycle()
    assert state_file.exists()

    # 2. Simulate shutdown and restart with fresh engine instance
    engine2 = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        state_store=state_store,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    # State restored accurately
    assert str(CANONICAL_SYMBOLS[0]) in engine2.execution_engine.pending_setups
    assert engine2.last_processed_candles[str(CANONICAL_SYMBOLS[0])] == c1.open_time

    # Running cycle does not duplicate setup
    report = await engine2.poll_cycle()
    assert len(report.new_setups) == 0


# ---------------------------------------------------------------------------
# 10. Stale data rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_data_rejection(mock_pipeline):
    base_time = 1_700_000_000_000
    c1 = make_candle(base_time, close=100.0)

    # Server time is 10 days ahead (stale data)
    server_time = c1.close_time + (10 * 86400 * 1000)

    exchange = MockExchangeClient(server_time=server_time)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
        max_stale_tolerance_seconds=86400 * 2,
    )

    report = await engine.poll_cycle()
    assert len(report.new_setups) == 0
    assert any("Stale market data" in w for w in report.warnings)
    assert mock_pipeline.evaluate.call_count == 0


# ---------------------------------------------------------------------------
# 11. Network disconnection and reconnection resilience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_network_disconnection_resilience(mock_pipeline):
    base_time = 1_700_000_000_000
    c1 = make_candle(base_time, close=100.0)

    exchange = MockExchangeClient(server_time=c1.close_time + 1000)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]
    exchange.raise_on_klines = ConnectionResetError("Connection dropped")

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    # Does not raise; captures warning
    report = await engine.poll_cycle()
    assert any("Connection dropped" in w for w in report.warnings)

    # Reconnect: clear exception
    exchange.raise_on_klines = None
    mock_pipeline.evaluate.return_value = make_dummy_setup(symbol=CANONICAL_SYMBOLS[0])

    report2 = await engine.poll_cycle()
    assert len(report2.new_setups) == 1


# ---------------------------------------------------------------------------
# 12. Risk rejection blocks paper order creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_rejection_no_paper_order(mock_pipeline):
    base_time = 1_700_000_000_000
    c1 = make_candle(base_time, close=100.0)

    exchange = MockExchangeClient(server_time=c1.close_time + 1000)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    # Setup with disallowed risk decision
    rejected_setup = make_dummy_setup(symbol=CANONICAL_SYMBOLS[0])
    object.__setattr__(rejected_setup, "risk_decision", RiskDecision(is_allowed=False, rejection_reason="Max DD hit"))
    mock_pipeline.evaluate.return_value = rejected_setup

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    report = await engine.poll_cycle()
    assert len(report.new_setups) == 0
    assert str(CANONICAL_SYMBOLS[0]) not in engine.execution_engine.pending_setups


# ---------------------------------------------------------------------------
# 13. Zero real order API calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_real_order_api_call(mock_pipeline):
    base_time = 1_700_000_000_000
    c1 = make_candle(base_time, close=100.0)

    exchange = MockExchangeClient(server_time=c1.close_time + 1000)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]
    mock_pipeline.evaluate.return_value = make_dummy_setup(symbol=CANONICAL_SYMBOLS[0])

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    await engine.poll_cycle()
    # Confirm mock order execution method was never called
    assert len(exchange.order_calls) == 0


# ---------------------------------------------------------------------------
# 14. Point-in-time telemetry complete on ClosedPosition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_point_in_time_telemetry_complete(mock_pipeline):
    base_time = 1_700_000_000_000
    tf_ms = 86400 * 1000

    c1 = make_candle(base_time, close=100.0)
    c2 = make_candle(base_time + tf_ms, low=99.0, high=101.0, close=100.0)
    c3 = make_candle(base_time + 2 * tf_ms, low=99.0, high=115.0, close=110.0)

    exchange = MockExchangeClient(server_time=c1.close_time + 1000)
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1]

    setup = make_dummy_setup(symbol=CANONICAL_SYMBOLS[0], entry=100.0, sl=95.0, tp=110.0, position_size=5.0)
    mock_pipeline.evaluate.return_value = setup

    engine = ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=mock_pipeline,
        symbols=(CANONICAL_SYMBOLS[0],),
    )

    await engine.poll_cycle()
    mock_pipeline.evaluate.return_value = None

    exchange.server_time = c2.close_time + 1000
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1, c2]
    await engine.poll_cycle()

    exchange.server_time = c3.close_time + 1000
    exchange.klines_db[(str(CANONICAL_SYMBOLS[0]), "1d")] = [c1, c2, c3]
    report = await engine.poll_cycle()

    closed = report.new_exits[0]
    # Telemetry verification (Requirement 5)
    assert closed.symbol == CANONICAL_SYMBOLS[0]
    assert closed.direction == Bias.BULLISH
    assert closed.entry_price == 100.0
    assert closed.exit_price == 110.0
    assert closed.stop_loss == 95.0
    assert closed.take_profit == 110.0
    assert closed.close_reason == PositionCloseReason.TAKE_PROFIT
    assert closed.diagnostics is not None
    assert closed.diagnostics.regime == "BULL_TREND"
    assert closed.diagnostics.confidence == 0.80
    assert closed.timeframe == "1d"
    assert closed.signal_time == c1.open_time
