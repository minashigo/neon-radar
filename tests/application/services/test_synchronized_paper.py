"""Multi-asset chronology and iteration-order regression tests."""

from dataclasses import replace
from itertools import permutations
from unittest.mock import MagicMock

import pytest
from test_forward_paper_trading import MockExchangeClient, make_candle, make_dummy_setup

from neon_radar.application.services.paper_trading_engine import ForwardPaperTradingEngine
from neon_radar.application.services.portfolio_engine import PortfolioEngine
from neon_radar.application.services.risk.manager import RiskManager
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import Symbol
from neon_radar.domain.portfolio import OpenPosition
from neon_radar.domain.risk import PortfolioRiskPolicy

BASE = 1_700_000_000_000
BTC, ETH, SOL = map(Symbol, ("BTCUSDT", "ETHUSDT", "SOLUSDT"))


def engine_for(order, candles, portfolio=None, policy=None):
    exchange = MockExchangeClient(server_time=max(c.close_time for c in candles.values()) + 1000)
    for symbol, candle in candles.items():
        exchange.klines_db[str(symbol), "1d"] = [candle]
    pipeline = MagicMock()
    pipeline.risk_manager = RiskManager(policy or PortfolioRiskPolicy(base_risk_per_trade_pct=0.01))
    pipeline.evaluate.return_value = None
    engine = ForwardPaperTradingEngine(
        exchange=exchange, pipeline=pipeline, symbols=tuple(order), portfolio_engine=portfolio
    )
    return engine


def portfolio_with_btc():
    portfolio = PortfolioEngine(10000)
    portfolio.open_position(
        OpenPosition(
            symbol=BTC,
            direction=Bias.BULLISH,
            entry_price=100,
            quantity=10,
            position_size=1000,
            stop_loss=50,
            take_profit=300,
            opened_at=BASE - 1,
        )
    )
    return portfolio


@pytest.mark.asyncio
@pytest.mark.parametrize("order", list(permutations((BTC, ETH, SOL))))
async def test_close_loss_cannot_cancel_another_assets_open_fill(order):
    candles = {
        BTC: make_candle(BASE, open_price=100, high=101, low=89, close=90),
        ETH: make_candle(BASE, open_price=100, high=101, low=99, close=100),
        SOL: make_candle(BASE, open_price=100, high=101, low=99, close=100),
    }
    engine = engine_for(order, candles, portfolio_with_btc())
    assert engine.execution_engine.execute_setup(
        make_dummy_setup(symbol=ETH, quote_size=1000, sl=90, tp=120), BASE - 1
    )
    report = await engine.poll_cycle()
    assert not report.warnings
    assert {p.symbol for p in engine.portfolio_engine.state.positions} == {BTC, ETH}
    assert report.equity == 9900
    assert tuple(p.symbol for p in report.new_fills) == (ETH,)
    assert tuple(
        call.kwargs["series"].symbol for call in engine.pipeline.evaluate.call_args_list
    ) == (BTC, ETH, SOL)
    assert all(
        call.kwargs["portfolio"].account.total_capital == 9900
        for call in engine.pipeline.evaluate.call_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [(BTC, ETH), (ETH, BTC)])
async def test_intrabar_profit_cannot_fund_a_previous_open_entry(order):
    portfolio = portfolio_with_btc()
    portfolio.restore_state(
        portfolio.state.account, positions=(replace(portfolio.state.positions[0], take_profit=110),)
    )
    engine = engine_for(
        order,
        {
            BTC: make_candle(BASE, open_price=100, high=111, low=99, close=110),
            ETH: make_candle(BASE, open_price=100, high=101, low=99, close=100),
        },
        portfolio,
    )
    # A restored order exceeds the current 100 risk budget by 1. The later BTC
    # profit raises the budget to 101, but is unavailable when ETH opens.
    setup = make_dummy_setup(symbol=ETH, quote_size=1010, sl=90, tp=120)
    engine.execution_engine.pending_setups["ETHUSDT"] = setup
    report = await engine.poll_cycle()
    assert not report.warnings
    assert not engine.portfolio_engine.state.positions
    assert not report.new_fills
    assert report.equity > 10000


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [(BTC, ETH), (ETH, BTC)])
async def test_known_gap_exit_precedes_all_entries_and_report_retains_fill(order):
    portfolio = portfolio_with_btc()
    portfolio.restore_state(
        portfolio.state.account, positions=(replace(portfolio.state.positions[0], take_profit=110),)
    )
    engine = engine_for(
        order,
        {
            BTC: make_candle(BASE, open_price=120, high=125, low=115, close=120),
            ETH: make_candle(BASE, open_price=100, high=101, low=99, close=100),
        },
        portfolio,
        PortfolioRiskPolicy(max_concurrent_trades=1),
    )
    engine.execution_engine.pending_setups["ETHUSDT"] = make_dummy_setup(symbol=ETH)
    report = await engine.poll_cycle()
    assert not report.warnings
    assert tuple(p.symbol for p in report.new_fills) == (ETH,)
    assert tuple(p.symbol for p in report.new_exits) == (BTC,)
    assert tuple(p.symbol for p in engine.portfolio_engine.state.positions) == (ETH,)


@pytest.mark.asyncio
@pytest.mark.parametrize("order", list(permutations((BTC, ETH, SOL))))
async def test_signal_budget_uses_fixed_priority_and_same_close_snapshot(order):
    candles = {s: make_candle(BASE, open_price=100, high=101, low=99, close=100) for s in order}
    engine = engine_for(order, candles, policy=PortfolioRiskPolicy(max_concurrent_trades=2))
    engine.pipeline.evaluate.side_effect = lambda series, **kwargs: make_dummy_setup(
        symbol=series.symbol
    )
    report = await engine.poll_cycle()
    assert tuple(p.symbol for p in report.new_setups) == (BTC, ETH)
    assert tuple(engine.execution_engine.pending_setups) == ("BTCUSDT", "ETHUSDT")


@pytest.mark.asyncio
async def test_misaligned_data_defer_all_mutations_until_complete_batch():
    first = make_candle(BASE, open_price=100, high=101, low=99, close=100)
    later = make_candle(BASE + 86400000, open_price=100, high=101, low=99, close=100)
    engine = engine_for((BTC, ETH), {BTC: later, ETH: first})
    assert engine.execution_engine.execute_setup(make_dummy_setup(symbol=ETH), BASE - 1)
    state = engine.portfolio_engine.state
    report = await engine.poll_cycle()
    assert report.warnings
    assert not report.symbols_evaluated
    assert not engine.last_processed_candles
    assert engine.portfolio_engine.state == state
    assert engine.execution_engine.pending_setups
    engine.exchange.klines_db["ETHUSDT", "1d"] = [first, later]
    report = await engine.poll_cycle()
    assert not report.warnings
    assert len(report.new_fills) == 1
    assert set(engine.last_processed_candles.values()) == {later.open_time}


@pytest.mark.asyncio
async def test_report_includes_fill_closed_in_the_same_batch():
    engine = engine_for(
        (ETH,), {ETH: make_candle(BASE, open_price=100, high=115, low=99, close=112)}
    )
    engine.execution_engine.execute_setup(make_dummy_setup(symbol=ETH, tp=110), BASE - 1)
    report = await engine.poll_cycle()
    assert len(report.new_fills) == len(report.new_exits) == 1
    assert not engine.portfolio_engine.state.positions


def test_all_720_batch_permutations_preserve_fills_exits_equity_and_observations():
    from neon_radar.application.services.execution import PaperExecutionEngine

    symbols = tuple(map(Symbol, ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")))
    candles = {s: make_candle(BASE, open_price=100, high=101, low=99, close=100) for s in symbols}
    candles[BTC] = make_candle(BASE, open_price=120, high=125, low=80, close=100)
    candles[ETH] = make_candle(BASE, open_price=100, high=125, low=99, close=122)
    candles[SOL] = make_candle(BASE, open_price=105, high=110, low=95, close=101)
    reference = None
    count = 0
    for order in permutations(symbols):
        portfolio = portfolio_with_btc()
        portfolio.restore_state(
            portfolio.state.account,
            positions=(replace(portfolio.state.positions[0], take_profit=110),),
        )
        execution = PaperExecutionEngine(
            portfolio, risk_manager=RiskManager(PortfolioRiskPolicy(base_risk_per_trade_pct=0.01))
        )
        for symbol in (ETH, SOL):
            assert execution.execute_setup(make_dummy_setup(symbol=symbol, sl=90, tp=120), BASE - 1)
        observations = []
        fills = execution.process_market_batch(
            {s: candles[s] for s in order},
            on_equity_update=lambda equity, time, observed=observations: observed.append(
                (equity, time)
            ),
        )
        signature = (
            portfolio.state,
            portfolio.history,
            fills,
            execution.pending_setups,
            observations,
        )
        assert tuple(p.symbol for p in fills) == (ETH, SOL)
        assert tuple(p.symbol for p in portfolio.history) == (BTC, ETH)
        assert tuple(p.symbol for p in portfolio.state.positions) == (SOL,)
        if reference is None:
            reference = signature
        assert signature == reference
        count += 1
    assert count == 720


def test_mixed_batch_timestamps_are_rejected_before_mutation():
    from neon_radar.application.services.execution import PaperExecutionEngine

    portfolio = portfolio_with_btc()
    execution = PaperExecutionEngine(portfolio)
    state = portfolio.state
    with pytest.raises(ValueError, match="same open_time"):
        execution.process_market_batch(
            {
                BTC: make_candle(BASE),
                ETH: make_candle(BASE + 86400000),
            }
        )
    assert portfolio.state == state
    assert not portfolio.history
    with pytest.raises(ValueError, match="same close_time"):
        execution.process_market_batch(
            {
                BTC: make_candle(BASE),
                ETH: make_candle(BASE, tf_seconds=3600),
            }
        )
    assert portfolio.state == state
    with pytest.raises(ValueError, match="same close_time"):
        execution.process_market_batch(
            {
                BTC: make_candle(BASE),
                ETH: replace(make_candle(BASE), close_time=None),
            }
        )
    assert portfolio.state == state


@pytest.mark.asyncio
async def test_partial_network_failure_does_not_execute_the_other_symbol():
    candles = {s: make_candle(BASE, close=100) for s in (BTC, ETH)}
    engine = engine_for((BTC, ETH), candles)
    engine.execution_engine.execute_setup(make_dummy_setup(symbol=BTC), BASE - 1)
    original = engine.exchange.get_klines

    async def failing(symbol, timeframe, **kwargs):
        if symbol == ETH:
            raise ConnectionError("ETH unavailable")
        return await original(symbol, timeframe, **kwargs)

    engine.exchange.get_klines = failing
    report = await engine.poll_cycle()
    assert report.warnings
    assert not report.new_fills
    assert not report.symbols_evaluated
    assert not engine.last_processed_candles
    assert engine.execution_engine.pending_setups


@pytest.mark.asyncio
async def test_restart_preserves_shared_watermark_and_does_not_repeat_fills(tmp_path):
    from neon_radar.infrastructure.storage.paper_state_store import JsonPaperStateStore

    candles = {s: make_candle(BASE, close=100) for s in (BTC, ETH)}
    engine = engine_for((ETH, BTC), candles)
    engine.state_store = JsonPaperStateStore(tmp_path / "paper.json")
    engine.execution_engine.execute_setup(make_dummy_setup(symbol=ETH), BASE - 1)
    first = await engine.poll_cycle()
    assert len(first.new_fills) == 1
    restarted = engine_for((BTC, ETH), candles)
    restarted.state_store = engine.state_store
    assert restarted.load_state()
    state = restarted.portfolio_engine.state
    repeated = await restarted.poll_cycle()
    assert not repeated.new_fills
    assert not repeated.new_exits
    assert not repeated.symbols_evaluated
    assert restarted.portfolio_engine.state == state
    assert restarted.last_processed_candles == engine.last_processed_candles


@pytest.mark.asyncio
async def test_legacy_skewed_watermark_never_rewinds_shared_portfolio():
    candles = {s: make_candle(BASE, close=100) for s in (BTC, ETH)}
    engine = engine_for((BTC, ETH), candles)
    engine.last_processed_candles = {"BTCUSDT": BASE, "ETHUSDT": BASE - 86400000}
    engine.execution_engine.execute_setup(make_dummy_setup(symbol=ETH), BASE - 1)
    report = await engine.poll_cycle()
    assert not report.new_fills
    assert not report.symbols_evaluated
    later = make_candle(BASE + 86400000, open_price=100, high=101, low=99, close=100)
    engine.exchange.server_time = later.close_time + 1000
    for symbol in (BTC, ETH):
        engine.exchange.klines_db[str(symbol), "1d"] = [candles[symbol], later]
    report = await engine.poll_cycle()
    assert not report.warnings
    assert len(report.new_fills) == 1
    assert set(engine.last_processed_candles.values()) == {later.open_time}


@pytest.mark.asyncio
async def test_optional_inputs_finish_before_any_signal_registration():
    candles = {s: make_candle(BASE, close=100) for s in (BTC, ETH)}
    engine = engine_for((ETH, BTC), candles)
    checked = []

    async def funding(symbol):
        # No early symbol may register its new order before later I/O completes.
        assert not engine.execution_engine.pending_setups
        checked.append(symbol)
        return None

    engine.exchange.get_funding_rate = funding
    engine.pipeline.evaluate.side_effect = lambda series, **kwargs: make_dummy_setup(
        symbol=series.symbol
    )
    report = await engine.poll_cycle()
    assert tuple(checked) == (BTC, ETH)
    assert len(report.new_setups) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("order", [(BTC, ETH), (ETH, BTC)])
async def test_drawdown_guard_sees_common_open_before_close_recovers(order):
    from neon_radar.domain.risk import DrawdownTier

    engine = engine_for(
        order,
        {
            BTC: make_candle(BASE, open_price=90, high=101, low=89, close=100),
            ETH: make_candle(BASE, open_price=100, high=101, low=99, close=100),
        },
        portfolio_with_btc(),
        PortfolioRiskPolicy(
            base_risk_per_trade_pct=0.01,
            enable_circuit_breaker=True,
            drawdown_tiers=(DrawdownTier(0.5, 0, is_trading_halt=True),),
        ),
    )
    assert engine.execution_engine.execute_setup(make_dummy_setup(symbol=ETH), BASE - 1)
    report = await engine.poll_cycle()
    assert not report.warnings
    assert not report.new_fills
    assert tuple(p.symbol for p in engine.portfolio_engine.state.positions) == (BTC,)
    assert report.equity == 10000
    assert report.drawdown_pct >= 1
