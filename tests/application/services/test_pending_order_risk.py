"""Regression tests for pending orders sharing a forward portfolio risk budget."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_forward_paper_trading import MockExchangeClient, make_candle, make_dummy_setup

from neon_radar.application.services.paper_trading_engine import ForwardPaperTradingEngine
from neon_radar.application.services.risk.manager import RiskManager
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import Symbol
from neon_radar.domain.risk import PortfolioRiskPolicy
from neon_radar.infrastructure.storage.paper_state_store import JsonPaperStateStore


def make_engine(policy, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"), store=None):
    candle = make_candle(1_700_000_000_000, close=100.0)
    exchange = MockExchangeClient(server_time=candle.close_time + 1000)
    pipeline = MagicMock()
    pipeline.risk_manager = RiskManager(policy)

    def evaluate(series, portfolio, drawdown=None, **kwargs):
        analysis = SimpleNamespace(
            market_state=SimpleNamespace(primary_series=series),
            score=SimpleNamespace(bias=Bias.BULLISH),
        )
        decision = pipeline.risk_manager.evaluate(analysis, portfolio, drawdown)
        if not decision.is_allowed:
            return None
        return replace(
            make_dummy_setup(symbol=series.symbol, quote_size=1000.0),
            risk_decision=decision,
        )

    pipeline.evaluate.side_effect = evaluate
    for symbol in symbols:
        exchange.klines_db[(symbol, "1d")] = [candle]
    return ForwardPaperTradingEngine(
        exchange=exchange,
        pipeline=pipeline,
        symbols=tuple(Symbol(s) for s in symbols),
        state_store=store,
    )


@pytest.mark.asyncio
async def test_pending_orders_reserve_concurrent_slots():
    engine = make_engine(PortfolioRiskPolicy(max_concurrent_trades=3))
    report = await engine.poll_cycle()
    assert len(report.new_setups) == 3
    assert len(engine.execution_engine.pending_setups) == 3
    assert not engine.portfolio_engine.state.positions
    for symbol in engine.symbols:
        engine.execution_engine.process_market_tick(
            symbol,
            make_candle(
                1_700_086_400_000,
                high=101,
                low=99,
                close=100,
            ),
        )
    assert len(engine.portfolio_engine.state.positions) == 3
    assert not engine.execution_engine.pending_setups


@pytest.mark.asyncio
async def test_pending_orders_reduce_next_pipeline_exposure_budget():
    engine = make_engine(PortfolioRiskPolicy(max_portfolio_exposure_pct=0.15))
    await engine.poll_cycle()
    calls = engine.pipeline.evaluate.call_args_list
    # The first 1000 quote order is visible to the second risk evaluation.
    assert calls[1].kwargs["portfolio"].total_exposure == 1000.0
    assert engine.portfolio_engine.state.account.free_capital == 10000.0
    assert sum(s.quote_size for s in engine.execution_engine.pending_setups.values()) <= 1500.0


@pytest.mark.asyncio
async def test_pending_orders_consume_portfolio_heat():
    engine = make_engine(
        PortfolioRiskPolicy(
            enable_portfolio_heat=True,
            max_portfolio_heat_pct=0.005,
        )
    )
    report = await engine.poll_cycle()
    # Each proposed order risks 50 quote; the entire heat budget is 50.
    assert len(report.new_setups) == 1


@pytest.mark.asyncio
async def test_restart_retains_pending_risk_reservations(tmp_path):
    store = JsonPaperStateStore(tmp_path / "paper.json")
    policy = PortfolioRiskPolicy(max_concurrent_trades=1)
    first = make_engine(policy, symbols=("BTCUSDT",), store=store)
    await first.poll_cycle()
    restarted = make_engine(policy, symbols=("ETHUSDT",), store=store)
    report = await restarted.poll_cycle()
    assert not report.new_setups
    assert tuple(restarted.execution_engine.pending_setups) == ("BTCUSDT",)


@pytest.mark.asyncio
async def test_duplicate_pending_order_cannot_overwrite_reservation():
    engine = make_engine(PortfolioRiskPolicy(), symbols=("BTCUSDT",))
    await engine.poll_cycle()
    execution = engine.execution_engine
    original = execution.pending_setups["BTCUSDT"]
    timestamp = execution.pending_setup_timestamps["BTCUSDT"]
    assert not execution.execute_setup(replace(original, quote_size=9000), timestamp + 1)
    assert execution.pending_setups["BTCUSDT"] == original
    assert execution.pending_setup_timestamps["BTCUSDT"] == timestamp


@pytest.mark.asyncio
async def test_cancellation_releases_budget_without_changing_actual_account():
    engine = make_engine(PortfolioRiskPolicy(max_concurrent_trades=1))
    await engine.poll_cycle()
    execution = engine.execution_engine
    account = engine.portfolio_engine.state.account
    execution.cancel_setup("BTCUSDT")
    assert execution.portfolio_for_risk().total_exposure == 0
    assert not execution.pending_setup_timestamps
    assert engine.portfolio_engine.state.account == account
    assert execution.execute_setup(make_dummy_setup(symbol=Symbol("ETHUSDT")), 123)


@pytest.mark.asyncio
async def test_fill_counts_own_reservation_once_and_retains_others():
    engine = make_engine(
        PortfolioRiskPolicy(max_concurrent_trades=2), symbols=("BTCUSDT", "ETHUSDT")
    )
    await engine.poll_cycle()
    execution = engine.execution_engine
    execution.process_market_tick(
        Symbol("BTCUSDT"),
        make_candle(
            1_700_086_400_000,
            high=101,
            low=99,
            close=100,
        ),
    )
    assert len(engine.portfolio_engine.state.positions) == 1
    assert tuple(execution.pending_setups) == ("ETHUSDT",)
    assert tuple(execution.pending_setup_timestamps) == ("ETHUSDT",)
    assert len(execution.portfolio_for_risk().positions) == 2
    assert execution.portfolio_for_risk().total_exposure == 2000
    assert engine.portfolio_engine.state.account.free_capital == 9000


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["exposure", "risk", "circuit_breaker"])
async def test_fill_revalidates_after_portfolio_changes(change):
    from neon_radar.domain.portfolio import AccountState
    from neon_radar.domain.risk import DrawdownTier

    policy = PortfolioRiskPolicy(
        base_risk_per_trade_pct=0.005,
        max_portfolio_exposure_pct=0.1 if change == "exposure" else 1.0,
        enable_circuit_breaker=change == "circuit_breaker",
        drawdown_tiers=(DrawdownTier(5, 0, is_trading_halt=True),),
    )
    engine = make_engine(policy, symbols=("BTCUSDT",))
    await engine.poll_cycle()
    assert engine.execution_engine.pending_setups
    # A loss elsewhere reduces equity before the reserved order can fill.
    engine.portfolio_engine.restore_state(account=AccountState(9000, 9000))
    engine.execution_engine.process_market_tick(
        Symbol("BTCUSDT"),
        make_candle(
            1_700_086_400_000,
            high=101,
            low=99,
            close=100,
        ),
    )
    assert not engine.portfolio_engine.state.positions
    assert not engine.execution_engine.pending_setups
    assert not engine.execution_engine.pending_setup_timestamps


@pytest.mark.asyncio
async def test_live_tick_uses_same_fill_risk_guard():
    engine = make_engine(PortfolioRiskPolicy(), symbols=("BTCUSDT",))
    await engine.poll_cycle()
    engine.pipeline.risk_manager.policy = PortfolioRiskPolicy(max_portfolio_exposure_pct=0.01)
    engine.process_live_tick(Symbol("BTCUSDT"), 100.0, 1_700_086_400_000)
    assert not engine.portfolio_engine.state.positions
    assert not engine.execution_engine.pending_setups


@pytest.mark.asyncio
async def test_pending_shorts_respect_direction_specific_limit():
    from neon_radar.domain.risk import ShortRiskConstraints

    engine = make_engine(
        PortfolioRiskPolicy(
            short_constraints=ShortRiskConstraints(enabled=True, max_concurrent_shorts=1),
        )
    )
    execution = engine.execution_engine
    assert execution.execute_setup(make_dummy_setup(direction=Bias.BEARISH), 1)
    assert not execution.execute_setup(
        make_dummy_setup(
            symbol=Symbol("ETHUSDT"),
            direction=Bias.BEARISH,
        ),
        1,
    )
    assert execution.execute_setup(make_dummy_setup(symbol=Symbol("SOLUSDT")), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_recovered_order_retains_signal_time_and_can_fill(tmp_path, legacy):
    import json

    store = JsonPaperStateStore(tmp_path / "paper.json")
    first = make_engine(
        PortfolioRiskPolicy(max_concurrent_trades=1), symbols=("BTCUSDT",), store=store
    )
    await first.poll_cycle()
    signal_time = first.execution_engine.pending_setup_timestamps["BTCUSDT"]
    if legacy:
        payload = json.loads(store.file_path.read_text())
        del payload["pending_setup_timestamps"]
        store.file_path.write_text(json.dumps(payload))
    restarted = make_engine(
        PortfolioRiskPolicy(max_concurrent_trades=1), symbols=("BTCUSDT",), store=store
    )
    assert restarted.execution_engine.pending_setup_timestamps["BTCUSDT"] == signal_time
    restarted.execution_engine.process_market_tick(
        Symbol("BTCUSDT"),
        make_candle(
            signal_time + 86400000,
            high=101,
            low=99,
            close=100,
        ),
    )
    (position,) = restarted.portfolio_engine.state.positions
    assert position.signal_time == signal_time
    assert not restarted.execution_engine.pending_setups
    assert restarted.execution_engine.portfolio_for_risk().total_exposure == 1000


@pytest.mark.asyncio
async def test_legacy_overbooked_orders_cannot_bypass_fill_limit():
    engine = make_engine(PortfolioRiskPolicy(max_concurrent_trades=1))
    execution = engine.execution_engine
    # Simulate an old snapshot created before reservation checks existed.
    execution.pending_setups = {
        s: make_dummy_setup(symbol=Symbol(s)) for s in ("BTCUSDT", "ETHUSDT")
    }
    for symbol in ("BTCUSDT", "ETHUSDT"):
        execution.process_market_tick(
            Symbol(symbol),
            make_candle(
                1_700_086_400_000,
                high=101,
                low=99,
                close=100,
            ),
        )
    assert len(engine.portfolio_engine.state.positions) == 1
    assert not execution.pending_setups
