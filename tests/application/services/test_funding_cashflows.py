"""Funding cash reaches the shared account before subsequent decisions, exactly once."""

import json
from dataclasses import replace
from itertools import permutations
from unittest.mock import MagicMock

import pytest
from test_forward_paper_trading import MockExchangeClient, make_dummy_setup

from neon_radar.application.services.execution import PaperExecutionEngine
from neon_radar.application.services.paper_trading_engine import ForwardPaperTradingEngine
from neon_radar.application.services.portfolio_engine import PortfolioEngine
from neon_radar.application.services.risk.drawdown import DrawdownMonitor
from neon_radar.application.services.risk.manager import RiskManager
from neon_radar.domain.enums import Bias
from neon_radar.domain.events import FundingSettled
from neon_radar.domain.execution_costs import (
    BinanceFundingModel,
    BinanceFuturesFeeModel,
    ExecutionCostSummary,
    FixedSlippageModel,
    SettlementFundingModel,
)
from neon_radar.domain.funding import FundingSettlement
from neon_radar.domain.models import OHLCV, Symbol
from neon_radar.domain.portfolio import OpenPosition
from neon_radar.domain.risk import DrawdownTier, PortfolioRiskPolicy
from neon_radar.infrastructure.storage.paper_state_store import JsonPaperStateStore

BTC, ETH, SOL = map(Symbol, ("BTCUSDT", "ETHUSDT", "SOLUSDT"))


class History:
    def __init__(self, *events):
        self.events = events
        self.calls = []

    def get_funding_events(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        return tuple(e for e in self.events if e.symbol == symbol and start < e.timestamp <= end)


def position(symbol=BTC, direction=Bias.BULLISH, quantity=10, **kwargs):
    return OpenPosition(
        symbol=symbol,
        direction=direction,
        entry_price=100,
        quantity=quantity,
        position_size=100 * quantity,
        stop_loss=50 if direction == Bias.BULLISH else 150,
        take_profit=200 if direction == Bias.BULLISH else 50,
        opened_at=0,
        **kwargs,
    )


def execution(*events, capital=10000, positions=None):
    portfolio = PortfolioEngine(capital)
    for p in (position(),) if positions is None else positions:
        portfolio.open_position(p)
    engine = PaperExecutionEngine(
        portfolio,
        fee_model=BinanceFuturesFeeModel(0, 0),
        slippage_model=FixedSlippageModel(0),
        funding_model=SettlementFundingModel(),
        funding_provider=History(*events),
    )
    return portfolio, engine


def candle(start=10, end=19, close=100):
    return OHLCV(
        open_time=start,
        close_time=end,
        open=100,
        high=max(101, close),
        low=99,
        close=close,
        volume=1,
    )


def forward(portfolio, engine, store=None):
    pipeline = MagicMock()
    pipeline.risk_manager = RiskManager()
    pipeline.evaluate.return_value = None
    return ForwardPaperTradingEngine(
        exchange=MockExchangeClient(),
        pipeline=pipeline,
        portfolio_engine=portfolio,
        execution_engine=engine,
        state_store=store,
        symbols=(BTC, ETH),
    )


@pytest.mark.parametrize(
    "direction,rate,cost",
    [
        (Bias.BULLISH, 0.01, 10),
        (Bias.BEARISH, 0.01, -10),
        (Bias.BULLISH, -0.01, -10),
        (Bias.BEARISH, -0.01, 10),
    ],
)
def test_payment_immediately_changes_equity_free_cash_and_statistics(direction, rate, cost):
    portfolio, engine = execution(
        FundingSettlement(BTC, 5, rate, 100), positions=(position(direction=direction),)
    )
    assert engine.settle_funding(5) == cost
    assert portfolio.state.account.total_capital == 10000 - cost
    assert portfolio.state.account.free_capital == 9000 - cost
    assert portfolio.state.statistics.total_funding == cost
    assert portfolio.state.positions[0].funding_paid == cost
    assert portfolio.state.positions[0].funding_processed_until == 5
    assert portfolio.history == ()
    # A mark-to-market update preserves the cash adjustment.
    portfolio.update_market_prices({str(BTC): 100})
    assert portfolio.state.account.total_capital == 10000 - cost


def test_repeated_and_overlapping_queries_publish_each_payment_once():
    portfolio, engine = execution(
        FundingSettlement(BTC, 5, 0.01, 100), FundingSettlement(BTC, 15, -0.02, 100)
    )
    published = []
    portfolio.subscribe(published.append)
    for t in [4, 5, 5, 4, 10, 15, 15, 20]:
        engine.settle_funding(t)
    assert portfolio.state.account.total_capital == 10010
    assert portfolio.state.positions[0].funding_paid == -10
    assert [e.timestamp for e in published] == [5, 15]
    assert all(isinstance(e, FundingSettled) for e in published)


@pytest.mark.parametrize("direction,cost", [(Bias.BULLISH, 30), (Bias.BEARISH, -30)])
def test_close_reports_full_funding_but_does_not_debit_it_twice(direction, cost):
    portfolio, engine = execution(
        FundingSettlement(BTC, 5, 0.01, 100),
        FundingSettlement(BTC, 20, 0.02, 100),
        positions=(position(direction=direction),),
    )
    engine.settle_funding(10)
    engine._close_at(portfolio.state.positions[0], 100, "TAKE_PROFIT", 20)
    assert portfolio.history[0].execution_summary.funding_cost == cost
    assert portfolio.history[0].net_pnl == -cost
    assert portfolio.state.account.total_capital == 10000 - cost
    assert portfolio.state.account.free_capital == 10000 - cost
    assert portfolio.state.statistics.total_funding == cost


def test_direct_terminal_close_reconciles_already_paid_cash():
    portfolio, engine = execution(FundingSettlement(BTC, 5, 0.01, 100))
    engine.settle_funding(5)
    portfolio.close_position(str(BTC), 100, 10, ExecutionCostSummary(1, 2, 3, 10, 0))
    assert portfolio.state.account.total_capital == 9984
    assert portfolio.state.statistics.total_funding == 10
    assert portfolio.history[0].net_pnl == -16


@pytest.mark.parametrize("order", list(permutations((BTC, ETH, SOL))))
def test_simultaneous_cross_asset_payments_net_atomically_in_every_order(order):
    events = {
        s: FundingSettlement(s, 5, rate, 100)
        for s, rate in ((BTC, 0.01), (ETH, -0.02), (SOL, 0.01))
    }
    # Only 1 USDT free: sequential debits would fail despite zero group cost.
    portfolio, engine = execution(
        *(events[s] for s in order), capital=3001, positions=tuple(position(s) for s in order)
    )
    assert engine.settle_funding(5) == 0
    assert portfolio.state.account.free_capital == 1
    assert portfolio.state.statistics.total_funding == 0
    assert {p.symbol: p.funding_paid for p in portfolio.state.positions} == {
        BTC: 10,
        ETH: -20,
        SOL: 10,
    }
    assert {p.funding_processed_until for p in portfolio.state.positions} == {5}


def test_funding_at_open_blocks_fill_using_reduced_risk_budget():
    portfolio, engine = execution(FundingSettlement(BTC, 10, 0.2, 100))
    engine.risk_manager = RiskManager(PortfolioRiskPolicy(base_risk_per_trade_pct=0.01))
    setup = make_dummy_setup(symbol=ETH, quote_size=1000, sl=90.1, tp=120)
    assert engine.execute_setup(setup, 1)  # risk 99 <= 100 before payment
    fills = engine.process_market_batch({BTC: candle(), ETH: candle()})
    assert fills == ()  # equity 9800 -> risk budget 98
    assert {p.symbol for p in portfolio.state.positions} == {BTC}
    assert str(ETH) not in engine.pending_setups
    assert portfolio.state.account.total_capital == 9800


def test_interior_funding_does_not_retroactively_cancel_open_fill():
    portfolio, engine = execution(FundingSettlement(BTC, 15, 0.2, 100))
    engine.risk_manager = RiskManager(PortfolioRiskPolicy(base_risk_per_trade_pct=0.01))
    assert engine.execute_setup(make_dummy_setup(symbol=ETH, quote_size=1000, sl=90.1, tp=120), 1)
    fills = engine.process_market_batch({BTC: candle(), ETH: candle()})
    assert tuple(p.symbol for p in fills) == (ETH,)
    assert portfolio.state.account.total_capital == 9800


def test_future_credit_does_not_finance_earlier_open_entry():
    portfolio, engine = execution(FundingSettlement(BTC, 15, -0.2, 100), capital=1000, positions=())
    assert engine.execute_setup(make_dummy_setup(symbol=ETH, quote_size=200), 0)
    portfolio.open_position(position(quantity=9))  # simulate budget changing after registration
    assert engine.process_market_batch({BTC: candle(), ETH: candle()}) == ()
    assert portfolio.state.account.free_capital == 280
    assert {p.symbol for p in portfolio.state.positions} == {BTC}


def test_credit_at_open_can_finance_later_entry():
    portfolio, engine = execution(FundingSettlement(BTC, 10, -0.2, 100), capital=1000, positions=())
    assert engine.execute_setup(make_dummy_setup(symbol=ETH, quote_size=200), 0)
    portfolio.open_position(position(quantity=9))
    assert tuple(p.symbol for p in engine.process_market_batch({BTC: candle(), ETH: candle()})) == (
        ETH,
    )
    assert portfolio.state.account.free_capital == 80


def test_funding_updates_drawdown_before_pending_fill():
    portfolio, engine = execution(FundingSettlement(BTC, 10, 0.6, 100))
    monitor = DrawdownMonitor()
    engine.risk_manager = RiskManager(
        PortfolioRiskPolicy(
            base_risk_per_trade_pct=0.01,
            enable_circuit_breaker=True,
            drawdown_tiers=(DrawdownTier(5, 0, is_trading_halt=True),),
        )
    )
    engine.drawdown_provider = lambda: monitor.update(portfolio.state.account.total_capital, 10)
    assert engine.execute_setup(make_dummy_setup(symbol=ETH, quote_size=500, sl=99), 1)
    assert (
        engine.process_market_batch({BTC: candle(), ETH: candle()}, on_equity_update=monitor.update)
        == ()
    )
    assert monitor.max_drawdown_pct == 6


def test_interior_payment_observation_uses_no_future_close_pnl():
    _portfolio, engine = execution(FundingSettlement(BTC, 15, 0.02, 100))
    observations = []
    engine.process_market_batch(
        {BTC: candle(close=110)}, on_equity_update=lambda eq, t: observations.append((t, eq))
    )
    assert (15, 9980) in observations
    assert observations[-1] == (19, 10080)
    assert [t for t, _ in observations] == sorted(t for t, _ in observations)


def test_new_entry_at_settlement_timestamp_is_excluded():
    portfolio, engine = execution(
        FundingSettlement(BTC, 10, 0.9, 100), FundingSettlement(BTC, 15, 0.01, 100), positions=()
    )
    assert engine.execute_setup(make_dummy_setup(symbol=BTC, quote_size=1000, sl=90, tp=120), 1)
    engine.process_market_batch({BTC: candle()})
    assert portfolio.state.positions[0].funding_paid == 10
    assert portfolio.state.account.total_capital == 9990


def test_same_bar_exit_uses_existing_open_timestamp_convention():
    portfolio, engine = execution(FundingSettlement(BTC, 15, 0.5, 100), positions=())
    assert engine.execute_setup(make_dummy_setup(symbol=BTC, quote_size=1000, sl=99.5, tp=120), 1)
    engine.process_market_batch({BTC: candle()})
    assert portfolio.history[0].exit_time == 10
    assert portfolio.history[0].execution_summary.funding_cost == 0
    assert portfolio.state.statistics.total_funding == 0


def test_all_symbol_histories_are_validated_before_first_cash_mutation():
    portfolio, engine = execution(
        FundingSettlement(BTC, 5, 0.01, 100), positions=(position(), position(ETH))
    )
    original = engine.funding_provider.get_funding_events

    def fetch(symbol, start, end):
        if symbol == ETH:
            raise ValueError("history unavailable")
        return original(symbol, start, end)

    engine.funding_provider.get_funding_events = fetch
    before = portfolio.state
    with pytest.raises(ValueError, match="unavailable"):
        engine.settle_funding(10)
    assert portfolio.state == before


def test_pending_entry_history_failure_leaves_prices_cash_and_order_unchanged():
    portfolio, engine = execution()
    assert engine.execute_setup(make_dummy_setup(symbol=ETH, quote_size=500), 1)
    original = engine.funding_provider.get_funding_events

    def fetch(symbol, start, end):
        if symbol == ETH:
            raise ValueError("no ETH coverage")
        return original(symbol, start, end)

    engine.funding_provider.get_funding_events = fetch
    before = portfolio.state
    pending = dict(engine.pending_setups)
    with pytest.raises(ValueError, match="coverage"):
        engine.process_market_batch({BTC: candle(close=110), ETH: candle()})
    assert portfolio.state == before
    assert engine.pending_setups == pending


def test_cash_shortfall_does_not_waive_funding_or_advance_cursor():
    portfolio, engine = execution(FundingSettlement(BTC, 5, 1, 100), capital=1001)
    before = portfolio.state
    with pytest.raises(ValueError, match="insolvency"):
        engine.settle_funding(5)
    assert portfolio.state == before


def test_close_before_already_booked_funding_rejected_before_mutation():
    portfolio, engine = execution(FundingSettlement(BTC, 5, 0.01, 100))
    engine.settle_funding(5)
    before = portfolio.state
    with pytest.raises(ValueError, match="before already booked"):
        portfolio.close_position(str(BTC), 100, 4, ExecutionCostSummary(0, 0, 0, 10, 0))
    assert portfolio.state == before
    assert portfolio.history == ()


def test_legacy_model_cannot_reinterpret_position_even_when_net_paid_is_zero():
    portfolio, engine = execution(
        FundingSettlement(BTC, 5, 0.01, 100), FundingSettlement(BTC, 6, -0.01, 100)
    )
    engine.settle_funding(6)
    assert portfolio.state.positions[0].funding_paid == 0
    engine.funding_model = BinanceFundingModel()
    before = portfolio.state
    with pytest.raises(ValueError, match="switch funded"):
        engine.process_market_batch({BTC: candle()})
    assert portfolio.state == before


def test_old_bar_cannot_replay_decisions_before_an_already_booked_payment():
    portfolio, engine = execution(FundingSettlement(BTC, 15, 0.01, 100))
    engine.settle_funding(15)
    before = portfolio.state
    with pytest.raises(ValueError, match="cannot replay"):
        engine.process_market_batch({BTC: candle()})
    assert portfolio.state == before


def test_restart_preserves_cursor_paid_cash_statistics_and_final_pnl(tmp_path):
    events = (FundingSettlement(BTC, 5, 0.01, 100), FundingSettlement(BTC, 15, 0.02, 100))
    portfolio, engine = execution(*events)
    store = JsonPaperStateStore(tmp_path / "state.json")
    fwd = forward(portfolio, engine, store)
    engine.settle_funding(5, on_equity_update=fwd.drawdown_monitor.update)
    fwd.save_state(5)
    restored_portfolio, restored_engine = execution(*events, positions=())
    restored = forward(restored_portfolio, restored_engine, store)
    assert restored_portfolio.state == portfolio.state
    assert restored.drawdown_monitor.max_drawdown_pct == fwd.drawdown_monitor.max_drawdown_pct
    assert restored_engine.settle_funding(5) == 0
    for p, ex in ((portfolio, engine), (restored_portfolio, restored_engine)):
        ex._close_at(p.state.positions[0], 100, "TAKE_PROFIT", 20)
    assert restored_portfolio.state == portfolio.state
    assert restored_portfolio.history == portfolio.history
    assert restored_portfolio.state.account.total_capital == 9970


def test_wrong_model_on_restart_requires_explicit_migration_even_with_no_open_positions(tmp_path):
    portfolio, engine = execution(positions=())
    store = JsonPaperStateStore(tmp_path / "state.json")
    forward(portfolio, engine, store).save_state(1)
    legacy = PaperExecutionEngine(PortfolioEngine(10000))
    with pytest.raises(ValueError, match="migration"):
        forward(legacy.portfolio_engine, legacy, store)


def test_legacy_v1_snapshot_defaults_and_statistics_reconstruction(tmp_path):
    portfolio, engine = execution(FundingSettlement(BTC, 5, 0.01, 100))
    engine._close_at(portfolio.state.positions[0], 100, "TAKE_PROFIT", 10)
    store = JsonPaperStateStore(tmp_path / "state.json")
    forward(portfolio, engine, store).save_state(10)
    payload = json.loads(store.file_path.read_text())
    payload["version"] = 1
    payload.pop("funding_mode")
    payload.pop("statistics")
    store.file_path.write_text(json.dumps(payload))
    new = PortfolioEngine(10000)
    forward(new, PaperExecutionEngine(new), store)
    assert new.state.statistics.total_funding == 10
    assert new.state.statistics.total_trades == 1
    assert new.state.account.total_capital == 9990


def test_point_tick_books_funding_even_without_entry_or_exit():
    portfolio, engine = execution(FundingSettlement(BTC, 5, 0.01, 100))
    fwd = forward(portfolio, engine)
    fwd.process_live_tick(BTC, 100, 10)
    fwd.process_live_tick(BTC, 100, 10)
    assert portfolio.state.account.total_capital == 9990
    assert portfolio.state.statistics.total_funding == 10
    assert fwd.drawdown_monitor.max_drawdown_pct == 0.1


@pytest.mark.parametrize(
    "fields",
    [
        {"funding_paid": float("nan")},
        {"funding_paid": 1},
        {"funding_processed_until": 0},
        {"funding_processed_until": True},
    ],
)
def test_invalid_position_funding_state_is_rejected(fields):
    with pytest.raises(ValueError, match=r"[Ff]unding|funding"):
        replace(position(), **fields)


def test_batch_result_and_funding_observations_ignore_input_order():
    reference = None
    for order in permutations((BTC, ETH, SOL)):
        portfolio, engine = execution(
            FundingSettlement(BTC, 10, 0.01, 100),
            FundingSettlement(ETH, 10, -0.02, 100),
            FundingSettlement(SOL, 15, 0.01, 100),
            positions=tuple(position(s) for s in (BTC, ETH, SOL)),
        )
        observations = []
        fills = engine.process_market_batch(
            {s: candle() for s in order},
            on_equity_update=lambda eq, t, target=observations: target.append((t, eq)),
        )
        result = portfolio.state, portfolio.history, fills, observations
        if reference is None:
            reference = result
        assert result == reference


def test_old_open_position_snapshot_defaults_to_unpaid_legacy_state(tmp_path):
    portfolio = PortfolioEngine(10000)
    portfolio.open_position(position())
    store = JsonPaperStateStore(tmp_path / "state.json")
    forward(portfolio, PaperExecutionEngine(portfolio), store).save_state(10)
    payload = json.loads(store.file_path.read_text())
    payload["version"] = 1
    payload.pop("funding_mode")
    payload.pop("statistics")
    for pos in payload["open_positions"]:
        pos.pop("funding_paid")
        pos.pop("funding_processed_until")
    store.file_path.write_text(json.dumps(payload))
    new = PortfolioEngine(10000)
    forward(new, PaperExecutionEngine(new), store)
    assert new.state.positions[0] == position()
    assert new.state.account == portfolio.state.account


def test_invalid_simultaneous_group_is_atomic():
    portfolio, _engine = execution()
    before = portfolio.state
    with pytest.raises(ValueError, match="missing position"):
        portfolio.apply_funding_settlements(
            (
                FundingSettlement(BTC, 5, 0.01, 100),
                FundingSettlement(ETH, 5, 0.01, 100),
            )
        )
    assert portfolio.state == before


def test_open_settlement_observes_current_open_prices_without_false_drawdown():
    portfolio, engine = execution(FundingSettlement(BTC, 10, 0.05, 100))
    observations = []
    monitor = DrawdownMonitor()

    def observe(eq, time):
        observations.append((time, eq))
        monitor.update(eq, time)

    engine.process_market_batch(
        {BTC: OHLCV(open_time=10, close_time=19, open=110, high=111, low=109, close=110, volume=1)},
        on_equity_update=observe,
    )
    assert observations[0] == (10, 10050)
    assert monitor.max_drawdown_pct == 0
    assert portfolio.state.account.total_capital == 10050


def test_cash_shortfall_halts_future_decisions_and_survives_restart(tmp_path):
    events = (FundingSettlement(BTC, 15, 1, 100),)
    portfolio, engine = execution(*events, capital=1500)
    assert engine.execute_setup(make_dummy_setup(symbol=ETH, quote_size=200), 1)
    with pytest.raises(ValueError, match="insolvency"):
        engine.process_market_batch({BTC: candle(), ETH: candle()})
    assert engine.funding_halt_reason is not None
    before = portfolio.state
    for operation in (
        lambda: engine.process_market_batch({BTC: candle(), ETH: candle()}),
        lambda: engine.execute_setup(make_dummy_setup(symbol=SOL, quote_size=50), 20),
        lambda: engine.settle_funding(20),
    ):
        with pytest.raises(ValueError, match="halted"):
            operation()
    assert portfolio.state == before
    store = JsonPaperStateStore(tmp_path / "state.json")
    forward(portfolio, engine, store).save_state(20)
    new_portfolio, new_engine = execution(*events, positions=())
    forward(new_portfolio, new_engine, store)
    assert new_engine.funding_halt_reason == engine.funding_halt_reason
    assert new_portfolio.state == replace(before, timestamp=20)
    with pytest.raises(ValueError, match="halted"):
        new_engine.settle_funding(30)
