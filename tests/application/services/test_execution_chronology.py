"""Execution chronology regressions; OHLC ambiguity uses an explicit conservative policy."""

from unittest.mock import MagicMock

import pytest
from test_forward_paper_trading import make_candle, make_dummy_setup

from neon_radar.application.services.execution import PaperExecutionEngine
from neon_radar.application.services.portfolio_engine import PortfolioEngine
from neon_radar.domain.enums import Bias
from neon_radar.domain.execution_costs import BinanceFundingModel
from neon_radar.domain.funding import FundingRate
from neon_radar.domain.models import Symbol
from neon_radar.domain.portfolio import OpenPosition
from neon_radar.infrastructure.providers.binance_funding import BinanceHistoricalFundingProvider

SYM = Symbol("BTCUSDT")
EIGHT_HOURS = 8 * 3600 * 1000


def opened_position(direction):
    portfolio = PortfolioEngine(10000)
    portfolio.open_position(
        OpenPosition(
            symbol=SYM,
            direction=direction,
            entry_price=100,
            quantity=1,
            position_size=100,
            stop_loss=90 if direction == Bias.BULLISH else 110,
            take_profit=110 if direction == Bias.BULLISH else 90,
            opened_at=1,
        )
    )
    return portfolio, PaperExecutionEngine(portfolio)


@pytest.mark.parametrize(
    "direction,open_price,expected",
    [
        (Bias.BULLISH, 120, 120),
        (Bias.BEARISH, 80, 80),
    ],
)
def test_existing_gap_tp_precedes_later_stop(direction, open_price, expected):
    portfolio, execution = opened_position(direction)
    execution.process_market_tick(
        SYM, make_candle(1000, open_price=open_price, high=125, low=75, close=100)
    )
    (closed,) = portfolio.history
    assert closed.close_reason == "TAKE_PROFIT"
    assert closed.exit_price == expected


@pytest.mark.parametrize(
    "direction,open_price,close,expected",
    [
        (Bias.BULLISH, 120, 115, 110),
        (Bias.BEARISH, 80, 85, 90),
    ],
)
def test_intrabar_entry_cannot_exit_at_pre_entry_open(direction, open_price, close, expected):
    portfolio = PortfolioEngine(10000)
    execution = PaperExecutionEngine(portfolio)
    execution.execute_setup(
        make_dummy_setup(
            direction=direction,
            sl=90 if direction == Bias.BULLISH else 110,
            tp=110 if direction == Bias.BULLISH else 90,
        ),
        0,
    )
    execution.process_market_tick(
        SYM,
        make_candle(
            1000,
            open_price=open_price,
            high=max(open_price, close, 101),
            low=min(open_price, close, 99),
            close=close,
        ),
    )
    (closed,) = portfolio.history
    assert closed.exit_price == expected


@pytest.mark.parametrize(
    "direction,open_price,high,low",
    [
        (Bias.BULLISH, 120, 125, 99),
        (Bias.BEARISH, 80, 101, 75),
    ],
)
def test_pre_entry_extreme_cannot_prove_take_profit(direction, open_price, high, low):
    portfolio = PortfolioEngine(10000)
    execution = PaperExecutionEngine(portfolio)
    execution.execute_setup(
        make_dummy_setup(
            direction=direction,
            sl=90 if direction == Bias.BULLISH else 110,
            tp=110 if direction == Bias.BULLISH else 90,
        ),
        0,
    )
    execution.process_market_tick(
        SYM, make_candle(1000, open_price=open_price, high=high, low=low, close=100)
    )
    assert not portfolio.history
    assert len(portfolio.state.positions) == 1


@pytest.mark.parametrize(
    "direction,open_price,high,low,stop,target",
    [
        (Bias.BULLISH, 90, 95, 85, 80, 120),
        (Bias.BEARISH, 110, 115, 105, 120, 80),
    ],
)
def test_resting_limit_can_fill_at_better_open_without_limit_touch(
    direction, open_price, high, low, stop, target
):
    portfolio = PortfolioEngine(10000)
    execution = PaperExecutionEngine(portfolio)
    execution.execute_setup(make_dummy_setup(direction=direction, sl=stop, tp=target), 0)
    execution.process_market_tick(
        SYM, make_candle(1000, open_price=open_price, high=high, low=low, close=open_price)
    )
    (position,) = portfolio.state.positions
    assert position.entry_price == open_price
    assert position.position_size == position.quantity * open_price
    assert position.entry_fee == pytest.approx(position.position_size * 0.0002)
    assert portfolio.state.account.free_capital == pytest.approx(10000 - position.position_size)
    assert not execution.pending_setups


def test_new_fill_is_marked_at_bar_close():
    portfolio = PortfolioEngine(10000)
    execution = PaperExecutionEngine(portfolio)
    execution.execute_setup(make_dummy_setup(), 0)
    execution.process_market_tick(
        SYM, make_candle(1000, open_price=100, high=103, low=99, close=102)
    )
    assert portfolio.state.positions[0].unrealized_pnl == 2
    assert portfolio.state.account.total_capital == 10002


@pytest.mark.parametrize("direction", [Bias.BULLISH, Bias.BEARISH])
def test_zero_duration_trade_has_no_funding(direction):
    provider = MagicMock()
    provider.get_funding_rate_at.return_value = FundingRate(SYM, 0.0001, timestamp=EIGHT_HOURS)
    assert (
        BinanceFundingModel().calculate_funding_cost(
            1000, SYM, direction, EIGHT_HOURS, EIGHT_HOURS, provider
        )
        == 0
    )
    provider.get_funding_rate_at.assert_not_called()


def test_funding_settlement_precedes_new_entry_at_same_boundary():
    provider = MagicMock()
    provider.get_funding_rate_at.side_effect = lambda symbol, t: FundingRate(
        SYM, 0.0001 if t == EIGHT_HOURS else 0.0002, timestamp=t
    )
    cost = BinanceFundingModel().calculate_funding_cost(
        1000, SYM, Bias.BULLISH, EIGHT_HOURS, 2 * EIGHT_HOURS, provider
    )
    assert cost == pytest.approx(0.2)
    provider.get_funding_rate_at.assert_called_once_with(SYM, 2 * EIGHT_HOURS)


def test_historical_funding_lookup_never_returns_future_record():
    provider = BinanceHistoricalFundingProvider()
    provider._rates[str(SYM)] = [FundingRate(SYM, 0.0001, timestamp=EIGHT_HOURS)]
    assert provider.get_funding_rate_at(SYM, EIGHT_HOURS - 1) is None
    assert provider.get_funding_rate_at(SYM, EIGHT_HOURS).rate == 0.0001


@pytest.mark.parametrize(
    "direction,open_price,stop,target,close",
    [
        (Bias.BULLISH, 120, 90, 110, 115),
        (Bias.BEARISH, 80, 110, 90, 85),
    ],
)
def test_intrabar_entry_stop_still_precedes_uncertain_target(
    direction, open_price, stop, target, close
):
    portfolio = PortfolioEngine(10000)
    execution = PaperExecutionEngine(portfolio)
    execution.execute_setup(make_dummy_setup(direction=direction, sl=stop, tp=target), 0)
    execution.process_market_tick(
        SYM, make_candle(1000, open_price=open_price, high=125, low=75, close=close)
    )
    (closed,) = portfolio.history
    assert closed.close_reason == "STOP_LOSS"
    assert closed.exit_price == stop


@pytest.mark.parametrize(
    "direction,open_price,expected",
    [
        (Bias.BULLISH, 80, 80),
        (Bias.BEARISH, 120, 120),
    ],
)
def test_existing_gap_stop_precedes_later_target(direction, open_price, expected):
    portfolio, execution = opened_position(direction)
    execution.process_market_tick(
        SYM, make_candle(1000, open_price=open_price, high=125, low=75, close=100)
    )
    (closed,) = portfolio.history
    assert closed.close_reason == "STOP_LOSS"
    assert closed.exit_price == expected


def test_favorable_short_gap_revalidates_actual_larger_quote_size():
    from neon_radar.application.services.risk.manager import RiskManager
    from neon_radar.domain.risk import PortfolioRiskPolicy

    portfolio = PortfolioEngine(10000)
    execution = PaperExecutionEngine(
        portfolio,
        risk_manager=RiskManager(
            PortfolioRiskPolicy(max_portfolio_exposure_pct=0.01),
        ),
    )
    assert execution.execute_setup(make_dummy_setup(direction=Bias.BEARISH, sl=120, tp=80), 0)
    execution.process_market_tick(
        SYM, make_candle(1000, open_price=110, high=115, low=105, close=110)
    )
    assert not portfolio.state.positions
    assert not execution.pending_setups
    assert not execution.pending_setup_timestamps


@pytest.mark.parametrize(
    "direction,price,stop,target",
    [
        (Bias.BULLISH, 99, 90, 110),
        (Bias.BEARISH, 101, 110, 90),
    ],
)
def test_live_tick_fills_limit_at_better_price(direction, price, stop, target):
    from neon_radar.application.services.paper_trading_engine import ForwardPaperTradingEngine

    pipeline = MagicMock()
    engine = ForwardPaperTradingEngine(exchange=MagicMock(), pipeline=pipeline)
    assert engine.execution_engine.execute_setup(
        make_dummy_setup(direction=direction, sl=stop, tp=target), 0
    )
    engine.process_live_tick(SYM, price, 1000)
    (position,) = engine.portfolio_engine.state.positions
    assert position.entry_price == price
    assert position.position_size == price * position.quantity
    assert not engine.execution_engine.pending_setups
