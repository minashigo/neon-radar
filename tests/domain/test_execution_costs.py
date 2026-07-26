"""Tests for execution cost models."""

import pytest

from neon_radar.domain.enums import Bias
from neon_radar.domain.execution_costs import (
    BinanceFundingModel,
    BinanceFuturesFeeModel,
    ExecutionCostSummary,
    FixedSlippageModel,
)
from neon_radar.domain.models import Symbol
from neon_radar.domain.trading.execution import ExecutionType


def test_execution_cost_summary_math():
    summary = ExecutionCostSummary(
        entry_fee=1.0,
        exit_fee=1.0,
        slippage_cost=5.0,
        funding_cost=-0.5,  # we earned funding
        gross_pnl=100.0,
    )
    assert summary.total_cost == 1.0 + 1.0 + 5.0 - 0.5
    assert summary.net_pnl == 100.0 - 6.5
    assert summary.net_pnl == 93.5


def test_binance_fee_model():
    model = BinanceFuturesFeeModel(maker_fee_pct=0.0002, taker_fee_pct=0.0005)
    notional = 1000.0

    # Maker Entry
    assert model.calculate_entry_fee(notional, ExecutionType.MAKER) == 0.2
    # Taker Entry
    assert model.calculate_entry_fee(notional, ExecutionType.TAKER) == 0.5
    # Maker Exit
    assert model.calculate_exit_fee(notional, ExecutionType.MAKER) == 0.2
    # Taker Exit
    assert model.calculate_exit_fee(notional, ExecutionType.TAKER) == 0.5


def test_fixed_slippage_model():
    model = FixedSlippageModel(slippage_pct=0.001)
    notional = 1000.0
    symbol = Symbol("BTCUSDT")

    # Slippage applies to Taker
    assert model.calculate_slippage(notional, symbol, ExecutionType.TAKER, Bias.BULLISH) == 1.0
    # Slippage does not apply to Maker
    assert model.calculate_slippage(notional, symbol, ExecutionType.MAKER, Bias.BULLISH) == 0.0


class MockFundingProvider:
    def __init__(self, rates_by_time):
        self.rates_by_time = rates_by_time

    def get_funding_rate_at(self, symbol, timestamp):
        from dataclasses import dataclass

        @dataclass
        class FR:
            rate: float

        rate = self.rates_by_time.get(timestamp)
        return FR(rate) if rate is not None else None


def test_binance_funding_model_long():
    model = BinanceFundingModel()
    symbol = Symbol("BTCUSDT")
    notional = 1000.0
    8 * 60 * 60 * 1000

    # Start at 1 hr before an 8-hour boundary (e.g. 7 hours in ms)
    entry_time = 7 * 60 * 60 * 1000

    # Next boundary is 8 hours
    b1 = 8 * 60 * 60 * 1000
    b2 = 16 * 60 * 60 * 1000
    exit_time = 17 * 60 * 60 * 1000

    provider = MockFundingProvider(
        {
            b1: 0.0001,  # Positive funding
            b2: -0.0002,  # Negative funding
        }
    )

    # Bullish: Pay positive, receive negative
    # cost_pct = 0.0001 + (-0.0002) = -0.0001
    cost = model.calculate_funding_cost(
        notional, symbol, Bias.BULLISH, entry_time, exit_time, provider
    )

    assert cost == pytest.approx(-0.1)


def test_binance_funding_model_short():
    model = BinanceFundingModel()
    symbol = Symbol("BTCUSDT")
    notional = 1000.0
    8 * 60 * 60 * 1000

    entry_time = 7 * 60 * 60 * 1000
    b1 = 8 * 60 * 60 * 1000
    b2 = 16 * 60 * 60 * 1000
    exit_time = 17 * 60 * 60 * 1000

    provider = MockFundingProvider(
        {
            b1: 0.0001,  # Positive funding
            b2: -0.0002,  # Negative funding
        }
    )

    # Bearish: Receive positive, pay negative
    # cost_pct = -(0.0001) + -(-0.0002) = -0.0001 + 0.0002 = 0.0001
    cost = model.calculate_funding_cost(
        notional, symbol, Bias.BEARISH, entry_time, exit_time, provider
    )

    assert cost == pytest.approx(0.1)
