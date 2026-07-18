"""Tests for Execution Cost Model."""

from unittest.mock import MagicMock

from neon_radar.domain.enums import Bias
from neon_radar.domain.funding import FundingRate
from neon_radar.domain.models import Symbol
from neon_radar.domain.trading.execution import (
    BinanceFundingModel,
    BinanceFuturesFeeModel,
    CostModel,
    ExecutionType,
    FixedSlippageModel,
)


def test_binance_futures_fee_model():
    """Test Maker and Taker fee logic."""
    model = BinanceFuturesFeeModel(maker_fee=0.0002, taker_fee=0.0005)

    assert model.calculate_entry_fee_pct(ExecutionType.MAKER) == 0.0002
    assert model.calculate_entry_fee_pct(ExecutionType.TAKER) == 0.0005
    assert model.calculate_exit_fee_pct(ExecutionType.MAKER) == 0.0002
    assert model.calculate_exit_fee_pct(ExecutionType.TAKER) == 0.0005

def test_fixed_slippage_model():
    """Test deterministic slippage logic."""
    model = FixedSlippageModel(slippage_pct=0.0005)
    symbol = Symbol("BTCUSDT")

    # Slippage applied only on TAKER
    assert model.calculate_slippage_pct(symbol, ExecutionType.TAKER, Bias.BULLISH) == 0.0005
    assert model.calculate_slippage_pct(symbol, ExecutionType.MAKER, Bias.BULLISH) == 0.0

def test_binance_funding_model_long():
    """Test funding cost for a LONG position."""
    model = BinanceFundingModel()
    symbol = Symbol("BTCUSDT")
    provider = MagicMock()

    # EIGHT_HOURS_MS = 28800000
    # Entry at 0
    # Next boundary is 0, but technically math.ceil(0/28800000)*28800000 is 0.
    # If entry is exactly 0, next_boundary is 0.
    # To test actual boundaries, entry at 1000
    entry_time = 1000
    # 8h = 28,800,000. next boundary is 28,800,000
    # Let's say exit is 2 * 28,800,000 + 1000 = 57,601,000
    exit_time = 57601000

    # It should hit exactly 2 boundaries: 28800000 and 57600000
    provider.get_funding_rate_at.side_effect = [
        FundingRate(timestamp=28800000, symbol=symbol, rate=0.0001),
        FundingRate(timestamp=57600000, symbol=symbol, rate=0.0002),
    ]

    cost = model.calculate_funding_cost_pct(
        symbol=symbol,
        direction=Bias.BULLISH,
        entry_time=entry_time,
        exit_time=exit_time,
        provider=provider
    )

    # Long pays positive funding -> 0.0001 + 0.0002 = 0.0003
    import pytest
    assert cost == pytest.approx(0.0003)
    assert provider.get_funding_rate_at.call_count == 2

def test_binance_funding_model_short():
    """Test funding cost for a SHORT position."""
    model = BinanceFundingModel()
    symbol = Symbol("BTCUSDT")
    provider = MagicMock()

    entry_time = 1000
    exit_time = 28801000  # Just passes the first boundary

    provider.get_funding_rate_at.side_effect = [
        FundingRate(timestamp=28800000, symbol=symbol, rate=0.0001),
    ]

    cost = model.calculate_funding_cost_pct(
        symbol=symbol,
        direction=Bias.BEARISH,
        entry_time=entry_time,
        exit_time=exit_time,
        provider=provider
    )

    # Short pays negative funding. Rate is 0.0001 -> short receives it -> cost is -0.0001
    import pytest
    assert cost == pytest.approx(-0.0001)

def test_cost_model_facade():
    """Test the facade properly aggregates fees, slippage, and funding."""
    cost_model = CostModel()
    symbol = Symbol("BTCUSDT")
    provider = MagicMock()
    provider.get_funding_rate_at.return_value = FundingRate(
        timestamp=28800000, symbol=symbol, rate=0.0001
    )

    costs = cost_model.calculate_costs(
        symbol=symbol,
        direction=Bias.BULLISH,
        entry_type=ExecutionType.MAKER,
        exit_type=ExecutionType.TAKER,
        entry_time=1000,
        exit_time=28801000,
        funding_provider=provider
    )

    # Fees: Maker (0.0002) + Taker (0.0005) = 0.0007
    # Slippage: Maker (0.0) + Taker (0.0005) = 0.0005
    # Funding: 1 boundary (0.0001) for Long = 0.0001

    import pytest
    assert costs.fees_pct == pytest.approx(0.0007)
    assert costs.slippage_pct == pytest.approx(0.0005)
    assert costs.funding_pct == pytest.approx(0.0001)
    assert costs.total_costs_pct == pytest.approx(0.0013)
