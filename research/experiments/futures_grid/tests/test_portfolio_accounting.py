"""Unit tests for Continuous Portfolio & Capital Accounting Engine."""

import pytest
from research.experiments.futures_grid.portfolio import (
    PortfolioAccount,
)


def test_single_wallet_balance_carries_forward():
    """Wallet balance must persist across transactions without artificial reset."""
    acct = PortfolioAccount(initial_capital=10000.0)
    assert acct.wallet_balance == 10000.0
    assert acct.equity({"BTCUSDT": 100.0}) == 10000.0

    # Simulate a realized loss of 2000 USDT
    inv = acct.get_inventory("BTCUSDT")
    inv.qty = 10.0
    inv.avg_entry = 100.0
    inv.allocated_margin = 1000.0

    # Market close at 80.0 (loss of 200 USDT + taker fee)
    # Price = 80, fill_p = 80 * 0.999 = 79.92
    # PnL = 10 * (79.92 - 100) = -200.8
    # Fee = 10 * 79.92 * 0.0005 = 0.3996
    net_pnl = acct.market_close_inventory("BTCUSDT", price=80.0, reason="GATE")
    assert net_pnl < -200.0
    assert acct.wallet_balance < 9800.0
    # Next check: wallet balance is permanently reduced, NOT reset to 10,000
    assert acct.wallet_balance == pytest.approx(10000.0 + net_pnl, rel=1e-5)


def test_margin_allocation_gating():
    """Available margin must respect existing allocations and remaining equity."""
    acct = PortfolioAccount(initial_capital=2000.0)

    # Allocate 1500 to ETH
    alloc1 = acct.allocate_margin("ETHUSDT", requested_margin=1500.0, prices={})
    assert alloc1 == 1500.0
    assert acct.total_margin_allocated() == 1500.0
    assert acct.available_margin({}) == 500.0

    # Try to allocate 1000 to SOL (only 500 available)
    alloc2 = acct.allocate_margin("SOLUSDT", requested_margin=1000.0, prices={})
    assert alloc2 == 500.0
    assert acct.available_margin({}) == 0.0

    # Releasing ETH restores available margin
    acct.release_margin("ETHUSDT")
    assert acct.total_margin_allocated() == 500.0
    assert acct.available_margin({}) == 1500.0


def test_simultaneous_positions_share_capital_pool():
    """Unrealized PnL from multiple concurrent positions aggregates continuously."""
    acct = PortfolioAccount(initial_capital=10000.0)

    btc_inv = acct.get_inventory("BTCUSDT")
    btc_inv.qty = 1.0
    btc_inv.avg_entry = 50000.0

    eth_inv = acct.get_inventory("ETHUSDT")
    eth_inv.qty = -10.0  # Short
    eth_inv.avg_entry = 3000.0

    # Current prices: BTC up to 52,000 (+2,000), ETH up to 3,100 (-1,000 for short)
    prices = {"BTCUSDT": 52000.0, "ETHUSDT": 3100.0}
    upnl = acct.unrealized_pnl(prices)
    # BTC upnl = 1 * (52000 - 50000) = +2000
    # ETH upnl = -10 * (3100 - 3000) = -1000
    # Net upnl = +1000
    assert upnl == pytest.approx(1000.0)
    assert acct.equity(prices) == pytest.approx(11000.0)


def test_bankruptcy_and_trading_termination():
    """When equity drops to maintenance margin or zero, account is liquidated and halted."""
    acct = PortfolioAccount(initial_capital=1000.0, maintenance_margin_rate=0.01)

    inv = acct.get_inventory("SOLUSDT")
    inv.qty = 50.0
    inv.avg_entry = 100.0  # Notional = 5000 USDT (5x leverage)

    # Price drops to 75 -> Loss = 50 * (75 - 100) = -1250 USDT
    # Equity = 1000 - 1250 = -250 <= MM
    prices = {"SOLUSDT": 75.0}
    acct.step(timestamp=1000, prices=prices)

    assert acct.is_bankrupt is True
    assert acct.liquidation_events_count == 1
    assert acct.equity(prices) == 0.0
    assert acct.return_pct(prices) == -1.0  # Strictly bounded at -100%
    assert acct.return_pct(prices) >= -1.0

    # Further allocation attempts must fail
    alloc = acct.allocate_margin("BTCUSDT", 500.0, prices)
    assert alloc == 0.0


def test_continuous_max_drawdown_tracking():
    """Max drawdown must capture intraday equity peaks and troughs."""
    acct = PortfolioAccount(initial_capital=10000.0)

    # Peak equity reaches 12,000
    acct.wallet_balance = 12000.0
    acct.step(timestamp=1000, prices={})
    assert acct.peak_equity == 12000.0
    assert acct.max_drawdown_pct == 0.0

    # Drawdown down to 9,000: DD = (12000 - 9000) / 12000 = 25%
    acct.wallet_balance = 9000.0
    acct.step(timestamp=2000, prices={})
    assert acct.max_drawdown_pct == pytest.approx(0.25)

    # Rebound to 10,000: Max DD should remain 25%
    acct.wallet_balance = 10000.0
    acct.step(timestamp=3000, prices={})
    assert acct.max_drawdown_pct == pytest.approx(0.25)
