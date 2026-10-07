"""Unit tests for Binance Futures Grid Simulator."""

from research.experiments.futures_grid.simulator import (
    GridConfig,
    GridMode,
    GridSession,
    GridStatus,
    GridType,
    simulate_grid_run,
)


def test_grid_level_generation():
    """Verify arithmetic and geometric grid level formulas."""
    cfg_arith = GridConfig(
        lower_price=100.0,
        upper_price=200.0,
        num_grids=10,
        grid_mode=GridMode.ARITHMETIC,
    )
    session_a = GridSession(cfg_arith, start_time=1000, initial_price=150.0)
    assert len(session_a.levels) == 11
    assert session_a.levels[0].price == 100.0
    assert session_a.levels[5].price == 150.0
    assert session_a.levels[10].price == 200.0

    cfg_geo = GridConfig(
        lower_price=100.0,
        upper_price=200.0,
        num_grids=10,
        grid_mode=GridMode.GEOMETRIC,
    )
    session_g = GridSession(cfg_geo, start_time=1000, initial_price=150.0)
    assert len(session_g.levels) == 11
    assert abs(session_g.levels[0].price - 100.0) < 1e-4
    assert abs(session_g.levels[10].price - 200.0) < 1e-4


def test_neutral_grid_oscillation_generates_profit():
    """Oscillation across grid levels buys low, sells high, yielding positive realized grid profit."""
    cfg = GridConfig(
        lower_price=90.0,
        upper_price=110.0,
        num_grids=10,  # 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110
        initial_margin=1000.0,
        leverage=2.0,
        grid_type=GridType.NEUTRAL,
        maker_fee_rate=0.0002,
        slippage_pct=0.0,
    )
    # Start at 100.
    # Bar 1: dips to 95, closes 96 (buys 98, 96)
    # Bar 2: rises to 105, closes 102 (sells 98, 100, 102, 104)
    candles = [
        {"open_time": 1000, "open": 100.0, "high": 100.0, "low": 95.0, "close": 96.0},
        {"open_time": 2000, "open": 96.0, "high": 105.0, "low": 96.0, "close": 102.0},
    ]

    res = simulate_grid_run(cfg, candles)
    assert res.completed_grids_count > 0
    assert res.realized_grid_profit > 0
    assert res.total_fees_paid > 0
    assert res.total_fills_count >= 4


def test_downward_runaway_accumulates_long_inventory():
    """Continuous price drop fills all lower buys, creating underwater long inventory."""
    cfg = GridConfig(
        lower_price=50.0,
        upper_price=150.0,
        num_grids=10,  # steps of 10: 50, 60, 70, 80, 90, 100, 110...
        initial_margin=1000.0,
        leverage=2.0,
        stop_loss_roi=None,  # No SL
    )
    # Start at 100. Falls monotonically to 50
    candles = [
        {"open_time": 1000, "open": 100.0, "high": 100.0, "low": 75.0, "close": 75.0},
        {"open_time": 2000, "open": 75.0, "high": 75.0, "low": 50.0, "close": 50.0},
    ]

    res = simulate_grid_run(cfg, candles)
    # Has filled buys at 90, 80, 70, 60, 50
    assert res.peak_inventory_abs > 0
    # Final price 50 is below avg entry price (~70), so net PnL is negative
    assert res.net_pnl < 0


def test_regime_gate_market_close():
    """Terminating session on Regime Gate triggers market order with taker fee and slippage."""
    cfg = GridConfig(
        lower_price=90.0,
        upper_price=110.0,
        num_grids=10,
        initial_margin=1000.0,
        leverage=2.0,
        slippage_pct=0.001,
        taker_fee_rate=0.0005,
    )
    session = GridSession(cfg, start_time=1000, initial_price=100.0)

    # Process candle dipping to 95 (buys 98, 96)
    session.process_candle(
        timestamp=2000, open_price=100.0, high_price=100.0, low_price=95.0, close_price=96.0
    )
    assert session.inventory_qty > 0

    # Gate closes session at 96.0
    res = session.close(timestamp=3000, exit_price=96.0, reason="GATE")
    assert res.status == GridStatus.CLOSED_GATE
    assert session.inventory_qty == 0.0  # Fully closed
    # Last event must be CLOSED_GATE
    assert res.events[-1].event_type == "CLOSED_GATE"


def test_liquidation_under_extreme_leverage():
    """Severe market crash against accumulated long inventory triggers liquidation."""
    cfg = GridConfig(
        lower_price=80.0,
        upper_price=120.0,
        num_grids=10,
        initial_margin=1000.0,
        leverage=10.0,  # High leverage
        maintenance_margin_rate=0.01,
        stop_loss_roi=None,
    )
    # Start at 100, drops to 20
    candles = [
        {"open_time": 1000, "open": 100.0, "high": 100.0, "low": 20.0, "close": 20.0},
    ]
    res = simulate_grid_run(cfg, candles)
    assert res.status == GridStatus.LIQUIDATED
    assert res.net_pnl < -900.0  # Lost almost all capital
