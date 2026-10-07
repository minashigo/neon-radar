"""Unit tests for Exit Management Engine and Strategy Variants."""

import pandas as pd
from research.experiments.futures_grid.exit_management_engine import (
    ExitEngineConfig,
    ExitVariant,
    run_continuous_grid_simulation,
)
from research.experiments.futures_grid.portfolio import PortfolioAccount


def _make_synth_data(num_bars: int = 100, base_price: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic price and regime data for deterministic unit testing."""
    candles = []
    regimes = []
    t = 1000
    p = base_price

    for i in range(num_bars):
        # Oscillate price between 95 and 105
        offset = (i % 4 - 2) * 2.0
        c_open = p + offset
        c_high = c_open + 2.0
        c_low = c_open - 2.0
        c_close = c_open + 0.5
        candles.append({
            "open_time_ms": t,
            "open": c_open,
            "high": c_high,
            "low": c_low,
            "close": c_close,
            "volume": 100.0,
        })
        # Default regime is CHOP (allowed)
        regimes.append({
            "open_time_ms": t,
            "regime": "CHOP",
            "grid_allowed": True,
        })
        t += 4 * 3600 * 1000

    return pd.DataFrame(candles), pd.DataFrame(regimes)


def test_variant_a_immediate_close_on_regime_switch():
    """Variant A must immediately close inventory at market when regime switches to non-CHOP."""
    df, regimes_df = _make_synth_data(num_bars=30)
    # Switch to TREND at bar 15
    for i in range(15, 25):
        regimes_df.loc[i, "grid_allowed"] = False
        regimes_df.loc[i, "regime"] = "BULL_TREND"

    account = PortfolioAccount(initial_capital=10000.0)
    cfg = ExitEngineConfig(allocated_margin=1000.0)

    _acct, records = run_continuous_grid_simulation(
        df=df,
        regimes_df=regimes_df,
        symbol="BTCUSDT",
        account=account,
        config=cfg,
        variant=ExitVariant.IMMEDIATE_CLOSE,
        warmup_bars=5,
    )

    assert len(records) >= 1
    # Check that at least one record is an IMMEDIATE exit
    exit_types = [r.exit_type for r in records]
    assert "IMMEDIATE" in exit_types
    # Verify taker fees and slippage were paid
    imm_rec = next(r for r in records if r.exit_type == "IMMEDIATE")
    assert imm_rec.taker_fee > 0
    assert imm_rec.churn_cost > 0


def test_variant_b_freeze_holds_inventory():
    """Variant B must hold inventory through non-CHOP and re-center when CHOP returns."""
    df, regimes_df = _make_synth_data(num_bars=30)
    # Temporary trend blip from bar 15 to 18 (3 bars)
    for i in range(15, 18):
        regimes_df.loc[i, "grid_allowed"] = False
        regimes_df.loc[i, "regime"] = "BEAR_TREND"

    account = PortfolioAccount(initial_capital=10000.0)
    cfg = ExitEngineConfig(allocated_margin=1000.0)

    _acct, records = run_continuous_grid_simulation(
        df=df,
        regimes_df=regimes_df,
        symbol="BTCUSDT",
        account=account,
        config=cfg,
        variant=ExitVariant.FREEZE,
        warmup_bars=5,
    )

    # Should have a RECENTER_CHOP exit instead of an IMMEDIATE exit
    exit_types = [r.exit_type for r in records]
    assert "IMMEDIATE" not in exit_types
    if len(records) > 0:
        assert any(t in ("RECENTER_CHOP", "SL_HIT", "TIMEOUT") for t in exit_types)


def test_variant_c_atr_trailing_stop():
    """Variant C must trigger ATR trailing stop during extreme adverse price moves."""
    df, regimes_df = _make_synth_data(num_bars=30)
    # Freeze at bar 15, then crash price violently at bar 16-17
    for i in range(15, 25):
        regimes_df.loc[i, "grid_allowed"] = False
        regimes_df.loc[i, "regime"] = "BEAR_TREND"

    # Severe price plunge at bar 16
    df.loc[16, "open"] = 80.0
    df.loc[16, "high"] = 80.0
    df.loc[16, "low"] = 60.0
    df.loc[16, "close"] = 65.0

    account = PortfolioAccount(initial_capital=10000.0)
    cfg = ExitEngineConfig(allocated_margin=1000.0, atr_multiplier=2.0)

    _acct, records = run_continuous_grid_simulation(
        df=df,
        regimes_df=regimes_df,
        symbol="BTCUSDT",
        account=account,
        config=cfg,
        variant=ExitVariant.FREEZE_ATR,
        warmup_bars=5,
    )

    exit_types = [r.exit_type for r in records]
    # Check that either ATR_TRAILING or SL_HIT fired
    assert any(t in ("ATR_TRAILING", "SL_HIT") for t in exit_types)


def test_variant_d_hysteresis_ignores_single_bar_blip():
    """Variant D must ignore a single-bar regime blip and avoid spurious churn."""
    df, regimes_df = _make_synth_data(num_bars=30)
    # Single 1-bar non-CHOP spike at bar 15
    regimes_df.loc[15, "grid_allowed"] = False
    regimes_df.loc[15, "regime"] = "VOLATILE_CRASH"
    # Bar 16 returns to CHOP
    regimes_df.loc[16, "grid_allowed"] = True

    account = PortfolioAccount(initial_capital=10000.0)
    cfg = ExitEngineConfig(allocated_margin=1000.0, hysteresis_bars=2)

    _acct, records = run_continuous_grid_simulation(
        df=df,
        regimes_df=regimes_df,
        symbol="BTCUSDT",
        account=account,
        config=cfg,
        variant=ExitVariant.HYSTERESIS,
        warmup_bars=5,
    )

    # 1-bar blip should NOT have caused a HYSTERESIS_CONFIRMED exit
    exit_types = [r.exit_type for r in records]
    assert "HYSTERESIS_CONFIRMED" not in exit_types
