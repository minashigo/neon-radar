"""Quick smoke test for isolated and portfolio backtests."""

from research.experiments.futures_grid.backtest_exit_variants import (
    ExitVariant,
    run_isolated_backtest,
    run_multi_asset_portfolio_backtest,
)


def test_isolated_backtest_smoke():
    """Verify single-asset continuous backtest runs without errors."""
    res_a = run_isolated_backtest(
        symbol="BTCUSDT",
        timeframe="4h",
        initial_capital=10000.0,
        variant=ExitVariant.IMMEDIATE_CLOSE,
    )
    assert res_a.initial_capital == 10000.0
    assert res_a.return_pct >= -1.0
    assert 0.0 <= res_a.max_drawdown_pct <= 1.0


def test_shared_portfolio_backtest_smoke():
    """Verify shared multi-asset portfolio backtest runs without errors."""
    acct, _records, summary = run_multi_asset_portfolio_backtest(
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="4h",
        initial_capital=10000.0,
        margin_per_asset=1500.0,
        variant=ExitVariant.FREEZE_ATR,
    )
    assert summary.initial_capital == 10000.0
    assert summary.return_pct >= -1.0
    assert len(acct.history) > 100
