"""Walk-Forward Analysis (WFA) for Futures Grid Exit Management Variants.

Protocol:
- Rolling windows: 180-day In-Sample (Train) / 60-day Out-Of-Sample (Test) / 60-day step
- Evaluates 18 rolling OOS cycles across 2023-2026 data
- Uses strict continuous portfolio accounting per window
- Compares Variant A (Immediate Close), Variant B (Freeze), Variant C (Freeze + ATR), Variant D (Hysteresis)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from research.experiments.futures_grid.backtest_exit_variants import (
    ContinuousBacktestResult,
    _calculate_sharpe_and_sortino,
)
from research.experiments.futures_grid.churn_analysis import analyze_churn
from research.experiments.futures_grid.data_loader import (
    load_candles_df,
    resample_to_1d,
)
from research.experiments.futures_grid.exit_management_engine import (
    ExitEngineConfig,
    run_continuous_grid_simulation,
)
from research.experiments.futures_grid.portfolio import (
    ExitVariant,
    PortfolioAccount,
)
from research.experiments.futures_grid.regime_engine import precompute_regimes


@dataclass
class WfaCycleResult:
    cycle_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    variant_results: dict[str, ContinuousBacktestResult]


@dataclass
class WfaExitReport:
    symbol: str
    timeframe: str
    total_cycles: int
    cycles: list[WfaCycleResult]
    aggregate_oos_pnl: dict[str, float]
    aggregate_oos_return: dict[str, float]
    aggregate_oos_max_dd: dict[str, float]
    aggregate_oos_sharpe: dict[str, float]
    aggregate_oos_churn_cost: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "total_cycles": self.total_cycles,
            "aggregate_oos_pnl": {k: round(v, 2) for k, v in self.aggregate_oos_pnl.items()},
            "aggregate_oos_return": {k: round(v, 4) for k, v in self.aggregate_oos_return.items()},
            "aggregate_oos_max_dd": {k: round(v, 4) for k, v in self.aggregate_oos_max_dd.items()},
            "aggregate_oos_sharpe": {k: round(v, 4) for k, v in self.aggregate_oos_sharpe.items()},
            "aggregate_oos_churn_cost": {k: round(v, 2) for k, v in self.aggregate_oos_churn_cost.items()},
            "cycle_summaries": [
                {
                    "cycle": c.cycle_index,
                    "pnl": {k: round(v.net_pnl, 2) for k, v in c.variant_results.items()},
                    "max_dd": {k: round(v.max_drawdown_pct, 4) for k, v in c.variant_results.items()},
                }
                for c in self.cycles
            ],
        }


def run_wfa_exit_variants(
    symbol: str = "BTCUSDT",
    timeframe: str = "4h",
    is_window_days: int = 180,
    oos_window_days: int = 60,
    step_days: int = 60,
    variants: tuple[ExitVariant, ...] = (
        ExitVariant.IMMEDIATE_CLOSE,
        ExitVariant.FREEZE,
        ExitVariant.FREEZE_ATR,
        ExitVariant.HYSTERESIS,
    ),
    config: ExitEngineConfig | None = None,
) -> WfaExitReport:
    """Executes 18-cycle rolling Walk-Forward Analysis comparing exit variants."""
    config = config or ExitEngineConfig()
    df_raw = load_candles_df(symbol)
    df = resample_to_1d(df_raw) if timeframe == "1d" else df_raw
    regimes_df = precompute_regimes(symbol, timeframe, df)

    t_min = int(df["open_time_ms"].min())
    t_max = int(df["open_time_ms"].max())
    ms_day = 86400 * 1000

    is_len_ms = is_window_days * ms_day
    oos_len_ms = oos_window_days * ms_day
    step_ms = step_days * ms_day

    cycles: list[WfaCycleResult] = []
    cycle_idx = 0
    t_start = t_min

    while t_start + is_len_ms + oos_len_ms <= t_max:
        train_start = t_start
        train_end = t_start + is_len_ms
        test_start = train_end
        test_end = test_start + oos_len_ms

        # Slice OOS data
        mask_oos = (df["open_time_ms"] >= test_start) & (df["open_time_ms"] < test_end)
        df_oos = df[mask_oos].reset_index(drop=True)
        reg_oos = regimes_df[mask_oos].reset_index(drop=True)

        if len(df_oos) > 30:
            cycle_variants: dict[str, ContinuousBacktestResult] = {}
            for var in variants:
                acct = PortfolioAccount(initial_capital=10000.0)
                acct, records = run_continuous_grid_simulation(
                    df=df_oos,
                    regimes_df=reg_oos,
                    symbol=symbol,
                    account=acct,
                    config=config,
                    variant=var,
                    warmup_bars=5,
                )
                last_p = float(df_oos.iloc[-1]["close"])
                final_eq = acct.equity({symbol: last_p})
                ret_pct = acct.return_pct({symbol: last_p})
                sharpe, sortino = _calculate_sharpe_and_sortino(acct)
                churn = analyze_churn(records, var.value)

                res = ContinuousBacktestResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    variant=var,
                    initial_capital=10000.0,
                    final_wallet=acct.wallet_balance,
                    final_equity=final_eq,
                    net_pnl=acct.net_pnl({symbol: last_p}),
                    return_pct=ret_pct,
                    max_drawdown_pct=acct.max_drawdown_pct,
                    sharpe_ratio=sharpe,
                    sortino_ratio=sortino,
                    total_maker_fees=acct.total_maker_fees,
                    total_taker_fees=acct.total_taker_fees,
                    total_churn_cost=acct.total_churn_cost,
                    total_realized_grid_profit=acct.total_realized_grid_profit,
                    is_bankrupt=acct.is_bankrupt,
                    liquidations_count=acct.liquidation_events_count,
                    churn_metrics=churn,
                )
                cycle_variants[var.value] = res

            cycles.append(
                WfaCycleResult(
                    cycle_index=cycle_idx,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    variant_results=cycle_variants,
                )
            )
            cycle_idx += 1

        t_start += step_ms

    # Compute aggregates across all OOS cycles
    agg_pnl: dict[str, float] = {}
    agg_ret: dict[str, float] = {}
    agg_dd: dict[str, float] = {}
    agg_sharpe: dict[str, float] = {}
    agg_churn: dict[str, float] = {}

    for var in variants:
        v_name = var.value
        pnls = [c.variant_results[v_name].net_pnl for c in cycles]
        dds = [c.variant_results[v_name].max_drawdown_pct for c in cycles]
        sharpes = [c.variant_results[v_name].sharpe_ratio for c in cycles]
        churns = [c.variant_results[v_name].total_churn_cost for c in cycles]

        agg_pnl[v_name] = sum(pnls)
        agg_ret[v_name] = (sum(pnls) / 10000.0)
        agg_dd[v_name] = max(dds) if dds else 0.0
        agg_sharpe[v_name] = float(np.mean(sharpes)) if sharpes else 0.0
        agg_churn[v_name] = sum(churns)

    return WfaExitReport(
        symbol=symbol,
        timeframe=timeframe,
        total_cycles=len(cycles),
        cycles=cycles,
        aggregate_oos_pnl=agg_pnl,
        aggregate_oos_return=agg_ret,
        aggregate_oos_max_dd=agg_dd,
        aggregate_oos_sharpe=agg_sharpe,
        aggregate_oos_churn_cost=agg_churn,
    )
