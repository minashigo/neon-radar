"""Walk-Forward Analysis (WFA) comparing Baseline A vs Experiment B for Futures Grid.

Protocol:
- Rolling windows: 180-day In-Sample (Train) / 60-day Out-Of-Sample (Test) / 60-day step.
- Grid hyperparameters (range_pct, num_grids, leverage) optimized strictly on Train.
- Frozen optimal parameters evaluated on the unseen OOS window.
- Compares pure continuous Grid vs Radar Regime-Gated Grid out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd  # noqa: TC002
from research.experiments.futures_grid.backtest_baseline import run_baseline_grid_backtest
from research.experiments.futures_grid.backtest_radar_grid import run_radar_grid_backtest
from research.experiments.futures_grid.metrics import GridMetricsReport, calculate_grid_metrics
from research.experiments.futures_grid.simulator import GridSessionResult  # noqa: TC002


@dataclass
class WfaGridCycleResult:
    cycle_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    baseline_best_params: dict[str, Any]
    radar_best_params: dict[str, Any]
    baseline_oos_metrics: GridMetricsReport
    radar_oos_metrics: GridMetricsReport
    baseline_oos_sessions: list[GridSessionResult]
    radar_oos_sessions: list[GridSessionResult]


@dataclass
class WfaGridReport:
    cycles: list[WfaGridCycleResult]
    baseline_aggregate_oos: GridMetricsReport
    radar_aggregate_oos: GridMetricsReport
    baseline_all_sessions: list[GridSessionResult]
    radar_all_sessions: list[GridSessionResult]


def run_grid_wfa(
    df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    is_window_days: int = 180,
    oos_window_days: int = 60,
    step_days: int = 60,
    range_candidates: tuple[float, ...] = (0.05, 0.10, 0.15),
    num_grids_candidates: tuple[int, ...] = (10, 20),
    leverage_candidates: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> WfaGridReport:
    """Performs full Walk-Forward Analysis on a given asset."""
    t_min = int(df["open_time_ms"].min())
    t_max = int(df["open_time_ms"].max())

    ms_per_day = 86400 * 1000
    is_ms = is_window_days * ms_per_day
    oos_ms = oos_window_days * ms_per_day
    step_ms = step_days * ms_per_day

    cycles: list[WfaGridCycleResult] = []
    current_start = t_min
    cycle_idx = 0

    base_global_sessions: list[GridSessionResult] = []
    radar_global_sessions: list[GridSessionResult] = []

    while current_start + is_ms + oos_ms <= t_max:
        train_start = current_start
        train_end = current_start + is_ms
        test_start = train_end
        test_end = train_end + oos_ms

        df_train = df[(df["open_time_ms"] >= train_start) & (df["open_time_ms"] < train_end)].reset_index(drop=True)
        sig_train = regimes_df[(regimes_df["open_time_ms"] >= train_start) & (regimes_df["open_time_ms"] < train_end)].reset_index(drop=True)

        df_test = df[(df["open_time_ms"] >= test_start) & (df["open_time_ms"] < test_end)].reset_index(drop=True)
        sig_test = regimes_df[(regimes_df["open_time_ms"] >= test_start) & (regimes_df["open_time_ms"] < test_end)].reset_index(drop=True)

        if len(df_train) < 60 or len(df_test) < 20:
            current_start += step_ms
            continue

        # 1. Optimize on Train for Baseline A
        best_base_params = None
        best_base_score = -999999.0

        for r_pct in range_candidates:
            for n_g in num_grids_candidates:
                for lev in leverage_candidates:
                    sess = run_baseline_grid_backtest(
                        df=df_train,
                        range_pct=r_pct,
                        num_grids=n_g,
                        leverage=lev,
                        warmup_bars=10,
                    )
                    m = calculate_grid_metrics(sess)
                    score = m.expectancy if m.total_sessions >= 2 else -500.0
                    if score > best_base_score:
                        best_base_score = score
                        best_base_params = {"range_pct": r_pct, "num_grids": n_g, "leverage": lev}

        if best_base_params is None:
            best_base_params = {"range_pct": 0.10, "num_grids": 10, "leverage": 2.0}

        # 2. Optimize on Train for Experiment B (Radar Grid)
        best_radar_params = None
        best_radar_score = -999999.0

        for r_pct in range_candidates:
            for n_g in num_grids_candidates:
                for lev in leverage_candidates:
                    sess = run_radar_grid_backtest(
                        df=df_train,
                        regimes_df=sig_train,
                        range_pct=r_pct,
                        num_grids=n_g,
                        leverage=lev,
                        warmup_bars=10,
                    )
                    m = calculate_grid_metrics(sess)
                    score = m.expectancy if m.total_sessions >= 1 else -500.0
                    if score > best_radar_score:
                        best_radar_score = score
                        best_radar_params = {"range_pct": r_pct, "num_grids": n_g, "leverage": lev}

        if best_radar_params is None:
            best_radar_params = {"range_pct": 0.10, "num_grids": 10, "leverage": 2.0}

        # 3. Evaluate frozen parameters on OOS
        base_oos_sess = run_baseline_grid_backtest(
            df=df_test,
            range_pct=best_base_params["range_pct"],
            num_grids=best_base_params["num_grids"],
            leverage=best_base_params["leverage"],
            warmup_bars=5,
        )
        base_oos_metrics = calculate_grid_metrics(base_oos_sess)
        base_global_sessions.extend(base_oos_sess)

        radar_oos_sess = run_radar_grid_backtest(
            df=df_test,
            regimes_df=sig_test,
            range_pct=best_radar_params["range_pct"],
            num_grids=best_radar_params["num_grids"],
            leverage=best_radar_params["leverage"],
            warmup_bars=5,
        )
        radar_oos_metrics = calculate_grid_metrics(radar_oos_sess)
        radar_global_sessions.extend(radar_oos_sess)

        cycles.append(
            WfaGridCycleResult(
                cycle_index=cycle_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                baseline_best_params=best_base_params,
                radar_best_params=best_radar_params,
                baseline_oos_metrics=base_oos_metrics,
                radar_oos_metrics=radar_oos_metrics,
                baseline_oos_sessions=base_oos_sess,
                radar_oos_sessions=radar_oos_sess,
            )
        )

        cycle_idx += 1
        current_start += step_ms

    return WfaGridReport(
        cycles=cycles,
        baseline_aggregate_oos=calculate_grid_metrics(base_global_sessions),
        radar_aggregate_oos=calculate_grid_metrics(radar_global_sessions),
        baseline_all_sessions=base_global_sessions,
        radar_all_sessions=radar_global_sessions,
    )
