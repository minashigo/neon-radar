"""Baseline A: Continuous Binance Futures Grid Backtest Engine.

Runs neutral futures grid sessions continuously over the historical period:
- When idle, initiates a new grid centered around current price.
- Grid runs until SL, TP, liquidation, or max session timeout.
- Once terminated, immediately re-deploys a new grid on the next bar.
- Accounts for maker/taker fees, slippage, and 8-hour funding rates.
"""

from __future__ import annotations

from typing import Any

import pandas as pd  # noqa: TC002
from research.experiments.futures_grid.data_loader import (
    build_funding_rate_map,
    df_to_candle_dicts,
    load_candles_df,
    resample_to_1d,
)
from research.experiments.futures_grid.metrics import calculate_grid_metrics
from research.experiments.futures_grid.simulator import (
    GridConfig,
    GridMode,
    GridSession,
    GridSessionResult,
    GridStatus,
    GridType,
)


def run_baseline_grid_backtest(
    df: pd.DataFrame,
    range_pct: float = 0.10,  # +/- 10% from entry price
    num_grids: int = 20,
    leverage: float = 2.0,
    initial_margin: float = 1000.0,
    grid_mode: GridMode = GridMode.ARITHMETIC,
    stop_loss_roi: float | None = 0.30,
    take_profit_roi: float | None = None,
    max_session_bars: int = 180,  # ~30 days on 4h
    warmup_bars: int = 60,
    include_funding: bool = True,
) -> list[GridSessionResult]:
    """Runs continuous Baseline A grid backtest over candle history."""
    candles = df_to_candle_dicts(df)
    funding_map = build_funding_rate_map(df) if include_funding else {}

    sessions: list[GridSessionResult] = []
    active_session: GridSession | None = None
    session_bars_count = 0

    for i in range(warmup_bars, len(candles) - 1):
        current_candle = candles[i]
        next_candle = candles[i + 1]

        # 1. Step active session through current candle
        if active_session is not None:
            t = current_candle["open_time"]
            fr = funding_map.get(t, None)
            is_active = active_session.process_candle(
                timestamp=t,
                open_price=current_candle["open"],
                high_price=current_candle["high"],
                low_price=current_candle["low"],
                close_price=current_candle["close"],
                funding_rate=fr,
            )
            session_bars_count += 1

            if not is_active:
                sessions.append(active_session.result())
                active_session = None
                session_bars_count = 0
            elif session_bars_count >= max_session_bars:
                # Re-balancing timeout: close inventory and reset grid at current price
                res = active_session.close(
                    timestamp=next_candle["open_time"],
                    exit_price=next_candle["open"],
                    reason="TIMEOUT",
                )
                sessions.append(res)
                active_session = None
                session_bars_count = 0

        # 2. If idle, deploy new grid at next candle open
        if active_session is None:
            p_ref = next_candle["open"]
            p_low = p_ref * (1.0 - range_pct)
            p_high = p_ref * (1.0 + range_pct)

            cfg = GridConfig(
                lower_price=p_low,
                upper_price=p_high,
                num_grids=num_grids,
                initial_margin=initial_margin,
                leverage=leverage,
                grid_type=GridType.NEUTRAL,
                grid_mode=grid_mode,
                stop_loss_roi=stop_loss_roi,
                take_profit_roi=take_profit_roi,
            )
            active_session = GridSession(
                config=cfg,
                start_time=next_candle["open_time"],
                initial_price=p_ref,
            )
            session_bars_count = 0

    # Close any open session at end of history
    if active_session is not None and active_session.status == GridStatus.ACTIVE:
        last_c = candles[-1]
        res = active_session.close(last_c["open_time"], last_c["close"], reason="TIMEOUT")
        sessions.append(res)

    return sessions


def run_baseline_grid_suite(
    symbols: list[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"),
    timeframes: list[str] = ("4h", "1d"),
    range_pcts: list[float] = (0.05, 0.10, 0.15),
    num_grids_list: list[int] = (10, 20),
    leverages: list[float] = (2.0, 3.0),
) -> dict[str, Any]:
    """Runs a multi-parameter grid suite for Baseline A."""
    results = {}

    for tf in timeframes:
        for sym in symbols:
            df_raw = load_candles_df(sym)
            df = resample_to_1d(df_raw) if tf == "1d" else df_raw

            for r_pct in range_pcts:
                for n_grids in num_grids_list:
                    for lev in leverages:
                        key = f"{sym}_{tf}_rng{int(r_pct*100)}%_n{n_grids}_lev{int(lev)}x"
                        sessions = run_baseline_grid_backtest(
                            df=df,
                            range_pct=r_pct,
                            num_grids=n_grids,
                            leverage=lev,
                        )
                        metrics = calculate_grid_metrics(sessions)
                        results[key] = {
                            "symbol": sym,
                            "timeframe": tf,
                            "range_pct": r_pct,
                            "num_grids": n_grids,
                            "leverage": lev,
                            "metrics": metrics.to_dict(),
                        }

    return results
