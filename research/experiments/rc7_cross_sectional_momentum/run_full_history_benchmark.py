import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "MATICUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "SOLUSDT",
    "UNIUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "ATOMUSDT",
]

LOOKBACK_CANDLES = 180  # 30 days of 4h candles
TX_FEE = 0.0004  # 0.04% per leg


def block_bootstrap(diff_returns: np.ndarray, block_size: int = 5, n_iterations: int = 10000):
    n = len(diff_returns)
    if n == 0:
        return 1.0
    n_blocks = n // block_size + 1
    means = []
    for _ in range(n_iterations):
        indices = np.random.randint(0, max(1, n - block_size + 1), n_blocks)
        sample = np.concatenate([diff_returns[i : i + block_size] for i in indices])[:n]
        means.append(np.mean(sample))
    means = np.array(means)
    p_value = np.sum(means <= 0) / n_iterations
    return p_value


def load_price_data():
    data_dir = Path("research/experiments/rc7_cross_sectional_momentum/data")

    close_data = {}
    open_data = {}
    for sym in UNIVERSE:
        p = data_dir / f"{sym}.csv"
        df = pd.read_csv(p, parse_dates=["open_time", "close_time"])
        df = df.set_index("open_time")
        close_data[sym] = df["close"]
        open_data[sym] = df["open"]

    df_close = pd.DataFrame(close_data).sort_index().ffill()
    df_open = pd.DataFrame(open_data).sort_index().ffill()

    return df_close, df_open


def run_benchmark():
    df_close, df_open = load_price_data()

    logger.info("Calculating momentum and weights...")
    # Momentum = Return over LOOKBACK_CANDLES
    mom = df_close.pct_change(periods=LOOKBACK_CANDLES)

    # Ranks (descending, 1 is best)
    ranks = mom.rank(axis=1, ascending=False)

    # Top-3 Portfolio Weights
    w_top3 = (ranks <= 3).astype(float)
    # Normalize weights
    w_top3 = w_top3.div(w_top3.sum(axis=1), axis=0).fillna(0)

    # Equal Weight Portfolio Weights
    valid_assets = df_close.notna()
    w_ew = valid_assets.astype(float)
    w_ew = w_ew.div(w_ew.sum(axis=1), axis=0).fillna(0)

    # Execution occurs at T+1 open, held until T+2 open.
    # Therefore, the return of the asset from the signal at T (which is T close)
    # is the return from T+1 open to T+2 open.
    asset_fwd_returns = df_open.shift(-2) / df_open.shift(-1) - 1

    # Calculate gross portfolio returns
    port_ret_top3_gross = (w_top3 * asset_fwd_returns).sum(axis=1)
    port_ret_ew_gross = (w_ew * asset_fwd_returns).sum(axis=1)

    # Calculate transaction costs
    turnover_top3 = w_top3.diff().abs().sum(axis=1)
    turnover_ew = w_ew.diff().abs().sum(axis=1)

    cost_top3 = turnover_top3 * TX_FEE
    cost_ew = turnover_ew * TX_FEE

    port_ret_top3_net = port_ret_top3_gross - cost_top3
    port_ret_ew_net = port_ret_ew_gross - cost_ew

    results = pd.DataFrame(
        {
            "gross_top3": port_ret_top3_gross,
            "gross_ew": port_ret_ew_gross,
            "net_top3": port_ret_top3_net,
            "net_ew": port_ret_ew_net,
        },
        index=df_close.index,
    )

    # Now, evaluate based on WFA OOS windows
    start_date = date(2023, 1, 1)
    end_date = date(2026, 8, 1)

    current_start = start_date
    windows = []

    oos_daily_top3 = pd.Series(dtype=float)
    oos_daily_ew = pd.Series(dtype=float)

    while True:
        is_end = current_start + relativedelta(months=6)
        oos_end = is_end + relativedelta(months=2)
        if oos_end > end_date:
            break

        is_end_dt = pd.to_datetime(is_end).tz_localize("UTC")
        oos_end_dt = pd.to_datetime(oos_end).tz_localize("UTC")

        mask = (results.index >= is_end_dt) & (results.index < oos_end_dt)
        window_res = results.loc[mask]

        if len(window_res) > 0:
            daily_top3 = (1 + window_res["net_top3"]).resample("D").prod() - 1
            daily_ew = (1 + window_res["net_ew"]).resample("D").prod() - 1

            oos_daily_top3 = pd.concat([oos_daily_top3, daily_top3])
            oos_daily_ew = pd.concat([oos_daily_ew, daily_ew])

            mean_t = daily_top3.mean()
            std_t = daily_top3.std()
            sharpe_top3 = (mean_t / std_t * np.sqrt(365)) if std_t != 0 else 0

            mean_e = daily_ew.mean()
            std_e = daily_ew.std()
            sharpe_ew = (mean_e / std_e * np.sqrt(365)) if std_e != 0 else 0

            windows.append(
                {
                    "window": f"{is_end} to {oos_end}",
                    "top3_sharpe": sharpe_top3,
                    "ew_sharpe": sharpe_ew,
                    "top3_cum_ret": (1 + daily_top3).prod() - 1,
                    "ew_cum_ret": (1 + daily_ew).prod() - 1,
                }
            )

        current_start += relativedelta(months=2)

    oos_daily_top3 = oos_daily_top3[~oos_daily_top3.index.duplicated()]
    oos_daily_ew = oos_daily_ew[~oos_daily_ew.index.duplicated()]

    diff_returns = (oos_daily_top3 - oos_daily_ew).values
    p_val = block_bootstrap(diff_returns)

    def calc_pf(daily_rets):
        gains = daily_rets[daily_rets > 0].sum()
        losses = abs(daily_rets[daily_rets < 0].sum())
        return gains / losses if losses != 0 else float("inf")

    def calc_max_dd(daily_rets):
        cum = (1 + daily_rets).cumprod()
        roll_max = cum.cummax()
        drawdown = cum / roll_max - 1
        return drawdown.min()

    overall_top3_sharpe = (
        (oos_daily_top3.mean() / oos_daily_top3.std() * np.sqrt(365))
        if oos_daily_top3.std() != 0
        else 0
    )
    overall_ew_sharpe = (
        (oos_daily_ew.mean() / oos_daily_ew.std() * np.sqrt(365)) if oos_daily_ew.std() != 0 else 0
    )

    res = {
        "p_value_paired_diff": float(p_val),
        "top3_overall_sharpe": float(overall_top3_sharpe),
        "ew_overall_sharpe": float(overall_ew_sharpe),
        "top3_overall_pf": float(calc_pf(oos_daily_top3)),
        "ew_overall_pf": float(calc_pf(oos_daily_ew)),
        "top3_max_dd": float(calc_max_dd(oos_daily_top3)),
        "ew_max_dd": float(calc_max_dd(oos_daily_ew)),
        "top3_cum_ret": float((1 + oos_daily_top3).prod() - 1),
        "ew_cum_ret": float((1 + oos_daily_ew).prod() - 1),
        "win_windows_pct": float(
            sum(1 for w in windows if w["top3_sharpe"] > w["ew_sharpe"]) / len(windows)
            if windows
            else 0
        ),
        "windows": windows,
    }

    with open(
        "research/experiments/rc7_cross_sectional_momentum/full_history_results.json", "w"
    ) as f:
        json.dump(res, f, indent=2)

    # Also save raw returns
    df_returns = pd.DataFrame({"top3_net": oos_daily_top3, "ew_net": oos_daily_ew})
    df_returns.to_csv("research/experiments/rc7_cross_sectional_momentum/full_history_returns.csv")

    logger.info("Done.")


if __name__ == "__main__":
    run_benchmark()
