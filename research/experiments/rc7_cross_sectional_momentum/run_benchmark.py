import asyncio
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.config.loader import ConfigLoader, _strip_meta
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.models import Symbol
from neon_radar.domain.trading.execution import CostModel
from neon_radar.infrastructure.exchanges.binance import BinanceClient

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
TX_FEE = 0.0004  # 0.04% per leg (as requested for control rerun)


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


async def get_price_data():
    config = ConfigLoader(Path("config.json")).load()
    raw = json.loads(Path("scoring_rules.json").read_text(encoding="utf-8"))
    scoring_config = ScoringRulesConfig.model_validate(_strip_meta(raw))
    from neon_radar.config.scoring_loader import load_rules

    rules = tuple(load_rules(Path("scoring_rules.json")))
    binance_client = BinanceClient(config.api)
    symbols = [Symbol(s) for s in UNIVERSE]

    start_date = date(2023, 1, 1)
    end_date = date(2026, 8, 1)

    bt_base = TradeBacktester(
        exchange=binance_client, scoring_config=scoring_config, rules=rules, cost_model=CostModel()
    )

    logger.info("Prefetching data for entire period...")
    await bt_base._prefetch(symbols, "4h", start_date, end_date)

    # Build DataFrames for Close and Open prices
    close_data = {}
    open_data = {}
    for sym in symbols:
        series = bt_base._series_cache.get((str(sym), "4h"))
        if series:
            # We use open_time as the index for alignment
            close_data[str(sym)] = {c.open_time: c.close for c in series.candles}
            open_data[str(sym)] = {c.open_time: c.open for c in series.candles}

    df_close = pd.DataFrame(close_data).sort_index().ffill()
    df_open = pd.DataFrame(open_data).sort_index().ffill()

    # Ensure index is datetime UTC
    df_close.index = pd.to_datetime(df_close.index, unit="ms", utc=True)
    df_open.index = pd.to_datetime(df_open.index, unit="ms", utc=True)

    return df_close, df_open


def run_benchmark():
    df_close, df_open = asyncio.run(get_price_data())

    logger.info("Calculating momentum and weights...")
    # Momentum = Return over LOOKBACK_CANDLES
    mom = df_close.pct_change(periods=LOOKBACK_CANDLES)

    # Ranks (descending, 1 is best)
    ranks = mom.rank(axis=1, ascending=False)

    # Top-3 Portfolio Weights
    # If rank <= 3, weight = 1/3, else 0
    w_top3 = (ranks <= 3).astype(float)
    # Normalize weights (in case of ties or missing data causing less than 3)
    w_top3 = w_top3.div(w_top3.sum(axis=1), axis=0).fillna(0)

    # Equal Weight Portfolio Weights
    valid_assets = df_close.notna()
    w_ew = valid_assets.astype(float)
    w_ew = w_ew.div(w_ew.sum(axis=1), axis=0).fillna(0)

    # Execution occurs at T+1 open, held until T+2 open.
    # Therefore, the return of the asset from the signal at T (which is T close)
    # is the return from T+1 open to T+2 open.
    # asset_fwd_returns at row i should be open[i+2] / open[i+1] - 1
    asset_fwd_returns = df_open.shift(-2) / df_open.shift(-1) - 1

    # The weight decided at T is executed at T+1 open.
    # So the return earned between T+1 open and T+2 open uses the weight computed at T.

    # Calculate gross portfolio returns
    port_ret_top3_gross = (w_top3 * asset_fwd_returns).sum(axis=1)
    port_ret_ew_gross = (w_ew * asset_fwd_returns).sum(axis=1)

    # Calculate transaction costs
    # Weight change from T-1 to T.
    turnover_top3 = w_top3.diff().abs().sum(axis=1)
    turnover_ew = w_ew.diff().abs().sum(axis=1)

    # The turnover determined at T costs us at T+1 open.
    # We subtract it from the return of the period starting at T+1 open.
    cost_top3 = turnover_top3 * TX_FEE
    cost_ew = turnover_ew * TX_FEE

    port_ret_top3_net = port_ret_top3_gross - cost_top3
    port_ret_ew_net = port_ret_ew_gross - cost_ew

    # Combine into a results dataframe
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

    # To store OOS daily returns for total stats
    oos_daily_top3 = pd.Series(dtype=float)
    oos_daily_ew = pd.Series(dtype=float)

    while True:
        is_end = current_start + relativedelta(months=6)
        oos_end = is_end + relativedelta(months=2)
        if oos_end > end_date:
            break

        # OOS window masks
        # Since indices are UTC datetimes, we make timezone aware comparisons
        is_end_dt = pd.to_datetime(is_end).tz_localize("UTC")
        oos_end_dt = pd.to_datetime(oos_end).tz_localize("UTC")

        mask = (results.index >= is_end_dt) & (results.index < oos_end_dt)
        window_res = results.loc[mask]

        if len(window_res) > 0:
            # Resample to daily returns
            # port_ret is already a 4h return.
            # Daily return = prod(1 + r) - 1
            daily_top3 = (1 + window_res["net_top3"]).resample("D").prod() - 1
            daily_ew = (1 + window_res["net_ew"]).resample("D").prod() - 1

            oos_daily_top3 = pd.concat([oos_daily_top3, daily_top3])
            oos_daily_ew = pd.concat([oos_daily_ew, daily_ew])

            # Metrics for window
            # Annualized Sharpe (sqrt(365) since daily)
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

    # Overall OOS metrics
    # Drop duplicates if any overlap (shouldn't be)
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

    overall_top3_pf = calc_pf(oos_daily_top3)
    overall_ew_pf = calc_pf(oos_daily_ew)

    overall_top3_dd = calc_max_dd(oos_daily_top3)
    overall_ew_dd = calc_max_dd(oos_daily_ew)

    overall_top3_cum = (1 + oos_daily_top3).prod() - 1
    overall_ew_cum = (1 + oos_daily_ew).prod() - 1

    win_windows = sum(1 for w in windows if w["top3_sharpe"] > w["ew_sharpe"])
    win_rate = win_windows / len(windows) if windows else 0

    res = {
        "p_value_paired_diff": float(p_val),
        "top3_overall_sharpe": float(overall_top3_sharpe),
        "ew_overall_sharpe": float(overall_ew_sharpe),
        "top3_overall_pf": float(overall_top3_pf),
        "ew_overall_pf": float(overall_ew_pf),
        "top3_max_dd": float(overall_top3_dd),
        "ew_max_dd": float(overall_ew_dd),
        "top3_cum_ret": float(overall_top3_cum),
        "ew_cum_ret": float(overall_ew_cum),
        "win_windows_pct": float(win_rate),
        "median_window_top3_sharpe": float(np.median([w["top3_sharpe"] for w in windows])),
        "median_window_ew_sharpe": float(np.median([w["ew_sharpe"] for w in windows])),
        "mean_window_top3_sharpe": float(np.mean([w["top3_sharpe"] for w in windows])),
        "mean_window_ew_sharpe": float(np.mean([w["ew_sharpe"] for w in windows])),
        "windows": windows,
    }

    with open(
        "research/experiments/rc7_cross_sectional_momentum/benchmark_results_004.json", "w"
    ) as f:
        json.dump(res, f, indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    run_benchmark()
