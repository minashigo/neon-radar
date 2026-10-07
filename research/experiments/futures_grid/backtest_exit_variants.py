"""Continuous Backtester for Futures Grid Exit Management Variants.

Runs backtests across 6 assets (BTC, ETH, SOL, BNB, XRP, ADA) across timeframes:
1. Isolated Continuous Account per asset (10,000 USDT each)
2. Synchronized Shared Multi-Asset Portfolio (10,000 USDT total shared across 6 assets)
Compares:
- Variant A: Immediate Close on non-CHOP
- Variant B: Freeze (hold inventory, normal SL)
- Variant C: Freeze + ATR Trailing Stop (2.5x ATR)
- Variant D: Hysteresis (N=2 confirmation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from research.experiments.futures_grid.churn_analysis import (
    ChurnMetrics,
    analyze_churn,
)
from research.experiments.futures_grid.data_loader import (
    load_candles_df,
    resample_to_1d,
)
from research.experiments.futures_grid.exit_management_engine import (
    ExitEngineConfig,
    GridInstance,
    RegimeExitRecord,
    SessionState,
    _annotate_post_exit_prices,
    _execute_buy_fill,
    _execute_sell_fill,
    compute_atr_series,
    run_continuous_grid_simulation,
)
from research.experiments.futures_grid.portfolio import (
    ExitVariant,
    PortfolioAccount,
)
from research.experiments.futures_grid.regime_engine import precompute_regimes


@dataclass(frozen=True)
class ContinuousBacktestResult:
    symbol: str
    timeframe: str
    variant: ExitVariant
    initial_capital: float
    final_wallet: float
    final_equity: float
    net_pnl: float
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    total_maker_fees: float
    total_taker_fees: float
    total_churn_cost: float
    total_realized_grid_profit: float
    is_bankrupt: bool
    liquidations_count: int
    churn_metrics: ChurnMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "variant": self.variant.value,
            "initial_capital": self.initial_capital,
            "final_wallet": round(self.final_wallet, 2),
            "final_equity": round(self.final_equity, 2),
            "net_pnl": round(self.net_pnl, 2),
            "return_pct": round(self.return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "total_maker_fees": round(self.total_maker_fees, 2),
            "total_taker_fees": round(self.total_taker_fees, 2),
            "total_churn_cost": round(self.total_churn_cost, 2),
            "total_realized_grid_profit": round(self.total_realized_grid_profit, 2),
            "is_bankrupt": self.is_bankrupt,
            "liquidations_count": self.liquidations_count,
            "churn_metrics": self.churn_metrics.to_dict(),
        }


def _calculate_sharpe_and_sortino(account: PortfolioAccount) -> tuple[float, float]:
    """Calculates annualized Sharpe and Sortino ratios from daily resampled equity."""
    if not account.history or len(account.history) < 10:
        return 0.0, 0.0

    df_hist = pd.DataFrame([
        {"timestamp": s.timestamp, "equity": s.equity}
        for s in account.history
    ])
    df_hist["dt"] = pd.to_datetime(df_hist["timestamp"], unit="ms", utc=True)
    df_daily = df_hist.set_index("dt").resample("1D").last().dropna()
    returns = df_daily["equity"].pct_change().dropna()

    if len(returns) < 5 or returns.std() == 0:
        return 0.0, 0.0

    ann_factor = math.sqrt(365.0)
    mean_ret = float(returns.mean())
    std_ret = float(returns.std())
    sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 0 else 0.0

    neg_rets = returns[returns < 0]
    downside_std = float(neg_rets.std()) if len(neg_rets) > 1 else std_ret
    sortino = (mean_ret / downside_std) * ann_factor if downside_std > 0 else 0.0

    return float(np.clip(sharpe, -10.0, 10.0)), float(np.clip(sortino, -10.0, 10.0))


def run_isolated_backtest(
    symbol: str,
    timeframe: str = "4h",
    initial_capital: float = 10000.0,
    variant: ExitVariant = ExitVariant.IMMEDIATE_CLOSE,
    config: ExitEngineConfig | None = None,
) -> ContinuousBacktestResult:
    """Runs a single-asset continuous backtest for a specific ExitVariant."""
    config = config or ExitEngineConfig()
    df_raw = load_candles_df(symbol)
    df = resample_to_1d(df_raw) if timeframe == "1d" else df_raw
    regimes_df = precompute_regimes(symbol, timeframe, df)

    account = PortfolioAccount(initial_capital=initial_capital)
    acct, records = run_continuous_grid_simulation(
        df=df,
        regimes_df=regimes_df,
        symbol=symbol,
        account=account,
        config=config,
        variant=variant,
    )

    last_p = float(df.iloc[-1]["close"])
    final_eq = acct.equity({symbol: last_p})
    ret_pct = acct.return_pct({symbol: last_p})
    sharpe, sortino = _calculate_sharpe_and_sortino(acct)
    churn = analyze_churn(records, variant.value)

    return ContinuousBacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        variant=variant,
        initial_capital=initial_capital,
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


def run_multi_asset_portfolio_backtest(
    symbols: list[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"),
    timeframe: str = "4h",
    initial_capital: float = 10000.0,
    margin_per_asset: float = 1500.0,
    variant: ExitVariant = ExitVariant.IMMEDIATE_CLOSE,
    config: ExitEngineConfig | None = None,
) -> tuple[PortfolioAccount, list[RegimeExitRecord], ContinuousBacktestResult]:
    """Runs a synchronized continuous multi-asset portfolio backtest across all symbols."""
    base_cfg = config or ExitEngineConfig(allocated_margin=margin_per_asset)

    # 1. Load data and align timestamps
    dfs: dict[str, pd.DataFrame] = {}
    regimes_dfs: dict[str, pd.DataFrame] = {}
    atr_series_dict: dict[str, pd.Series] = {}

    for sym in symbols:
        raw = load_candles_df(sym)
        df_sym = resample_to_1d(raw) if timeframe == "1d" else raw
        dfs[sym] = df_sym
        regimes_dfs[sym] = precompute_regimes(sym, timeframe, df_sym)
        atr_series_dict[sym] = compute_atr_series(df_sym, period=base_cfg.atr_period)

    # Determine common timestamp indices
    common_times = set(dfs[symbols[0]]["open_time_ms"])
    for sym in symbols[1:]:
        common_times = common_times.intersection(set(dfs[sym]["open_time_ms"]))
    common_times_list = sorted(common_times)

    # Build aligned dictionaries
    aligned_candles: dict[str, dict[int, dict[str, Any]]] = {}
    aligned_regimes: dict[str, dict[int, dict[str, Any]]] = {}
    for sym in symbols:
        df_sym = dfs[sym]
        reg_sym = regimes_dfs[sym]
        c_dict = {row["open_time_ms"]: row for row in df_sym.to_dict("records")}
        r_dict = {row["open_time_ms"]: row for row in reg_sym.to_dict("records")}
        aligned_candles[sym] = c_dict
        aligned_regimes[sym] = r_dict

    account = PortfolioAccount(initial_capital=initial_capital)
    all_records: list[RegimeExitRecord] = []
    active_grids: dict[str, GridInstance | None] = dict.fromkeys(symbols)

    n_bars = len(common_times_list)
    warmup_bars = 60

    for i in range(warmup_bars, n_bars - 1):
        if account.is_bankrupt:
            break

        t_curr = common_times_list[i]
        t_next = common_times_list[i + 1]

        prices_next_open = {sym: float(aligned_candles[sym][t_next]["open"]) for sym in symbols}

        # Step 1: Evaluate state transitions for each symbol
        for sym in symbols:
            c_next = aligned_candles[sym][t_next]
            reg_curr = aligned_regimes[sym][t_curr]
            is_grid_allowed = bool(reg_curr["grid_allowed"])
            next_open = prices_next_open[sym]
            next_open_time = t_next

            inv = account.get_inventory(sym)
            grid = active_grids[sym]

            if grid is not None:
                grid.bars_in_session += 1

                if not is_grid_allowed:
                    grid.non_chop_streak += 1

                    if variant == ExitVariant.IMMEDIATE_CLOSE:
                        grid.cancel_pending_orders()
                        if inv.qty != 0.0:
                            qty = inv.qty
                            avg_e = inv.avg_entry
                            pnl = account.market_close_inventory(sym, price=next_open, reason="CHURN")
                            fee = abs(qty) * next_open * account.taker_fee
                            slip = abs(qty) * next_open * account.slippage
                            all_records.append(
                                RegimeExitRecord(
                                    timestamp=next_open_time,
                                    symbol=sym,
                                    variant=variant,
                                    exit_type="IMMEDIATE",
                                    position_qty=qty,
                                    entry_price=avg_e,
                                    exit_price=next_open,
                                    net_pnl=pnl,
                                    taker_fee=fee,
                                    slippage_cost=slip,
                                    churn_cost=fee + slip,
                                )
                            )
                        account.release_margin(sym)
                        active_grids[sym] = None

                    elif variant == ExitVariant.FREEZE:
                        if grid.state == SessionState.ACTIVE:
                            grid.cancel_pending_orders()
                            grid.state = SessionState.FROZEN

                    elif variant == ExitVariant.FREEZE_ATR:
                        if grid.state == SessionState.ACTIVE:
                            grid.cancel_pending_orders()
                            grid.state = SessionState.FROZEN_ATR
                            grid.high_watermark = next_open
                            grid.low_watermark = next_open
                            atr_val = atr_series_dict[sym].iloc[i]
                            if math.isnan(atr_val):
                                atr_val = next_open * 0.02
                            if inv.qty > 0:
                                grid.trailing_stop_price = next_open - base_cfg.atr_multiplier * atr_val
                            elif inv.qty < 0:
                                grid.trailing_stop_price = next_open + base_cfg.atr_multiplier * atr_val

                    elif variant == ExitVariant.HYSTERESIS:
                        if grid.non_chop_streak >= base_cfg.hysteresis_bars:
                            grid.cancel_pending_orders()
                            if inv.qty != 0.0:
                                qty = inv.qty
                                avg_e = inv.avg_entry
                                pnl = account.market_close_inventory(sym, price=next_open, reason="CHURN")
                                fee = abs(qty) * next_open * account.taker_fee
                                slip = abs(qty) * next_open * account.slippage
                                all_records.append(
                                    RegimeExitRecord(
                                        timestamp=next_open_time,
                                        symbol=sym,
                                        variant=variant,
                                        exit_type="HYSTERESIS_CONFIRMED",
                                        position_qty=qty,
                                        entry_price=avg_e,
                                        exit_price=next_open,
                                        net_pnl=pnl,
                                        taker_fee=fee,
                                        slippage_cost=slip,
                                        churn_cost=fee + slip,
                                    )
                                )
                            account.release_margin(sym)
                            active_grids[sym] = None

                else:
                    # Regime is CHOP
                    grid.non_chop_streak = 0
                    if grid.state in (SessionState.FROZEN, SessionState.FROZEN_ATR):
                        if inv.qty != 0.0:
                            qty = inv.qty
                            avg_e = inv.avg_entry
                            pnl = account.market_close_inventory(sym, price=next_open, reason="RECENTER")
                            fee = abs(qty) * next_open * account.taker_fee
                            slip = abs(qty) * next_open * account.slippage
                            all_records.append(
                                RegimeExitRecord(
                                    timestamp=next_open_time,
                                    symbol=sym,
                                    variant=variant,
                                    exit_type="RECENTER_CHOP",
                                    position_qty=qty,
                                    entry_price=avg_e,
                                    exit_price=next_open,
                                    net_pnl=pnl,
                                    taker_fee=fee,
                                    slippage_cost=slip,
                                    churn_cost=fee + slip,
                                )
                            )
                        account.release_margin(sym)
                        active_grids[sym] = None

                # Check timeout
                if active_grids[sym] is not None and grid.bars_in_session >= base_cfg.max_session_bars:
                    grid.cancel_pending_orders()
                    if inv.qty != 0.0:
                        account.market_close_inventory(sym, price=next_open, reason="TIMEOUT")
                    account.release_margin(sym)
                    active_grids[sym] = None

            # Deploy fresh grid if IDLE and CHOP allowed
            if active_grids[sym] is None and is_grid_allowed:
                alloc = account.allocate_margin(sym, base_cfg.allocated_margin, prices_next_open)
                if alloc >= 100.0:
                    active_grids[sym] = GridInstance(
                        symbol=sym,
                        config=base_cfg,
                        variant=variant,
                        start_time=next_open_time,
                        initial_price=next_open,
                        actual_margin=alloc,
                    )

        # Step 2: Process Intrabar price moves for each symbol on Candle i+1
        prices_next_close = {}
        for sym in symbols:
            c_next = aligned_candles[sym][t_next]
            p_open = float(c_next["open"])
            p_high = float(c_next["high"])
            p_low = float(c_next["low"])
            p_close = float(c_next["close"])
            prices_next_close[sym] = p_close

            grid = active_grids[sym]
            inv = account.get_inventory(sym)

            if grid is not None:
                legs = (
                    [(p_open, p_low), (p_low, p_high), (p_high, p_close)]
                    if p_close >= p_open
                    else [(p_open, p_high), (p_high, p_low), (p_low, p_close)]
                )

                # Check ATR trailing stop breach
                if grid.state == SessionState.FROZEN_ATR and grid.trailing_stop_price is not None:
                    atr_val = atr_series_dict[sym].iloc[i]
                    if math.isnan(atr_val):
                        atr_val = p_open * 0.02

                    if inv.qty > 0:
                        if p_high > grid.high_watermark:
                            grid.high_watermark = p_high
                            grid.trailing_stop_price = max(
                                grid.trailing_stop_price,
                                p_high - base_cfg.atr_multiplier * atr_val,
                            )
                        if p_low <= grid.trailing_stop_price:
                            qty = inv.qty
                            avg_e = inv.avg_entry
                            exit_p = grid.trailing_stop_price
                            pnl = account.market_close_inventory(sym, price=exit_p, reason="SL")
                            fee = abs(qty) * exit_p * account.taker_fee
                            slip = abs(qty) * exit_p * account.slippage
                            all_records.append(
                                RegimeExitRecord(
                                    timestamp=t_next,
                                    symbol=sym,
                                    variant=variant,
                                    exit_type="ATR_TRAILING",
                                    position_qty=qty,
                                    entry_price=avg_e,
                                    exit_price=exit_p,
                                    net_pnl=pnl,
                                    taker_fee=fee,
                                    slippage_cost=slip,
                                    churn_cost=fee + slip,
                                )
                            )
                            account.release_margin(sym)
                            active_grids[sym] = None

                    elif inv.qty < 0:
                        if p_low < grid.low_watermark:
                            grid.low_watermark = p_low
                            grid.trailing_stop_price = min(
                                grid.trailing_stop_price,
                                p_low + base_cfg.atr_multiplier * atr_val,
                            )
                        if p_high >= grid.trailing_stop_price:
                            qty = inv.qty
                            avg_e = inv.avg_entry
                            exit_p = grid.trailing_stop_price
                            pnl = account.market_close_inventory(sym, price=exit_p, reason="SL")
                            fee = abs(qty) * exit_p * account.taker_fee
                            slip = abs(qty) * exit_p * account.slippage
                            all_records.append(
                                RegimeExitRecord(
                                    timestamp=t_next,
                                    symbol=sym,
                                    variant=variant,
                                    exit_type="ATR_TRAILING",
                                    position_qty=qty,
                                    entry_price=avg_e,
                                    exit_price=exit_p,
                                    net_pnl=pnl,
                                    taker_fee=fee,
                                    slippage_cost=slip,
                                    churn_cost=fee + slip,
                                )
                            )
                            account.release_margin(sym)
                            active_grids[sym] = None

                # Normal grid order fills
                if active_grids[sym] is not None and grid.state == SessionState.ACTIVE:
                    for p_from, p_to in legs:
                        if p_to < p_from:
                            buys = [lvl for lvl in grid.levels if lvl.has_buy_order and p_to <= lvl.price <= p_from]
                            buys.sort(key=lambda x: x.price, reverse=True)
                            for lvl in buys:
                                _execute_buy_fill(account, sym, grid, lvl)
                        else:
                            sells = [lvl for lvl in grid.levels if lvl.has_sell_order and p_from <= lvl.price <= p_to]
                            sells.sort(key=lambda x: x.price)
                            for lvl in sells:
                                _execute_sell_fill(account, sym, grid, lvl)

                # Stop loss check
                if active_grids[sym] is not None and inv.qty != 0.0:
                    upnl = inv.unrealized_pnl(p_close)
                    max_loss = grid.actual_margin * base_cfg.stop_loss_roi
                    if upnl <= -max_loss:
                        qty = inv.qty
                        avg_e = inv.avg_entry
                        pnl = account.market_close_inventory(sym, price=p_close, reason="SL")
                        fee = abs(qty) * p_close * account.taker_fee
                        slip = abs(qty) * p_close * account.slippage
                        all_records.append(
                            RegimeExitRecord(
                                timestamp=t_next,
                                symbol=sym,
                                variant=variant,
                                exit_type="SL_HIT",
                                position_qty=qty,
                                entry_price=avg_e,
                                exit_price=p_close,
                                net_pnl=pnl,
                                taker_fee=fee,
                                slippage_cost=slip,
                                churn_cost=fee + slip,
                            )
                        )
                        account.release_margin(sym)
                        active_grids[sym] = None

        # Step continuous shared portfolio accounting
        account.step(timestamp=t_next, prices=prices_next_close)

    # Final wrap-up
    last_t = common_times_list[-1]
    final_prices = {sym: float(aligned_candles[sym][last_t]["close"]) for sym in symbols}
    for sym in symbols:
        inv = account.get_inventory(sym)
        if inv.qty != 0.0:
            account.market_close_inventory(sym, price=final_prices[sym], reason="TIMEOUT")
        account.release_margin(sym)

    account.step(timestamp=last_t, prices=final_prices)

    final_eq = account.equity(final_prices)
    ret_pct = account.return_pct(final_prices)
    sharpe, sortino = _calculate_sharpe_and_sortino(account)
    # Annotate post-exit prices per symbol
    for sym in symbols:
        sym_records = [r for r in all_records if r.symbol == sym]
        sym_candles = dfs[sym].to_dict("records")
        _annotate_post_exit_prices(sym_candles, sym_records)

    churn = analyze_churn(all_records, variant.value)

    summary = ContinuousBacktestResult(
        symbol="PORTFOLIO_6ASSETS",
        timeframe=timeframe,
        variant=variant,
        initial_capital=initial_capital,
        final_wallet=account.wallet_balance,
        final_equity=final_eq,
        net_pnl=account.net_pnl(final_prices),
        return_pct=ret_pct,
        max_drawdown_pct=account.max_drawdown_pct,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        total_maker_fees=account.total_maker_fees,
        total_taker_fees=account.total_taker_fees,
        total_churn_cost=account.total_churn_cost,
        total_realized_grid_profit=account.total_realized_grid_profit,
        is_bankrupt=account.is_bankrupt,
        liquidations_count=account.liquidation_events_count,
        churn_metrics=churn,
    )

    return account, all_records, summary
