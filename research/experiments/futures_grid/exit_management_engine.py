"""Exit Management Engine for Binance Futures Grid + Neon Radar Filter.

Evaluates strategies to mitigate regime churn (CHOP -> non-CHOP -> CHOP):
- Variant A (IMMEDIATE_CLOSE): Cancel orders and close inventory at market on non-CHOP.
- Variant B (FREEZE): Cancel entry orders, preserve inventory under protective SL, pause until CHOP.
- Variant C (FREEZE_ATR): Cancel entry orders, preserve inventory, activate ATR trailing stop.
- Variant D (HYSTERESIS): Wait for N=2 consecutive non-CHOP bars before closing.

Operates with strict continuous portfolio accounting via PortfolioAccount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd
from research.experiments.futures_grid.portfolio import (
    ExitVariant,
    PortfolioAccount,
)


class SessionState(StrEnum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"
    FROZEN_ATR = "FROZEN_ATR"
    CLOSED = "CLOSED"


@dataclass
class OrderLevel:
    index: int
    price: float
    has_buy_order: bool = False
    has_sell_order: bool = False
    qty: float = 0.0


@dataclass
class RegimeExitRecord:
    timestamp: int
    symbol: str
    variant: ExitVariant
    exit_type: str  # "IMMEDIATE", "ATR_TRAILING", "SL_HIT", "RECENTER_CHOP", "TIMEOUT"
    position_qty: float
    entry_price: float
    exit_price: float
    net_pnl: float
    taker_fee: float
    slippage_cost: float
    churn_cost: float
    # Post-exit price tracking (set in analysis)
    post_6bar_price: float = 0.0
    post_12bar_price: float = 0.0
    post_24bar_price: float = 0.0
    is_protective: bool = False
    is_spurious: bool = False


@dataclass
class ExitEngineConfig:
    range_pct: float = 0.10
    num_grids: int = 20
    leverage: float = 2.0
    allocated_margin: float = 1000.0  # Target margin per grid
    stop_loss_roi: float = 0.30       # -30% ROI stop loss
    atr_period: int = 14
    atr_multiplier: float = 2.5       # For Variant C
    hysteresis_bars: int = 2          # For Variant D
    max_session_bars: int = 180       # Re-centering timeout
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0005
    slippage_pct: float = 0.001


def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Computes lookahead-free rolling ATR."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


class GridInstance:
    """Manages an active grid instance for a single symbol under a continuous account."""

    def __init__(
        self,
        symbol: str,
        config: ExitEngineConfig,
        variant: ExitVariant,
        start_time: int,
        initial_price: float,
        actual_margin: float,
    ) -> None:
        self.symbol = symbol
        self.config = config
        self.variant = variant
        self.start_time = start_time
        self.initial_price = initial_price
        self.actual_margin = actual_margin
        self.last_price = initial_price

        self.state = SessionState.ACTIVE
        self.bars_in_session = 0
        self.non_chop_streak = 0  # For Variant D hysteresis

        # ATR Trailing stop trackers (for Variant C)
        self.trailing_stop_price: float | None = None
        self.high_watermark: float = initial_price
        self.low_watermark: float = initial_price

        # Grid levels
        self.levels: list[OrderLevel] = self._build_levels()
        self.notional_per_grid = (actual_margin * config.leverage) / config.num_grids
        self._setup_initial_orders(initial_price)

        self.completed_grids = 0
        self.total_fills = 0

    def _build_levels(self) -> list[OrderLevel]:
        levels = []
        n = self.config.num_grids
        p_low = self.initial_price * (1.0 - self.config.range_pct)
        p_high = self.initial_price * (1.0 + self.config.range_pct)
        delta = (p_high - p_low) / n
        for i in range(n + 1):
            p = p_low + i * delta
            levels.append(OrderLevel(index=i, price=p))
        return levels

    def _setup_initial_orders(self, price: float) -> None:
        for lvl in self.levels:
            qty = self.notional_per_grid / lvl.price
            lvl.qty = qty
            if lvl.price < price:
                lvl.has_buy_order = True
                lvl.has_sell_order = False
            elif lvl.price > price:
                lvl.has_sell_order = True
                lvl.has_buy_order = False

    def cancel_pending_orders(self) -> None:
        for lvl in self.levels:
            lvl.has_buy_order = False
            lvl.has_sell_order = False


def run_continuous_grid_simulation(
    df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    symbol: str,
    account: PortfolioAccount,
    config: ExitEngineConfig,
    variant: ExitVariant,
    warmup_bars: int = 60,
) -> tuple[PortfolioAccount, list[RegimeExitRecord]]:
    """Runs a continuous single-asset grid simulation over candle history.

    Operates strictly lookahead-free:
    - At candle i close: regime and ATR are computed.
    - If action required (enter or exit), executed at candle i+1 open.
    - Candle i+1 intrabar path processed.
    """
    atr_series = compute_atr_series(df, period=config.atr_period)
    records: list[RegimeExitRecord] = []
    active_grid: GridInstance | None = None

    candles = df.to_dict("records")
    regimes = regimes_df.to_dict("records")
    n_candles = len(candles)

    for i in range(warmup_bars, n_candles - 1):
        if account.is_bankrupt:
            break

        c_curr = candles[i]
        c_next = candles[i + 1]
        reg_curr = regimes[i]
        is_grid_allowed = bool(reg_curr["grid_allowed"])
        atr_curr = float(atr_series.iloc[i]) if not math.isnan(atr_series.iloc[i]) else (c_curr["close"] * 0.02)

        inv = account.get_inventory(symbol)

        # ----------------------------------------------------
        # 1. Evaluate State Transitions at Bar i Close -> Act at Bar i+1 Open
        # ----------------------------------------------------
        next_open = float(c_next["open"])
        next_open_time = int(c_next["open_time_ms"])

        if active_grid is not None:
            active_grid.bars_in_session += 1

            # Check Regime Transition
            if not is_grid_allowed:
                active_grid.non_chop_streak += 1

                if variant == ExitVariant.IMMEDIATE_CLOSE:
                    # Variant A: Immediate market close on non-CHOP
                    active_grid.cancel_pending_orders()
                    if inv.qty != 0.0:
                        qty = inv.qty
                        avg_e = inv.avg_entry
                        pnl = account.market_close_inventory(symbol, price=next_open, reason="CHURN")
                        fee = abs(qty) * next_open * account.taker_fee
                        slip = abs(qty) * next_open * account.slippage
                        records.append(
                            RegimeExitRecord(
                                timestamp=next_open_time,
                                symbol=symbol,
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
                    account.release_margin(symbol)
                    active_grid = None

                elif variant == ExitVariant.FREEZE:
                    # Variant B: Freeze grid, keep inventory open
                    if active_grid.state == SessionState.ACTIVE:
                        active_grid.cancel_pending_orders()
                        active_grid.state = SessionState.FROZEN

                elif variant == ExitVariant.FREEZE_ATR:
                    # Variant C: Freeze grid, activate ATR trailing stop
                    if active_grid.state == SessionState.ACTIVE:
                        active_grid.cancel_pending_orders()
                        active_grid.state = SessionState.FROZEN_ATR
                        active_grid.high_watermark = next_open
                        active_grid.low_watermark = next_open
                        if inv.qty > 0:
                            active_grid.trailing_stop_price = next_open - config.atr_multiplier * atr_curr
                        elif inv.qty < 0:
                            active_grid.trailing_stop_price = next_open + config.atr_multiplier * atr_curr

                elif variant == ExitVariant.HYSTERESIS:
                    # Variant D: Hysteresis confirmation
                    if active_grid.non_chop_streak >= config.hysteresis_bars:
                        active_grid.cancel_pending_orders()
                        if inv.qty != 0.0:
                            qty = inv.qty
                            avg_e = inv.avg_entry
                            pnl = account.market_close_inventory(symbol, price=next_open, reason="CHURN")
                            fee = abs(qty) * next_open * account.taker_fee
                            slip = abs(qty) * next_open * account.slippage
                            records.append(
                                RegimeExitRecord(
                                    timestamp=next_open_time,
                                    symbol=symbol,
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
                        account.release_margin(symbol)
                        active_grid = None

            else:
                # Regime is CHOP
                active_grid.non_chop_streak = 0
                if active_grid.state in (SessionState.FROZEN, SessionState.FROZEN_ATR):
                    # CHOP returned after a freeze!
                    # Re-center fresh grid: close stale inventory at market and deploy fresh grid
                    if inv.qty != 0.0:
                        qty = inv.qty
                        avg_e = inv.avg_entry
                        pnl = account.market_close_inventory(symbol, price=next_open, reason="RECENTER")
                        fee = abs(qty) * next_open * account.taker_fee
                        slip = abs(qty) * next_open * account.slippage
                        records.append(
                            RegimeExitRecord(
                                timestamp=next_open_time,
                                symbol=symbol,
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
                    account.release_margin(symbol)
                    active_grid = None

            # Check timeout
            if active_grid is not None and active_grid.bars_in_session >= config.max_session_bars:
                active_grid.cancel_pending_orders()
                if inv.qty != 0.0:
                    account.market_close_inventory(symbol, price=next_open, reason="TIMEOUT")
                account.release_margin(symbol)
                active_grid = None

        # ----------------------------------------------------
        # 2. Deploy fresh grid if IDLE and CHOP allowed
        # ----------------------------------------------------
        if active_grid is None and is_grid_allowed:
            allocated = account.allocate_margin(symbol, config.allocated_margin, {symbol: next_open})
            if allocated >= 100.0:  # Minimum viable margin
                active_grid = GridInstance(
                    symbol=symbol,
                    config=config,
                    variant=variant,
                    start_time=next_open_time,
                    initial_price=next_open,
                    actual_margin=allocated,
                )

        # ----------------------------------------------------
        # 3. Process Intrabar Price Action on Candle i+1
        # ----------------------------------------------------
        p_open = float(c_next["open"])
        p_high = float(c_next["high"])
        p_low = float(c_next["low"])
        p_close = float(c_next["close"])

        if active_grid is not None:
            # Traversal order: green vs red bar
            legs = (
                [(p_open, p_low), (p_low, p_high), (p_high, p_close)]
                if p_close >= p_open
                else [(p_open, p_high), (p_high, p_low), (p_low, p_close)]
            )

            # Check ATR trailing stop during bar traversal for Variant C
            if active_grid.state == SessionState.FROZEN_ATR and active_grid.trailing_stop_price is not None:
                # Update watermarks
                if inv.qty > 0:
                    if p_high > active_grid.high_watermark:
                        active_grid.high_watermark = p_high
                        active_grid.trailing_stop_price = max(
                            active_grid.trailing_stop_price,
                            p_high - config.atr_multiplier * atr_curr,
                        )
                    # Check stop breach
                    if p_low <= active_grid.trailing_stop_price:
                        # ATR Trailing stop triggered!
                        qty = inv.qty
                        avg_e = inv.avg_entry
                        exit_p = active_grid.trailing_stop_price
                        pnl = account.market_close_inventory(symbol, price=exit_p, reason="SL")
                        fee = abs(qty) * exit_p * account.taker_fee
                        slip = abs(qty) * exit_p * account.slippage
                        records.append(
                            RegimeExitRecord(
                                timestamp=next_open_time,
                                symbol=symbol,
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
                        account.release_margin(symbol)
                        active_grid = None

                elif inv.qty < 0:
                    if p_low < active_grid.low_watermark:
                        active_grid.low_watermark = p_low
                        active_grid.trailing_stop_price = min(
                            active_grid.trailing_stop_price,
                            p_low + config.atr_multiplier * atr_curr,
                        )
                    # Check stop breach
                    if p_high >= active_grid.trailing_stop_price:
                        qty = inv.qty
                        avg_e = inv.avg_entry
                        exit_p = active_grid.trailing_stop_price
                        pnl = account.market_close_inventory(symbol, price=exit_p, reason="SL")
                        fee = abs(qty) * exit_p * account.taker_fee
                        slip = abs(qty) * exit_p * account.slippage
                        records.append(
                            RegimeExitRecord(
                                timestamp=next_open_time,
                                symbol=symbol,
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
                        account.release_margin(symbol)
                        active_grid = None

            # Normal grid order fills (only when ACTIVE)
            if active_grid is not None and active_grid.state == SessionState.ACTIVE:
                for p_from, p_to in legs:
                    if p_to < p_from:
                        # Downward move -> Buy fills
                        buys = [lvl for lvl in active_grid.levels if lvl.has_buy_order and p_to <= lvl.price <= p_from]
                        buys.sort(key=lambda x: x.price, reverse=True)
                        for lvl in buys:
                            _execute_buy_fill(account, symbol, active_grid, lvl)
                    else:
                        # Upward move -> Sell fills
                        sells = [lvl for lvl in active_grid.levels if lvl.has_sell_order and p_from <= lvl.price <= p_to]
                        sells.sort(key=lambda x: x.price)
                        for lvl in sells:
                            _execute_sell_fill(account, symbol, active_grid, lvl)

            # Check protective Stop Loss (ROI on allocated margin)
            if active_grid is not None and inv.qty != 0.0:
                upnl = inv.unrealized_pnl(p_close)
                max_loss = active_grid.actual_margin * config.stop_loss_roi
                if upnl <= -max_loss:
                    qty = inv.qty
                    avg_e = inv.avg_entry
                    pnl = account.market_close_inventory(symbol, price=p_close, reason="SL")
                    fee = abs(qty) * p_close * account.taker_fee
                    slip = abs(qty) * p_close * account.slippage
                    records.append(
                        RegimeExitRecord(
                            timestamp=next_open_time,
                            symbol=symbol,
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
                    account.release_margin(symbol)
                    active_grid = None

        # Step continuous portfolio accounting at bar close
        account.step(timestamp=int(c_next["open_time_ms"]), prices={symbol: p_close})

    # Close any open positions at end
    if not account.is_bankrupt:
        last_c = candles[-1]
        last_p = float(last_c["close"])
        inv = account.get_inventory(symbol)
        if inv.qty != 0.0:
            account.market_close_inventory(symbol, price=last_p, reason="TIMEOUT")
        account.release_margin(symbol)
        account.step(timestamp=int(last_c["open_time_ms"]), prices={symbol: last_p})

    # Annotate post-exit excursion prices for churn analysis
    _annotate_post_exit_prices(candles, records)

    return account, records


def _execute_buy_fill(
    account: PortfolioAccount,
    symbol: str,
    grid: GridInstance,
    lvl: OrderLevel,
) -> None:
    inv = account.get_inventory(symbol)
    qty = lvl.qty
    notional = qty * lvl.price
    fee = notional * account.maker_fee

    account.wallet_balance -= fee
    account.total_maker_fees += fee
    grid.total_fills += 1

    new_qty = inv.qty + qty
    if inv.qty >= 0:
        if new_qty > 0:
            inv.avg_entry = (inv.qty * inv.avg_entry + qty * lvl.price) / new_qty
    else:
        # Closing short
        realized = qty * (inv.avg_entry - lvl.price)
        account.wallet_balance += realized
        account.total_realized_grid_profit += realized
        grid.completed_grids += 1
        if new_qty >= 0:
            inv.avg_entry = lvl.price if new_qty > 0 else 0.0

    inv.qty = new_qty
    lvl.has_buy_order = False
    target = lvl.index + 1
    if target < len(grid.levels):
        grid.levels[target].has_sell_order = True


def _execute_sell_fill(
    account: PortfolioAccount,
    symbol: str,
    grid: GridInstance,
    lvl: OrderLevel,
) -> None:
    inv = account.get_inventory(symbol)
    qty = lvl.qty
    notional = qty * lvl.price
    fee = notional * account.maker_fee

    account.wallet_balance -= fee
    account.total_maker_fees += fee
    grid.total_fills += 1

    new_qty = inv.qty - qty
    if inv.qty > 0:
        # Closing long
        realized = qty * (lvl.price - inv.avg_entry)
        account.wallet_balance += realized
        account.total_realized_grid_profit += realized
        grid.completed_grids += 1
        if new_qty <= 0:
            inv.avg_entry = lvl.price if new_qty < 0 else 0.0
    else:
        if new_qty < 0:
            inv.avg_entry = (abs(inv.qty) * inv.avg_entry + qty * lvl.price) / abs(new_qty)

    inv.qty = new_qty
    lvl.has_sell_order = False
    target = lvl.index - 1
    if target >= 0:
        grid.levels[target].has_buy_order = True


def _annotate_post_exit_prices(candles: list[dict[str, Any]], records: list[RegimeExitRecord]) -> None:
    """Finds price +6, +12, +24 bars after exit to classify protective vs spurious."""
    time_to_idx = {c["open_time_ms"]: idx for idx, c in enumerate(candles)}
    n = len(candles)

    for rec in records:
        idx = time_to_idx.get(rec.timestamp)
        if idx is not None:
            if idx + 6 < n:
                rec.post_6bar_price = float(candles[idx + 6]["close"])
            if idx + 12 < n:
                rec.post_12bar_price = float(candles[idx + 12]["close"])
            if idx + 24 < n:
                rec.post_24bar_price = float(candles[idx + 24]["close"])

            # Classification:
            # If position was Long:
            # - Price dropped further (post_12 < exit_price * 0.98): Protective exit!
            # - Price recovered back to or above entry (post_12 >= entry_price): Spurious exit!
            # If position was Short:
            # - Price rose further (post_12 > exit_price * 1.02): Protective exit!
            # - Price dropped back to or below entry (post_12 <= entry_price): Spurious exit!
            ref_post = rec.post_12bar_price if rec.post_12bar_price > 0 else rec.exit_price
            if rec.position_qty > 0:
                if ref_post < rec.exit_price * 0.98:
                    rec.is_protective = True
                elif ref_post >= rec.entry_price:
                    rec.is_spurious = True
            elif rec.position_qty < 0:
                if ref_post > rec.exit_price * 1.02:
                    rec.is_protective = True
                elif ref_post <= rec.entry_price:
                    rec.is_spurious = True
