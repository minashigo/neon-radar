"""Binance Futures Grid Deterministic Simulator.

Accurately implements Binance USDS-M Futures Grid mechanics:
- Neutral, Long, and Short grid types
- Arithmetic and Geometric grid price spacing
- Level-by-level limit order execution (Maker fee)
- Dynamic inventory tracking and volume-weighted average entry price
- Realized grid profit vs unrealized mark-to-market PnL
- Tiered maintenance margin and liquidation checks
- 8-hour funding rates and taker fees on market close/stop
- Intrabar price path traversal (Open -> Extremum 1 -> Extremum 2 -> Close)
- Regime Gate termination ("Cancel all orders and close position at market")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GridType(StrEnum):
    NEUTRAL = "NEUTRAL"
    LONG = "LONG"
    SHORT = "SHORT"


class GridMode(StrEnum):
    ARITHMETIC = "ARITHMETIC"
    GEOMETRIC = "GEOMETRIC"


class GridStatus(StrEnum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    CLOSED_GATE = "CLOSED_GATE"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_TP = "CLOSED_TP"
    LIQUIDATED = "LIQUIDATED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class GridConfig:
    lower_price: float
    upper_price: float
    num_grids: int = 20  # Number of intervals
    initial_margin: float = 1000.0  # USDT
    leverage: float = 2.0
    grid_type: GridType = GridType.NEUTRAL
    grid_mode: GridMode = GridMode.ARITHMETIC
    maker_fee_rate: float = 0.0002  # 0.02% maker fee
    taker_fee_rate: float = 0.0005  # 0.05% taker fee
    slippage_pct: float = 0.001  # 0.1% slippage on market exit
    maintenance_margin_rate: float = 0.005  # 0.5% Tier 1 MMR
    liquidation_fee_rate: float = 0.005  # 0.5% clearance fee
    stop_loss_roi: float | None = 0.30  # -30% ROI stop loss
    take_profit_roi: float | None = None  # Optional TP


@dataclass
class GridOrderLevel:
    index: int
    price: float
    has_buy_order: bool = False
    has_sell_order: bool = False
    qty: float = 0.0


@dataclass
class GridEvent:
    timestamp: int
    event_type: str  # INIT, BUY_FILL, SELL_FILL, GRID_PROFIT, FUNDING, SL, TP, LIQUIDATION, GATE_CLOSE
    price: float
    qty: float
    fee: float
    realized_pnl: float
    wallet_balance: float
    inventory: float
    avg_entry: float
    notes: str = ""


@dataclass
class GridSessionResult:
    config: GridConfig
    status: GridStatus
    start_time: int
    end_time: int
    initial_margin: float
    final_wallet_balance: float
    realized_grid_profit: float
    total_fees_paid: float
    total_funding_paid: float
    net_pnl: float
    roi: float
    completed_grids_count: int
    total_fills_count: int
    peak_inventory_abs: float
    max_notional_exposure: float
    max_drawdown_pct: float
    events: list[GridEvent] = field(default_factory=list)

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def is_liquidation(self) -> bool:
        return self.status == GridStatus.LIQUIDATED


class GridSession:
    """Simulates the lifecycle of a Binance Futures Grid Trading session."""

    def __init__(self, config: GridConfig, start_time: int, initial_price: float) -> None:
        self.config = config
        self.status = GridStatus.ACTIVE
        self.start_time = start_time
        self.end_time = start_time
        self.last_price = initial_price

        # Build grid price levels
        self.levels: list[GridOrderLevel] = self._generate_levels()

        # Account balances
        self.wallet_balance = config.initial_margin
        self.peak_equity = self.wallet_balance
        self.max_drawdown_pct = 0.0

        # Inventory tracking
        self.inventory_qty = 0.0  # Positive = Long, Negative = Short
        self.avg_entry_price = 0.0
        self.peak_inventory_abs = 0.0
        self.max_notional_exposure = 0.0

        # Stats
        self.realized_grid_profit = 0.0
        self.total_fees = 0.0
        self.total_funding = 0.0
        self.completed_grids_count = 0
        self.total_fills_count = 0

        self.events: list[GridEvent] = []

        # Order sizing: Notional per grid
        # Total max notional = Initial Margin * Leverage
        self.notional_per_grid = (config.initial_margin * config.leverage) / config.num_grids

        # Setup initial orders
        self._initialize_orders(start_time, initial_price)

    def _generate_levels(self) -> list[GridOrderLevel]:
        levels = []
        n = self.config.num_grids
        p_low = self.config.lower_price
        p_high = self.config.upper_price

        if self.config.grid_mode == GridMode.ARITHMETIC:
            delta = (p_high - p_low) / n
            for i in range(n + 1):
                p = p_low + i * delta
                levels.append(GridOrderLevel(index=i, price=p))
        else:  # GEOMETRIC
            ratio = (p_high - p_low) ** (1.0 / n) if p_low <= 0 else (p_high / p_low) ** (1.0 / n)
            for i in range(n + 1):
                p = p_low * (ratio ** i)
                levels.append(GridOrderLevel(index=i, price=p))
        return levels

    def _initialize_orders(self, timestamp: int, current_price: float) -> None:
        """Sets up initial Buy Limit below current price and Sell Limit above."""
        self.events.append(
            GridEvent(
                timestamp=timestamp,
                event_type="INIT",
                price=current_price,
                qty=0.0,
                fee=0.0,
                realized_pnl=0.0,
                wallet_balance=self.wallet_balance,
                inventory=0.0,
                avg_entry=0.0,
                notes=f"Initialized {self.config.grid_type.value} grid [{self.config.lower_price:.2f} - {self.config.upper_price:.2f}]",
            )
        )

        for lvl in self.levels:
            qty = self.notional_per_grid / lvl.price
            lvl.qty = qty
            if self.config.grid_type == GridType.NEUTRAL:
                if lvl.price < current_price:
                    lvl.has_buy_order = True
                    lvl.has_sell_order = False
                elif lvl.price > current_price:
                    lvl.has_sell_order = True
                    lvl.has_buy_order = False
            elif self.config.grid_type == GridType.LONG:
                # Long grid: holds initial base long, buys below
                if lvl.price < current_price:
                    lvl.has_buy_order = True
                else:
                    lvl.has_sell_order = True
            elif self.config.grid_type == GridType.SHORT:
                if lvl.price > current_price:
                    lvl.has_sell_order = True
                else:
                    lvl.has_buy_order = True

    def unrealized_pnl(self, current_price: float) -> float:
        if self.inventory_qty == 0.0 or self.avg_entry_price == 0.0:
            return 0.0
        return self.inventory_qty * (current_price - self.avg_entry_price)

    def margin_balance(self, current_price: float) -> float:
        return self.wallet_balance + self.unrealized_pnl(current_price)

    def maintenance_margin(self, current_price: float) -> float:
        notional = abs(self.inventory_qty) * current_price
        return notional * self.config.maintenance_margin_rate

    def is_liquidatable(self, current_price: float) -> bool:
        if self.inventory_qty == 0.0:
            return False
        return self.margin_balance(current_price) <= self.maintenance_margin(current_price)

    def roi(self, current_price: float) -> float:
        return (self.margin_balance(current_price) - self.config.initial_margin) / self.config.initial_margin

    def process_candle(
        self,
        timestamp: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        funding_rate: float | None = None,
    ) -> bool:
        """Processes a candle with sequential intrabar path traversal."""
        if self.status != GridStatus.ACTIVE:
            return False

        self.end_time = timestamp

        # 1. 8-Hour Funding application at candle open
        if funding_rate is not None and funding_rate != 0.0 and self.inventory_qty != 0.0:
            notional = abs(self.inventory_qty) * open_price
            funding_cost = (
                notional * funding_rate
                if self.inventory_qty > 0
                else -notional * funding_rate
            )
            self.wallet_balance -= funding_cost
            self.total_funding += funding_cost
            self.events.append(
                GridEvent(
                    timestamp=timestamp,
                    event_type="FUNDING",
                    price=open_price,
                    qty=abs(self.inventory_qty),
                    fee=funding_cost,
                    realized_pnl=0.0,
                    wallet_balance=self.wallet_balance,
                    inventory=self.inventory_qty,
                    avg_entry=self.avg_entry_price,
                    notes=f"Funding fee: {funding_cost:.4f}",
                )
            )
            if self.is_liquidatable(open_price):
                self._terminate(GridStatus.LIQUIDATED, timestamp, open_price, "Funding liquidation")
                return False

        # 2. Determine realistic intrabar traversal path
        # If green bar: Open -> Low -> High -> Close
        # If red bar:   Open -> High -> Low -> Close
        if close_price >= open_price:
            path_legs = [
                (self.last_price, low_price),
                (low_price, high_price),
                (high_price, close_price),
            ]
        else:
            path_legs = [
                (self.last_price, high_price),
                (high_price, low_price),
                (low_price, close_price),
            ]

        for p_from, p_to in path_legs:
            if not self._traverse_leg(timestamp, p_from, p_to):
                return False

        self.last_price = close_price

        # Update drawdown tracking
        current_equity = self.margin_balance(close_price)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - current_equity) / self.peak_equity
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd

        return True

    def _traverse_leg(self, timestamp: int, p_from: float, p_to: float) -> bool:
        """Executes grid order fills along a price vector [p_from -> p_to]."""
        if p_to < p_from:
            # Downward move: triggers Buy Limit orders
            levels_to_check = [lvl for lvl in self.levels if p_to <= lvl.price <= p_from]
            # Sorted descending (crossing higher buys first)
            levels_to_check.sort(key=lambda x: x.price, reverse=True)

            for lvl in levels_to_check:
                if lvl.has_buy_order:
                    self._fill_buy(timestamp, lvl)

            # Check liquidation & Stop loss at the trough
            if self.is_liquidatable(p_to):
                self._terminate(GridStatus.LIQUIDATED, timestamp, p_to, "Adverse downward move liquidation")
                return False
            if self.config.stop_loss_roi is not None and self.roi(p_to) <= -abs(self.config.stop_loss_roi):
                self._terminate(GridStatus.CLOSED_SL, timestamp, p_to, f"Stop loss triggered at ROI {self.roi(p_to):.2%}")
                return False

        elif p_to > p_from:
            # Upward move: triggers Sell Limit orders
            levels_to_check = [lvl for lvl in self.levels if p_from <= lvl.price <= p_to]
            # Sorted ascending (crossing lower sells first)
            levels_to_check.sort(key=lambda x: x.price)

            for lvl in levels_to_check:
                if lvl.has_sell_order:
                    self._fill_sell(timestamp, lvl)

            # Check liquidation & Stop loss at the peak
            if self.is_liquidatable(p_to):
                self._terminate(GridStatus.LIQUIDATED, timestamp, p_to, "Adverse upward move liquidation")
                return False
            if self.config.stop_loss_roi is not None and self.roi(p_to) <= -abs(self.config.stop_loss_roi):
                self._terminate(GridStatus.CLOSED_SL, timestamp, p_to, f"Stop loss triggered at ROI {self.roi(p_to):.2%}")
                return False

        # Optional Take Profit check
        if self.config.take_profit_roi is not None and self.roi(p_to) >= self.config.take_profit_roi:
            self._terminate(GridStatus.CLOSED_TP, timestamp, p_to, f"Take profit reached at ROI {self.roi(p_to):.2%}")
            return False

        return True

    def _fill_buy(self, timestamp: int, lvl: GridOrderLevel) -> None:
        """Executes a Buy order fill at level."""
        qty = lvl.qty
        notional = qty * lvl.price
        fee = notional * self.config.maker_fee_rate

        self.wallet_balance -= fee
        self.total_fees += fee
        self.total_fills_count += 1

        # Inventory accounting
        new_qty = self.inventory_qty + qty
        if self.inventory_qty >= 0:
            # Adding to existing Long
            if new_qty > 0:
                self.avg_entry_price = (
                    self.inventory_qty * self.avg_entry_price + qty * lvl.price
                ) / new_qty
        else:
            # Buying to cover Short
            realized_short_pnl = qty * (self.avg_entry_price - lvl.price)
            self.wallet_balance += realized_short_pnl
            self.realized_grid_profit += realized_short_pnl
            if new_qty >= 0:
                self.avg_entry_price = lvl.price if new_qty > 0 else 0.0

        self.inventory_qty = new_qty
        self._update_exposure_stats(lvl.price)

        # Switch level: buy filled -> place counter-sell order 1 level above
        lvl.has_buy_order = False
        target_idx = lvl.index + 1
        if target_idx < len(self.levels):
            self.levels[target_idx].has_sell_order = True

        self.events.append(
            GridEvent(
                timestamp=timestamp,
                event_type="BUY_FILL",
                price=lvl.price,
                qty=qty,
                fee=fee,
                realized_pnl=0.0,
                wallet_balance=self.wallet_balance,
                inventory=self.inventory_qty,
                avg_entry=self.avg_entry_price,
                notes=f"Buy fill at lvl {lvl.index} ({lvl.price:.2f}), place sell at lvl {target_idx}",
            )
        )

    def _fill_sell(self, timestamp: int, lvl: GridOrderLevel) -> None:
        """Executes a Sell order fill at level."""
        qty = lvl.qty
        notional = qty * lvl.price
        fee = notional * self.config.maker_fee_rate

        self.wallet_balance -= fee
        self.total_fees += fee
        self.total_fills_count += 1

        # Inventory accounting
        new_qty = self.inventory_qty - qty
        if self.inventory_qty > 0:
            # Selling to close Long -> Grid profit!
            realized_long_pnl = qty * (lvl.price - self.avg_entry_price)
            self.wallet_balance += realized_long_pnl
            self.realized_grid_profit += realized_long_pnl
            self.completed_grids_count += 1
            if new_qty <= 0:
                self.avg_entry_price = lvl.price if new_qty < 0 else 0.0
        else:
            # Adding to existing Short
            if new_qty < 0:
                self.avg_entry_price = (
                    abs(self.inventory_qty) * self.avg_entry_price + qty * lvl.price
                ) / abs(new_qty)

        self.inventory_qty = new_qty
        self._update_exposure_stats(lvl.price)

        # Switch level: sell filled -> place counter-buy order 1 level below
        lvl.has_sell_order = False
        target_idx = lvl.index - 1
        if target_idx >= 0:
            self.levels[target_idx].has_buy_order = True

        self.events.append(
            GridEvent(
                timestamp=timestamp,
                event_type="SELL_FILL",
                price=lvl.price,
                qty=qty,
                fee=fee,
                realized_pnl=0.0,
                wallet_balance=self.wallet_balance,
                inventory=self.inventory_qty,
                avg_entry=self.avg_entry_price,
                notes=f"Sell fill at lvl {lvl.index} ({lvl.price:.2f}), place buy at lvl {target_idx}",
            )
        )

    def _update_exposure_stats(self, current_price: float) -> None:
        abs_inv = abs(self.inventory_qty)
        if abs_inv > self.peak_inventory_abs:
            self.peak_inventory_abs = abs_inv
        notional = abs_inv * current_price
        if notional > self.max_notional_exposure:
            self.max_notional_exposure = notional

    def close(self, timestamp: int, exit_price: float, reason: str = "GATE") -> GridSessionResult:
        """Terminates session on Regime Gate trigger or Timeout, closing open inventory at market."""
        if self.status == GridStatus.ACTIVE:
            status = GridStatus.CLOSED_GATE if reason == "GATE" else GridStatus.TIMEOUT
            self._terminate(status, timestamp, exit_price, reason)
        return self.result()

    def _terminate(self, status: GridStatus, timestamp: int, exit_price: float, notes: str) -> None:
        self.status = status
        self.end_time = timestamp

        # Cancel all open grid orders
        for lvl in self.levels:
            lvl.has_buy_order = False
            lvl.has_sell_order = False

        # If holding open inventory, close at market (Taker fee + Slippage)
        if self.inventory_qty != 0.0:
            if self.inventory_qty > 0:
                # Closing Long
                fill_price = exit_price * (1.0 - self.config.slippage_pct)
                pnl = self.inventory_qty * (fill_price - self.avg_entry_price)
            else:
                # Closing Short
                fill_price = exit_price * (1.0 + self.config.slippage_pct)
                pnl = abs(self.inventory_qty) * (self.avg_entry_price - fill_price)

            notional = abs(self.inventory_qty) * fill_price
            exit_fee = notional * self.config.taker_fee_rate
            if status == GridStatus.LIQUIDATED:
                exit_fee += notional * self.config.liquidation_fee_rate

            self.wallet_balance += (pnl - exit_fee)
            self.total_fees += exit_fee

            self.events.append(
                GridEvent(
                    timestamp=timestamp,
                    event_type=status.value,
                    price=fill_price,
                    qty=abs(self.inventory_qty),
                    fee=exit_fee,
                    realized_pnl=pnl,
                    wallet_balance=self.wallet_balance,
                    inventory=0.0,
                    avg_entry=self.avg_entry_price,
                    notes=f"{notes}: Closed inventory {self.inventory_qty:.4f} at {fill_price:.2f}",
                )
            )
            self.inventory_qty = 0.0
            self.avg_entry_price = 0.0
        else:
            self.events.append(
                GridEvent(
                    timestamp=timestamp,
                    event_type=status.value,
                    price=exit_price,
                    qty=0.0,
                    fee=0.0,
                    realized_pnl=0.0,
                    wallet_balance=self.wallet_balance,
                    inventory=0.0,
                    avg_entry=0.0,
                    notes=notes,
                )
            )

    def result(self) -> GridSessionResult:
        net_pnl = self.wallet_balance - self.config.initial_margin
        roi = net_pnl / self.config.initial_margin

        return GridSessionResult(
            config=self.config,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            initial_margin=self.config.initial_margin,
            final_wallet_balance=self.wallet_balance,
            realized_grid_profit=self.realized_grid_profit,
            total_fees_paid=self.total_fees,
            total_funding_paid=self.total_funding,
            net_pnl=net_pnl,
            roi=roi,
            completed_grids_count=self.completed_grids_count,
            total_fills_count=self.total_fills_count,
            peak_inventory_abs=self.peak_inventory_abs,
            max_notional_exposure=self.max_notional_exposure,
            max_drawdown_pct=self.max_drawdown_pct,
            events=self.events,
        )


def simulate_grid_run(
    config: GridConfig,
    candles: list[dict[str, Any]],
    funding_rates: dict[int, float] | None = None,
) -> GridSessionResult:
    """Convenience runner for a sequence of candles."""
    if not candles:
        raise ValueError("Candles list cannot be empty")

    first = candles[0]
    session = GridSession(config, start_time=first["open_time"], initial_price=first["open"])
    funding_rates = funding_rates or {}

    for c in candles:
        t = c["open_time"]
        fr = funding_rates.get(t, None)
        active = session.process_candle(
            timestamp=t,
            open_price=c["open"],
            high_price=c["high"],
            low_price=c["low"],
            close_price=c["close"],
            funding_rate=fr,
        )
        if not active:
            break

    if session.status == GridStatus.ACTIVE:
        last = candles[-1]
        session.close(last["open_time"], last["close"], reason="TIMEOUT")

    return session.result()
