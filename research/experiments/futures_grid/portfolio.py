"""Continuous Portfolio & Capital Accounting Engine for Futures Grid.

Strictly enforces realistic exchange-level portfolio accounting:
- Single continuous shared capital pool (e.g. 10,000 USDT)
- Wallet balance carries forward continuously across sessions and assets
- No capital reset or magical replenishment after losses
- Available margin gating: new grid allocations are constrained by remaining equity
- True cross/isolated margin and maintenance margin liquidation
- Strict bankruptcy handling: when equity <= maintenance margin, account is liquidated
- Returns are strictly bounded: Return >= -100%
- Continuous equity curve tracking for true path-dependent Max Drawdown
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExitVariant(StrEnum):
    IMMEDIATE_CLOSE = "IMMEDIATE_CLOSE"  # Variant A: close inventory at market on non-CHOP
    FREEZE = "FREEZE"                    # Variant B: cancel entry orders, keep inventory, SL active
    FREEZE_ATR = "FREEZE_ATR"            # Variant C: freeze grid, activate ATR trailing protection
    HYSTERESIS = "HYSTERESIS"            # Variant D: N-bar confirmation before switching


@dataclass
class AssetInventory:
    symbol: str
    qty: float = 0.0  # Positive = Long, Negative = Short
    avg_entry: float = 0.0
    allocated_margin: float = 0.0
    peak_price_since_freeze: float = 0.0
    trough_price_since_freeze: float = 0.0

    def unrealized_pnl(self, current_price: float) -> float:
        if self.qty == 0.0 or self.avg_entry == 0.0:
            return 0.0
        return self.qty * (current_price - self.avg_entry)

    def notional(self, current_price: float) -> float:
        return abs(self.qty) * current_price


@dataclass
class PortfolioSnapshot:
    timestamp: int
    wallet_balance: float
    equity: float
    margin_allocated: float
    available_margin: float
    active_positions_count: int
    total_notional: float
    drawdown_pct: float


class PortfolioAccount:
    """Manages continuous capital accounting across multiple assets and grid sessions."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        maintenance_margin_rate: float = 0.005,
        maker_fee_rate: float = 0.0002,
        taker_fee_rate: float = 0.0005,
        slippage_pct: float = 0.001,
        liquidation_fee_rate: float = 0.005,
    ) -> None:
        self.initial_capital = initial_capital
        self.wallet_balance = initial_capital
        self.peak_equity = initial_capital
        self.max_drawdown_pct = 0.0

        self.mmr = maintenance_margin_rate
        self.maker_fee = maker_fee_rate
        self.taker_fee = taker_fee_rate
        self.slippage = slippage_pct
        self.liq_fee = liquidation_fee_rate

        self.inventories: dict[str, AssetInventory] = {}
        self.is_bankrupt = False
        self.liquidation_events_count = 0

        self.history: list[PortfolioSnapshot] = []

        # Friction & performance accumulators
        self.total_maker_fees = 0.0
        self.total_taker_fees = 0.0
        self.total_funding_paid = 0.0
        self.total_realized_grid_profit = 0.0
        self.total_churn_cost = 0.0

    def get_inventory(self, symbol: str) -> AssetInventory:
        if symbol not in self.inventories:
            self.inventories[symbol] = AssetInventory(symbol=symbol)
        return self.inventories[symbol]

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        upnl = 0.0
        for sym, inv in self.inventories.items():
            if sym in prices and inv.qty != 0.0:
                upnl += inv.unrealized_pnl(prices[sym])
        return upnl

    def equity(self, prices: dict[str, float]) -> float:
        if self.is_bankrupt:
            return 0.0
        eq = self.wallet_balance + self.unrealized_pnl(prices)
        return max(0.0, eq)

    def total_maintenance_margin(self, prices: dict[str, float]) -> float:
        mm = 0.0
        for sym, inv in self.inventories.items():
            if sym in prices and inv.qty != 0.0:
                mm += inv.notional(prices[sym]) * self.mmr
        return mm

    def total_margin_allocated(self) -> float:
        return sum(inv.allocated_margin for inv in self.inventories.values())

    def available_margin(self, prices: dict[str, float]) -> float:
        if self.is_bankrupt:
            return 0.0
        eq = self.equity(prices)
        allocated = self.total_margin_allocated()
        return max(0.0, eq - allocated)

    def allocate_margin(self, symbol: str, requested_margin: float, prices: dict[str, float]) -> float:
        """Allocates margin for a new grid session if available."""
        if self.is_bankrupt:
            return 0.0
        avail = self.available_margin(prices)
        alloc = min(requested_margin, avail)
        if alloc > 0:
            inv = self.get_inventory(symbol)
            inv.allocated_margin += alloc
        return alloc

    def release_margin(self, symbol: str) -> None:
        inv = self.get_inventory(symbol)
        inv.allocated_margin = 0.0

    def check_and_apply_liquidation(self, timestamp: int, prices: dict[str, float]) -> bool:
        """Checks if margin balance <= maintenance margin. If so, triggers liquidation."""
        if self.is_bankrupt:
            return True

        eq = self.wallet_balance + self.unrealized_pnl(prices)
        mm = self.total_maintenance_margin(prices)

        if eq <= mm or eq <= 0.0:
            # LIQUIDATION
            self.is_bankrupt = True
            self.liquidation_events_count += 1

            # Force close all positions
            for sym, inv in list(self.inventories.items()):
                if inv.qty != 0.0 and sym in prices:
                    p = prices[sym]
                    fill_p = p * (1.0 - self.slippage) if inv.qty > 0 else p * (1.0 + self.slippage)
                    pnl = inv.qty * (fill_p - inv.avg_entry) if inv.qty > 0 else abs(inv.qty) * (inv.avg_entry - fill_p)
                    fee = inv.notional(fill_p) * (self.taker_fee + self.liq_fee)
                    self.wallet_balance += (pnl - fee)
                    self.total_taker_fees += fee
                    inv.qty = 0.0
                    inv.avg_entry = 0.0
                    inv.allocated_margin = 0.0

            self.wallet_balance = max(0.0, self.wallet_balance)
            self._record_snapshot(timestamp, prices)
            return True

        return False

    def market_close_inventory(
        self,
        symbol: str,
        price: float,
        reason: str = "GATE",
    ) -> float:
        """Closes accumulated inventory for symbol at market (Taker fee + slippage)."""
        inv = self.get_inventory(symbol)
        if inv.qty == 0.0:
            inv.allocated_margin = 0.0
            return 0.0

        if inv.qty > 0:
            fill_p = price * (1.0 - self.slippage)
            pnl = inv.qty * (fill_p - inv.avg_entry)
        else:
            fill_p = price * (1.0 + self.slippage)
            pnl = abs(inv.qty) * (inv.avg_entry - fill_p)

        fee = inv.notional(fill_p) * self.taker_fee
        net_trade_pnl = pnl - fee

        self.wallet_balance += net_trade_pnl
        self.total_taker_fees += fee

        if reason in ("GATE", "CHURN"):
            self.total_churn_cost += (fee + abs(pnl) if pnl < 0 else fee)

        inv.qty = 0.0
        inv.avg_entry = 0.0
        inv.allocated_margin = 0.0
        inv.peak_price_since_freeze = 0.0
        inv.trough_price_since_freeze = 0.0

        return net_trade_pnl

    def apply_funding(self, symbol: str, price: float, funding_rate: float) -> float:
        inv = self.get_inventory(symbol)
        if inv.qty == 0.0 or funding_rate == 0.0:
            return 0.0
        cost = inv.qty * price * funding_rate
        self.wallet_balance -= cost
        self.total_funding_paid += cost
        return cost

    def step(self, timestamp: int, prices: dict[str, float]) -> None:
        """Updates continuous equity curve and checks liquidation."""
        if self.is_bankrupt:
            self._record_snapshot(timestamp, prices)
            return

        if self.check_and_apply_liquidation(timestamp, prices):
            return

        eq = self.equity(prices)
        if eq > self.peak_equity:
            self.peak_equity = eq
        if self.peak_equity > 0:
            dd = (self.peak_equity - eq) / self.peak_equity
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd

        self._record_snapshot(timestamp, prices)

    def _record_snapshot(self, timestamp: int, prices: dict[str, float]) -> None:
        eq = 0.0 if self.is_bankrupt else self.equity(prices)
        notional = sum(inv.notional(prices.get(sym, 0.0)) for sym, inv in self.inventories.items())
        active = sum(1 for inv in self.inventories.values() if inv.qty != 0.0 or inv.allocated_margin > 0)
        dd = 1.0 if self.is_bankrupt else ((self.peak_equity - eq) / self.peak_equity if self.peak_equity > 0 else 0.0)

        self.history.append(
            PortfolioSnapshot(
                timestamp=timestamp,
                wallet_balance=self.wallet_balance,
                equity=eq,
                margin_allocated=self.total_margin_allocated(),
                available_margin=self.available_margin(prices),
                active_positions_count=active,
                total_notional=notional,
                drawdown_pct=dd,
            )
        )

    def net_pnl(self, prices: dict[str, float]) -> float:
        return self.equity(prices) - self.initial_capital

    def return_pct(self, prices: dict[str, float]) -> float:
        if self.initial_capital <= 0:
            return 0.0
        ret = self.net_pnl(prices) / self.initial_capital
        return max(-1.0, ret)  # Strictly bounded >= -100%
