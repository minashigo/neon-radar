"""Paper Trading Domain Models.

Contains models for representing virtual positions and portfolios.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from neon_radar.domain.enums import Bias
from neon_radar.domain.models import Symbol
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from neon_radar.domain.models import Kline
    from neon_radar.domain.trading.backtest import TradeDiagnostics
    from neon_radar.domain.trading.setup import TradeSetup

logger = get_logger(__name__)


@dataclass(slots=True)
class VirtualPosition:
    """Represents an active virtual position in Paper Trading."""

    symbol: str
    direction: Bias
    entry_time: int
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float  # We simplify to a single TP for now, or just use TP1 from setup

    # New fields for trade evaluation
    diagnostics: TradeDiagnostics | None = None
    analysis_snapshot: dict | None = None

    # Internal stats
    highest_price: float = field(init=False)
    lowest_price: float = field(init=False)

    def __post_init__(self) -> None:
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price

    @classmethod
    def from_setup(
        cls,
        symbol: Symbol,
        setup: TradeSetup,
        quantity: float,
        entry_time: int,
        analysis_snapshot: dict | None = None
    ) -> VirtualPosition:
        """Create a new VirtualPosition from a TradeSetup."""
        return cls(
            symbol=str(symbol),
            direction=setup.direction,
            entry_time=entry_time,
            entry_price=setup.entry_price,
            quantity=quantity,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit_1,  # Using TP1 as primary target for paper trading V1
            diagnostics=setup.diagnostics,
            analysis_snapshot=analysis_snapshot,
        )

    def update(self, kline: Kline) -> str | None:
        """Check if the kline hits SL or TP. Updates high/low watermarks.
        
        Returns the exit reason ("SL", "TP") or None if still active.
        """
        # Update watermarks
        if kline.high > self.highest_price:
            self.highest_price = kline.high
        if kline.low < self.lowest_price:
            self.lowest_price = kline.low

        if self.direction == Bias.BULLISH:
            if kline.low <= self.stop_loss:
                return "SL"
            if kline.high >= self.take_profit:
                return "TP"
        else:
            if kline.high >= self.stop_loss:
                return "SL"
            if kline.low <= self.take_profit:
                return "TP"

        return None

    def calculate_pnl(self, exit_price: float) -> float:
        """Calculate raw PnL in quote currency (excluding fees)."""
        if self.direction == Bias.BULLISH:
            return (exit_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - exit_price) * self.quantity


@dataclass
class VirtualPortfolio:
    """Manages the state of paper trading balance and active positions."""

    balance: float
    risk_per_trade: float
    positions: dict[str, VirtualPosition] = field(default_factory=dict)

    # Persistence
    filepath: Path | None = None

    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a position for the given symbol."""
        return symbol not in self.positions

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size (in base asset) based on fixed risk percentage."""
        risk_amount = self.balance * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit <= 0:
            return 0.0

        quantity = risk_amount / risk_per_unit
        return quantity

    def open_position(self, position: VirtualPosition) -> None:
        """Register a new open position."""
        if position.symbol in self.positions:
            raise ValueError(f"Position already exists for {position.symbol}")

        self.positions[position.symbol] = position
        logger.info(
            f"Opened {position.direction.value} on {position.symbol} "
            f"at {position.entry_price:.4f} (Qty: {position.quantity:.6f})"
        )
        self.save()

    def close_position(self, symbol: str, exit_price: float, reason: str, exit_time: int) -> float:
        """Close a position, realize PnL, and update balance. Returns realized PnL."""
        if symbol not in self.positions:
            raise ValueError(f"No active position for {symbol}")

        position = self.positions.pop(symbol)
        pnl = position.calculate_pnl(exit_price)

        # We simulate 0.05% Taker fee on entry and exit for realism
        fee_rate = 0.0005
        entry_fee = (position.entry_price * position.quantity) * fee_rate
        exit_fee = (exit_price * position.quantity) * fee_rate

        net_pnl = pnl - entry_fee - exit_fee
        self.balance += net_pnl

        logger.info(
            f"Closed {position.direction.value} on {position.symbol} "
            f"at {exit_price:.4f} (Reason: {reason}, Net PnL: {net_pnl:.2f}, New Balance: {self.balance:.2f})"
        )
        self.save()
        return net_pnl

    def save(self) -> None:
        """Persist portfolio state to JSON."""
        if not self.filepath:
            return

        try:
            state = {
                "balance": self.balance,
                "risk_per_trade": self.risk_per_trade,
                "positions": {
                    sym: {
                        "direction": p.direction.value,
                        "entry_time": p.entry_time,
                        "entry_price": p.entry_price,
                        "quantity": p.quantity,
                        "stop_loss": p.stop_loss,
                        "take_profit": p.take_profit,
                        "highest_price": p.highest_price,
                        "lowest_price": p.lowest_price,
                    }
                    for sym, p in self.positions.items()
                }
            }
            self.filepath.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")

    @classmethod
    def load(cls, filepath: Path, default_balance: float, risk_per_trade: float) -> VirtualPortfolio:
        """Load portfolio state from JSON, or create new if not exists."""
        if not filepath.exists():
            return cls(balance=default_balance, risk_per_trade=risk_per_trade, filepath=filepath)

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            portfolio = cls(
                balance=data.get("balance", default_balance),
                risk_per_trade=data.get("risk_per_trade", risk_per_trade),
                filepath=filepath
            )

            for sym, p_data in data.get("positions", {}).items():
                pos = VirtualPosition(
                    symbol=sym,
                    direction=Bias(p_data["direction"]),
                    entry_time=p_data["entry_time"],
                    entry_price=p_data["entry_price"],
                    quantity=p_data["quantity"],
                    stop_loss=p_data["stop_loss"],
                    take_profit=p_data["take_profit"],
                )
                pos.highest_price = p_data.get("highest_price", pos.entry_price)
                pos.lowest_price = p_data.get("lowest_price", pos.entry_price)
                portfolio.positions[sym] = pos

            return portfolio

        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}. Starting fresh.")
            return cls(balance=default_balance, risk_per_trade=risk_per_trade, filepath=filepath)
