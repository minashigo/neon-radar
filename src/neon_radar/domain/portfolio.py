"""Domain models for Portfolio Management.

Design notes
------------
* All models use `@dataclass(slots=True, frozen=True)` to maintain immutability.
* Time-based properties use Unix milliseconds for consistency with OHLCV data.
* Position size and capital amounts are represented in the quote asset (e.g. USDT)
  or base asset depending on context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.enums import Bias
    from neon_radar.domain.execution_costs import ExecutionCostSummary
    from neon_radar.domain.models import Symbol


class PositionCloseReason(StrEnum):
    """Reason for a position being closed."""

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    MANUAL = "MANUAL"
    TIME_EXIT = "TIME_EXIT"
    LIQUIDATION = "LIQUIDATION"
    OTHER = "OTHER"


@dataclass(slots=True, frozen=True)
class AccountState:
    """Represents the basic capital state of a trading account."""

    total_capital: float
    free_capital: float
    currency: str = "USDT"

    def __post_init__(self) -> None:
        if self.total_capital < 0:
            raise ValueError("total_capital cannot be negative")
        if self.free_capital < 0:
            raise ValueError("free_capital cannot be negative")
        if self.free_capital > self.total_capital:
            raise ValueError("free_capital cannot exceed total_capital")


@dataclass(slots=True, frozen=True)
class OpenPosition:
    """Represents a single active trading position."""

    symbol: Symbol
    direction: Bias
    entry_price: float
    quantity: float  # Base asset amount (e.g., amount of BTC)
    position_size: float  # Value in quote asset
    stop_loss: float
    take_profit: float
    opened_at: int
    capital_at_entry: float = 0.0
    unrealized_pnl: float = 0.0
    entry_fee: float = 0.0
    entry_slippage: float = 0.0
    entry_execution_type: str = "taker"

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError("entry_price must be > 0")
        if self.quantity <= 0:
            raise ValueError("quantity must be > 0")
        if self.position_size <= 0:
            raise ValueError("position_size must be > 0")

    @property
    def max_risk(self) -> float:
        """Calculates the max risk in quote asset if stop loss is hit."""
        # Max risk is the absolute distance * quantity
        return abs(self.entry_price - self.stop_loss) * self.quantity


@dataclass(slots=True, frozen=True)
class ClosedPosition:
    """Represents a finalized, closed trading position."""

    symbol: Symbol
    direction: Bias
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: int
    exit_time: int
    execution_summary: ExecutionCostSummary
    close_reason: PositionCloseReason
    stop_loss: float = 0.0
    take_profit: float = 0.0
    initial_risk: float = 0.0
    capital_at_entry: float = 0.0

    @property
    def duration(self) -> int:
        """Duration of the trade in milliseconds."""
        return self.exit_time - self.entry_time

    @property
    def net_pnl(self) -> float:
        """Realized PnL net of all costs."""
        return self.execution_summary.net_pnl

    @property
    def gross_pnl(self) -> float:
        """Realized gross PnL before costs."""
        return self.execution_summary.gross_pnl

    @property
    def notional(self) -> float:
        """Quote notional value at entry."""
        return self.entry_price * self.quantity

    @property
    def profit_r(self) -> float:
        """Realized profit in multiples of initial risk R."""
        if self.initial_risk > 0:
            return self.net_pnl / self.initial_risk
        return 0.0

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0


@dataclass(slots=True, frozen=True)
class PortfolioStatistics:
    """Trading statistics aggregated over the portfolio history."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def net_profit(self) -> float:
        return self.gross_profit - self.gross_loss - self.total_trading_costs

    @property
    def total_trading_costs(self) -> float:
        return self.total_fees + self.total_slippage + self.total_funding

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0.0:
            return self.gross_profit if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss

    @property
    def average_win(self) -> float:
        if self.winning_trades == 0:
            return 0.0
        return self.gross_profit / self.winning_trades

    @property
    def average_loss(self) -> float:
        if self.losing_trades == 0:
            return 0.0
        return self.gross_loss / self.losing_trades

    @property
    def expectancy(self) -> float:
        return (self.win_rate * self.average_win) - ((1 - self.win_rate) * self.average_loss)


@dataclass(slots=True, frozen=True)
class PortfolioState:
    """Aggregates the AccountState, active positions, and statistics at a point in time."""

    account: AccountState
    positions: tuple[OpenPosition, ...] = field(default_factory=tuple)
    statistics: PortfolioStatistics = field(default_factory=PortfolioStatistics)
    timestamp: int = 0

    @property
    def total_exposure(self) -> float:
        """Total active exposure in quote asset."""
        return sum(pos.position_size for pos in self.positions)

    @property
    def total_risk(self) -> float:
        """Total capital at risk (if all stop losses are hit)."""
        return sum(pos.max_risk for pos in self.positions)
