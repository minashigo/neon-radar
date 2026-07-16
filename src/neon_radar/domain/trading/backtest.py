"""Domain models for Trade-based backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.enums import Bias
    from neon_radar.domain.models import Symbol


class TradeStatus(StrEnum):
    """The outcome of a simulated trade."""

    OPEN = "open"
    WIN = "win"
    LOSS = "loss"
    BREAK_EVEN = "break_even"


@dataclass(slots=True, frozen=True)
class Trade:
    """A simulated trade execution."""

    symbol: Symbol
    direction: Bias
    entry_time: int
    entry_price: float
    stop_loss: float
    take_profit: float

    # These are populated when the trade closes
    exit_time: int | None = None
    exit_price: float | None = None
    status: TradeStatus = TradeStatus.OPEN

    @property
    def pnl_pct(self) -> float:
        """Percentage profit/loss of this trade."""
        if self.exit_price is None:
            return 0.0
        if self.entry_price == 0:
            return 0.0

        raw_pnl = (self.exit_price - self.entry_price) / self.entry_price
        if self.direction.name == "BEARISH":
            return -raw_pnl
        return raw_pnl


@dataclass(slots=True, frozen=True)
class BacktestReport:
    """Aggregated statistics of a Trade-based backtest."""

    total_trades: int
    win_rate: float
    wins: int
    losses: int
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float

    trades: tuple[Trade, ...] = field(default_factory=tuple)

    @classmethod
    def from_trades(cls, trades: list[Trade]) -> BacktestReport:
        """Compute metrics from a list of completed trades."""
        total = len(trades)
        if total == 0:
            return cls(
                total_trades=0,
                win_rate=0.0,
                wins=0,
                losses=0,
                avg_win_pct=0.0,
                avg_loss_pct=0.0,
                profit_factor=0.0,
                trades=(),
            )

        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct < 0]

        n_wins = len(wins)
        n_losses = len(losses)

        sum_wins = sum(t.pnl_pct for t in wins)
        sum_losses = sum(abs(t.pnl_pct) for t in losses)

        avg_win = sum_wins / n_wins if n_wins > 0 else 0.0
        avg_loss = sum_losses / n_losses if n_losses > 0 else 0.0

        profit_factor = sum_wins / sum_losses if sum_losses > 0 else float("inf")
        if n_losses == 0 and n_wins == 0:
            profit_factor = 0.0

        win_rate = n_wins / total

        return cls(
            total_trades=total,
            win_rate=win_rate,
            wins=n_wins,
            losses=n_losses,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            trades=tuple(trades),
        )
