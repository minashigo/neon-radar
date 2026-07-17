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


class TradeExitReason(StrEnum):
    """The reason why a trade was exited."""

    NONE = "none"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    FORCE_CLOSE = "force_close"


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
    exit_reason: TradeExitReason = TradeExitReason.NONE

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
class StatisticalValidationReport:
    """Results of statistical validation for a trading strategy."""

    is_valid: bool  # True if the validation was performed
    p_value: float
    t_statistic: float
    mc_expectancy_95_ci_lower: float
    mc_expectancy_95_ci_upper: float
    mc_probability_of_loss: float


@dataclass(slots=True, frozen=True)
class BacktestReport:
    """Aggregated statistics of a Trade-based backtest.

    This is a pure data transfer object (DTO). The calculation of these
    metrics is handled by `TradeAnalyzer`.
    """

    total_trades: int
    win_rate: float
    wins: int
    losses: int
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_holding_time_ms: float

    validation: StatisticalValidationReport | None = None
    trades: tuple[Trade, ...] = field(default_factory=tuple)
