"""Domain models for Trade-based backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.enums import Bias
    from neon_radar.domain.models import Symbol
    from neon_radar.domain.trading.execution import TradeCosts


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


class TradeEntryReason(StrEnum):
    """The reason why a trade was entered."""

    CONFIDENCE_THRESHOLD = "confidence_threshold"
    MANUAL = "manual"


@dataclass(slots=True, frozen=True)
class TradeDiagnostics:
    """Telemetry data collected at the moment the setup was generated."""

    adx: float | None
    atr: float | None
    rsi: float | None
    ema_spread_pct: float | None
    htf_trend: float | None
    confidence: float
    final_score: float
    triggered_rules: str  # e.g. "ema_trend:-0.40, rsi_momentum:0.25"
    entry_reason: TradeEntryReason


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
    costs: TradeCosts | None = None
    diagnostics: TradeDiagnostics | None = None

    @property
    def gross_pnl_pct(self) -> float:
        """Gross percentage profit/loss of this trade."""
        if self.exit_price is None:
            return 0.0
        if self.entry_price == 0:
            return 0.0

        raw_pnl = (self.exit_price - self.entry_price) / self.entry_price
        if self.direction.name == "BEARISH":
            return -raw_pnl
        return raw_pnl

    @property
    def net_pnl_pct(self) -> float:
        """Net percentage profit/loss after execution costs."""
        if self.costs is None:
            return self.gross_pnl_pct
        return self.gross_pnl_pct - self.costs.total_costs_pct


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

    # Gross metrics
    gross_avg_win_pct: float
    gross_avg_loss_pct: float
    gross_profit_factor: float
    gross_expectancy: float

    # Net metrics
    net_profit_pct: float
    net_avg_win_pct: float
    net_avg_loss_pct: float
    net_profit_factor: float
    net_expectancy: float
    net_sharpe_ratio: float

    # Cost metrics
    avg_trade_cost_pct: float
    avg_slippage_pct: float
    total_fees_pct: float
    total_funding_pct: float

    max_consecutive_wins: int
    max_consecutive_losses: int
    max_drawdown_pct: float
    avg_holding_time_ms: float

    validation: StatisticalValidationReport | None = None
    trades: tuple[Trade, ...] = field(default_factory=tuple)
