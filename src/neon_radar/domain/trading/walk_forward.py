"""Domain models for Walk-Forward Analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.config.scoring_models import ScoringRulesConfig
    from neon_radar.domain.trading.backtest import BacktestReport


@dataclass(slots=True, frozen=True)
class WalkForwardCycle:
    """Represents a single step in a Walk-Forward Analysis."""

    # In-Sample period
    is_start: date
    is_end: date

    # Out-Of-Sample period
    oos_start: date
    oos_end: date

    # The best configuration found during the In-Sample optimization
    optimized_config: ScoringRulesConfig

    # Metrics on the IS window for the selected configuration
    is_report: BacktestReport

    # Metrics on the OOS window using the optimized configuration
    oos_report: BacktestReport


@dataclass(slots=True, frozen=True)
class WalkForwardReport:
    """Aggregation of all Walk-Forward cycles."""

    cycles: tuple[WalkForwardCycle, ...]

    @property
    def is_valid(self) -> bool:
        return len(self.cycles) > 0

    @property
    def total_oos_trades(self) -> int:
        return sum(c.oos_report.total_trades for c in self.cycles)

    @property
    def avg_oos_expectancy(self) -> float:
        if not self.cycles:
            return 0.0
        return sum(c.oos_report.net_expectancy for c in self.cycles) / len(self.cycles)

    @property
    def avg_oos_profit_factor(self) -> float:
        if not self.cycles:
            return 0.0
        return sum(c.oos_report.net_profit_factor for c in self.cycles) / len(self.cycles)

    @property
    def avg_oos_win_rate(self) -> float:
        if not self.cycles:
            return 0.0
        return sum(c.oos_report.win_rate for c in self.cycles) / len(self.cycles)

    @property
    def avg_oos_sharpe(self) -> float:
        if not self.cycles:
            return 0.0
        return sum(c.oos_report.net_sharpe_ratio for c in self.cycles) / len(self.cycles)

    @property
    def avg_oos_max_drawdown(self) -> float:
        if not self.cycles:
            return 0.0
        return sum(c.oos_report.max_drawdown_pct for c in self.cycles) / len(self.cycles)

    @property
    def best_cycle(self) -> WalkForwardCycle | None:
        if not self.cycles:
            return None
        # Maximize Expectancy
        return max(self.cycles, key=lambda c: c.oos_report.net_expectancy)

    @property
    def worst_cycle(self) -> WalkForwardCycle | None:
        if not self.cycles:
            return None
        # Minimize Expectancy
        return min(self.cycles, key=lambda c: c.oos_report.net_expectancy)

    @property
    def profitable_cycles_pct(self) -> float:
        if not self.cycles:
            return 0.0
        profitable = sum(1 for c in self.cycles if c.oos_report.net_expectancy > 0)
        return profitable / len(self.cycles)
