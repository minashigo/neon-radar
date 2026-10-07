"""Performance and Risk Metrics for Futures Grid Experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research.experiments.futures_grid.simulator import GridSessionResult


@dataclass(frozen=True)
class GridMetricsReport:
    total_sessions: int
    winning_sessions: int
    losing_sessions: int
    win_rate: float
    total_net_pnl: float
    total_return_pct: float
    realized_grid_profit: float
    avg_session_pnl: float
    profit_factor: float
    expectancy: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    liquidations_count: int
    liquidation_rate: float
    completed_grids_total: int
    avg_completed_grids_per_session: float
    total_fees_paid: float
    total_funding_paid: float
    peak_inventory_max: float
    max_notional_exposure: float
    tail_var_95: float
    tail_cvar_95: float
    avg_duration_hours: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_sessions": self.total_sessions,
            "winning_sessions": self.winning_sessions,
            "losing_sessions": self.losing_sessions,
            "win_rate": round(self.win_rate, 4),
            "total_net_pnl": round(self.total_net_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "realized_grid_profit": round(self.realized_grid_profit, 2),
            "avg_session_pnl": round(self.avg_session_pnl, 2),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy": round(self.expectancy, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "liquidations_count": self.liquidations_count,
            "liquidation_rate": round(self.liquidation_rate, 4),
            "completed_grids_total": self.completed_grids_total,
            "avg_completed_grids_per_session": round(self.avg_completed_grids_per_session, 2),
            "total_fees_paid": round(self.total_fees_paid, 2),
            "total_funding_paid": round(self.total_funding_paid, 2),
            "peak_inventory_max": round(self.peak_inventory_max, 4),
            "max_notional_exposure": round(self.max_notional_exposure, 2),
            "tail_var_95": round(self.tail_var_95, 4),
            "tail_cvar_95": round(self.tail_cvar_95, 4),
            "avg_duration_hours": round(self.avg_duration_hours, 2),
        }


def calculate_grid_metrics(
    sessions: list[GridSessionResult],
    initial_capital: float = 10000.0,
) -> GridMetricsReport:
    """Computes comprehensive performance and risk metrics over grid sessions."""
    if not sessions:
        return GridMetricsReport(
            total_sessions=0,
            winning_sessions=0,
            losing_sessions=0,
            win_rate=0.0,
            total_net_pnl=0.0,
            total_return_pct=0.0,
            realized_grid_profit=0.0,
            avg_session_pnl=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=0.0,
            liquidations_count=0,
            liquidation_rate=0.0,
            completed_grids_total=0,
            avg_completed_grids_per_session=0.0,
            total_fees_paid=0.0,
            total_funding_paid=0.0,
            peak_inventory_max=0.0,
            max_notional_exposure=0.0,
            tail_var_95=0.0,
            tail_cvar_95=0.0,
            avg_duration_hours=0.0,
        )

    n = len(sessions)
    pnls = [s.net_pnl for s in sessions]
    rois = [s.roi for s in sessions]
    wins = [p for p in pnls if p > 0.0]
    losses = [p for p in pnls if p < 0.0]

    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n if n > 0 else 0.0

    total_net_pnl = sum(pnls)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    expectancy = total_net_pnl / n if n > 0 else 0.0

    # Risk metrics: Sharpe & Sortino (per-session basis)
    mean_roi = sum(rois) / n if n > 0 else 0.0
    var_roi = sum((r - mean_roi) ** 2 for r in rois) / (n - 1) if n > 1 else 0.0
    std_roi = math.sqrt(var_roi) if var_roi > 0 else 0.0
    sharpe_ratio = (mean_roi / std_roi) if std_roi > 0 else 0.0

    downside_rois = [min(0.0, r) for r in rois]
    downside_var = sum(r**2 for r in downside_rois) / n if n > 0 else 0.0
    downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0
    sortino_ratio = (mean_roi / downside_std) if downside_std > 0 else 0.0

    # Drawdown on cumulative equity
    equity = initial_capital
    peak = equity
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd

    total_return_pct = total_net_pnl / initial_capital if initial_capital > 0 else 0.0
    calmar_ratio = (total_return_pct / max_dd) if max_dd > 0 else 0.0

    # Liquidations
    liquidations = [s for s in sessions if s.is_liquidation]
    liquidations_count = len(liquidations)
    liquidation_rate = liquidations_count / n if n > 0 else 0.0

    # Grid stats
    realized_grid_profit = sum(s.realized_grid_profit for s in sessions)
    completed_grids_total = sum(s.completed_grids_count for s in sessions)
    avg_completed_grids = completed_grids_total / n if n > 0 else 0.0

    total_fees_paid = sum(s.total_fees_paid for s in sessions)
    total_funding_paid = sum(s.total_funding_paid for s in sessions)

    peak_inventory_max = max((s.peak_inventory_abs for s in sessions), default=0.0)
    max_notional_exposure = max((s.max_notional_exposure for s in sessions), default=0.0)

    # Tail losses (5% quantile VaR and CVaR on ROI)
    sorted_rois = sorted(rois)
    var_idx = int(0.05 * n)
    tail_var_95 = sorted_rois[var_idx] if sorted_rois else 0.0
    worst_tail = sorted_rois[: max(1, var_idx + 1)]
    tail_cvar_95 = sum(worst_tail) / len(worst_tail) if worst_tail else 0.0

    # Duration
    durations_h = [(s.end_time - s.start_time) / (1000 * 3600) for s in sessions]
    avg_duration_hours = sum(durations_h) / n if n > 0 else 0.0

    return GridMetricsReport(
        total_sessions=n,
        winning_sessions=n_wins,
        losing_sessions=n_losses,
        win_rate=win_rate,
        total_net_pnl=total_net_pnl,
        total_return_pct=total_return_pct,
        realized_grid_profit=realized_grid_profit,
        avg_session_pnl=expectancy,
        profit_factor=profit_factor,
        expectancy=expectancy,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown_pct=max_dd,
        calmar_ratio=calmar_ratio,
        liquidations_count=liquidations_count,
        liquidation_rate=liquidation_rate,
        completed_grids_total=completed_grids_total,
        avg_completed_grids_per_session=avg_completed_grids,
        total_fees_paid=total_fees_paid,
        total_funding_paid=total_funding_paid,
        peak_inventory_max=peak_inventory_max,
        max_notional_exposure=max_notional_exposure,
        tail_var_95=tail_var_95,
        tail_cvar_95=tail_cvar_95,
        avg_duration_hours=avg_duration_hours,
    )
