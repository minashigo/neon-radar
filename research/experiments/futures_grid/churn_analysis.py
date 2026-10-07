"""Churn Analysis and Regime Transition Dynamics for Futures Grid Exits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd  # noqa: TC002
from research.experiments.futures_grid.exit_management_engine import (
    RegimeExitRecord,  # noqa: TC002
)


@dataclass(frozen=True)
class ChurnMetrics:
    variant: str
    total_regime_exits: int
    immediate_exits: int
    atr_trailing_exits: int
    sl_exits: int
    recenter_exits: int
    timeout_exits: int
    total_taker_fees: float
    total_slippage_cost: float
    total_churn_cost: float
    avg_churn_cost_per_exit: float
    net_pnl_at_exit: float
    protective_exits_count: int
    protective_exit_rate: float
    spurious_exits_count: int
    spurious_exit_rate: float
    indeterminate_exits_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "total_regime_exits": self.total_regime_exits,
            "immediate_exits": self.immediate_exits,
            "atr_trailing_exits": self.atr_trailing_exits,
            "sl_exits": self.sl_exits,
            "recenter_exits": self.recenter_exits,
            "timeout_exits": self.timeout_exits,
            "total_taker_fees": round(self.total_taker_fees, 2),
            "total_slippage_cost": round(self.total_slippage_cost, 2),
            "total_churn_cost": round(self.total_churn_cost, 2),
            "avg_churn_cost_per_exit": round(self.avg_churn_cost_per_exit, 2),
            "net_pnl_at_exit": round(self.net_pnl_at_exit, 2),
            "protective_exits_count": self.protective_exits_count,
            "protective_exit_rate": round(self.protective_exit_rate, 4),
            "spurious_exits_count": self.spurious_exits_count,
            "spurious_exit_rate": round(self.spurious_exit_rate, 4),
            "indeterminate_exits_count": self.indeterminate_exits_count,
        }


def analyze_churn(records: list[RegimeExitRecord], variant_name: str) -> ChurnMetrics:
    """Computes comprehensive churn metrics from exit records."""
    if not records:
        return ChurnMetrics(
            variant=variant_name,
            total_regime_exits=0,
            immediate_exits=0,
            atr_trailing_exits=0,
            sl_exits=0,
            recenter_exits=0,
            timeout_exits=0,
            total_taker_fees=0.0,
            total_slippage_cost=0.0,
            total_churn_cost=0.0,
            avg_churn_cost_per_exit=0.0,
            net_pnl_at_exit=0.0,
            protective_exits_count=0,
            protective_exit_rate=0.0,
            spurious_exits_count=0,
            spurious_exit_rate=0.0,
            indeterminate_exits_count=0,
        )

    n_total = len(records)
    imm = sum(1 for r in records if r.exit_type in ("IMMEDIATE", "HYSTERESIS_CONFIRMED"))
    atr = sum(1 for r in records if r.exit_type == "ATR_TRAILING")
    sl = sum(1 for r in records if r.exit_type == "SL_HIT")
    recenter = sum(1 for r in records if r.exit_type == "RECENTER_CHOP")
    timeout = sum(1 for r in records if r.exit_type == "TIMEOUT")

    tot_fees = sum(r.taker_fee for r in records)
    tot_slip = sum(r.slippage_cost for r in records)
    tot_churn = sum(r.churn_cost for r in records)
    tot_pnl = sum(r.net_pnl for r in records)

    prot = sum(1 for r in records if r.is_protective)
    spur = sum(1 for r in records if r.is_spurious)
    indet = n_total - (prot + spur)

    return ChurnMetrics(
        variant=variant_name,
        total_regime_exits=n_total,
        immediate_exits=imm,
        atr_trailing_exits=atr,
        sl_exits=sl,
        recenter_exits=recenter,
        timeout_exits=timeout,
        total_taker_fees=tot_fees,
        total_slippage_cost=tot_slip,
        total_churn_cost=tot_churn,
        avg_churn_cost_per_exit=tot_churn / n_total if n_total > 0 else 0.0,
        net_pnl_at_exit=tot_pnl,
        protective_exits_count=prot,
        protective_exit_rate=prot / n_total if n_total > 0 else 0.0,
        spurious_exits_count=spur,
        spurious_exit_rate=spur / n_total if n_total > 0 else 0.0,
        indeterminate_exits_count=indet,
    )


def count_regime_flickers(regimes_df: pd.DataFrame) -> dict[str, int]:
    """Counts 1-bar flickers in the regime series."""
    allowed = regimes_df["grid_allowed"].tolist()
    n = len(allowed)
    chop_to_trend_flickers = 0  # True -> False -> True
    trend_to_chop_flickers = 0  # False -> True -> False

    for i in range(1, n - 1):
        if allowed[i - 1] and not allowed[i] and allowed[i + 1]:
            chop_to_trend_flickers += 1
        elif not allowed[i - 1] and allowed[i] and not allowed[i + 1]:
            trend_to_chop_flickers += 1

    return {
        "1bar_chop_interruption_flickers": chop_to_trend_flickers,
        "1bar_chop_pulse_flickers": trend_to_chop_flickers,
        "total_flickers": chop_to_trend_flickers + trend_to_chop_flickers,
    }
