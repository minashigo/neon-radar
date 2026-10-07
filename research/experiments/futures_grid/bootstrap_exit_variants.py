"""Synchronized 30-Day Calendar Block Bootstrap for Exit Management Variants.

Preserves cross-asset cross-sectional correlation and temporal autocorrelation.
Performs 2,000 bootstrap iterations comparing:
- Variant B (Freeze) vs Variant A (Immediate Close)
- Variant C (Freeze + ATR) vs Variant A (Immediate Close)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # noqa: TC002
from research.experiments.futures_grid.portfolio import ExitVariant


@dataclass(frozen=True)
class BootstrapStat:
    mean: float
    median: float
    std: float
    ci_lower_95: float
    ci_upper_95: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "std": round(self.std, 4),
            "ci_lower_95": round(self.ci_lower_95, 4),
            "ci_upper_95": round(self.ci_upper_95, 4),
        }


@dataclass(frozen=True)
class ExitVariantBootstrapComparison:
    candidate_variant: str
    reference_variant: str
    iterations: int
    block_days: int
    delta_pnl: BootstrapStat
    delta_return: BootstrapStat
    delta_sharpe: BootstrapStat
    p_superiority: float
    p_value: float
    is_significant: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_variant": self.candidate_variant,
            "reference_variant": self.reference_variant,
            "iterations": self.iterations,
            "block_days": self.block_days,
            "delta_pnl": self.delta_pnl.to_dict(),
            "delta_return": self.delta_return.to_dict(),
            "delta_sharpe": self.delta_sharpe.to_dict(),
            "p_superiority": round(self.p_superiority, 4),
            "p_value": round(self.p_value, 4),
            "is_significant": self.is_significant,
        }


def _calc_stat(vals: list[float]) -> BootstrapStat:
    arr = np.array(vals, dtype=float)
    return BootstrapStat(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        ci_lower_95=float(np.percentile(arr, 2.5)),
        ci_upper_95=float(np.percentile(arr, 97.5)),
    )


def run_synchronized_block_bootstrap(
    daily_equity_curves: dict[str, pd.Series],
    iterations: int = 2000,
    block_size_days: int = 30,
    seed: int = 42,
) -> dict[str, ExitVariantBootstrapComparison]:
    """Runs synchronized circular block bootstrap across daily return series.

    daily_equity_curves: dict mapping variant_name -> pd.Series of daily equity
    """
    random.seed(seed)
    np.random.seed(seed)

    # Compute daily return series
    daily_returns: dict[str, pd.Series] = {}
    for var_name, eq_series in daily_equity_curves.items():
        daily_returns[var_name] = eq_series.pct_change().fillna(0.0)

    # Common index
    common_idx = daily_returns[ExitVariant.IMMEDIATE_CLOSE.value].index
    n_days = len(common_idx)
    num_blocks = int(np.ceil(n_days / block_size_days))

    ref_name = ExitVariant.IMMEDIATE_CLOSE.value
    ref_returns = daily_returns[ref_name].to_numpy()

    comparisons: dict[str, ExitVariantBootstrapComparison] = {}

    candidates = [
        v for v in daily_returns if v != ref_name
    ]

    for cand_name in candidates:
        cand_returns = daily_returns[cand_name].to_numpy()

        delta_pnls: list[float] = []
        delta_returns_list: list[float] = []
        delta_sharpes: list[float] = []

        for _ in range(iterations):
            # Sample starting indices for blocks
            start_indices = np.random.randint(0, n_days, size=num_blocks)
            sampled_indices = []
            for s in start_indices:
                block = [(s + j) % n_days for j in range(block_size_days)]
                sampled_indices.extend(block)
            sampled_indices = sampled_indices[:n_days]

            # Reconstruct returns
            boot_ref = ref_returns[sampled_indices]
            boot_cand = cand_returns[sampled_indices]

            # Cumulative return
            ret_ref = float(np.prod(1.0 + boot_ref) - 1.0)
            ret_cand = float(np.prod(1.0 + boot_cand) - 1.0)
            pnl_ref = ret_ref * 10000.0
            pnl_cand = ret_cand * 10000.0

            # Sharpe
            ann = np.sqrt(365.0)
            std_ref = np.std(boot_ref)
            std_cand = np.std(boot_cand)
            sharpe_ref = (np.mean(boot_ref) / std_ref * ann) if std_ref > 1e-8 else 0.0
            sharpe_cand = (np.mean(boot_cand) / std_cand * ann) if std_cand > 1e-8 else 0.0

            delta_pnls.append(pnl_cand - pnl_ref)
            delta_returns_list.append(ret_cand - ret_ref)
            delta_sharpes.append(sharpe_cand - sharpe_ref)

        p_sup = float(np.mean(np.array(delta_pnls) > 0))
        p_val = float(np.mean(np.array(delta_pnls) <= 0))
        is_sig = p_val < 0.05

        comparisons[cand_name] = ExitVariantBootstrapComparison(
            candidate_variant=cand_name,
            reference_variant=ref_name,
            iterations=iterations,
            block_days=block_size_days,
            delta_pnl=_calc_stat(delta_pnls),
            delta_return=_calc_stat(delta_returns_list),
            delta_sharpe=_calc_stat(delta_sharpes),
            p_superiority=p_sup,
            p_value=p_val,
            is_significant=is_sig,
        )

    return comparisons
