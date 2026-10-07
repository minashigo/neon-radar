"""Block Bootstrap and Statistical Validation for Futures Grid Experiments.

Implements Paired Circular Block Bootstrap (CBB) to preserve temporal autocorrelation,
regime clustering, and equity path dependency.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from research.experiments.futures_grid.metrics import calculate_grid_metrics

if TYPE_CHECKING:
    from research.experiments.futures_grid.simulator import GridSessionResult


@dataclass(frozen=True)
class BootstrapDistribution:
    mean: float
    median: float
    std: float
    ci_lower_95: float
    ci_upper_95: float


@dataclass(frozen=True)
class GridBootstrapReport:
    iterations: int
    block_size: int
    baseline_return: BootstrapDistribution
    baseline_max_dd: BootstrapDistribution
    baseline_sharpe: BootstrapDistribution
    radar_return: BootstrapDistribution
    radar_max_dd: BootstrapDistribution
    radar_sharpe: BootstrapDistribution
    delta_net_pnl: BootstrapDistribution
    delta_sharpe: BootstrapDistribution
    delta_max_dd: BootstrapDistribution
    p_superior_pnl: float
    p_superior_sharpe: float
    p_value_superiority: float
    is_statistically_significant: bool


def _build_dist(values: list[float]) -> BootstrapDistribution:
    arr = np.array(values, dtype=float)
    return BootstrapDistribution(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        ci_lower_95=float(np.percentile(arr, 2.5)),
        ci_upper_95=float(np.percentile(arr, 97.5)),
    )


def run_paired_block_bootstrap(
    baseline_sessions: list[GridSessionResult],
    radar_sessions: list[GridSessionResult],
    iterations: int = 2000,
    block_size: int = 5,
    seed: int = 42,
) -> GridBootstrapReport:
    """Runs a paired block bootstrap to respect serial correlation."""
    if not baseline_sessions or not radar_sessions:
        raise ValueError("Sessions cannot be empty for bootstrap")

    rng = random.Random(seed)
    n_b = len(baseline_sessions)
    n_r = len(radar_sessions)

    # Number of blocks to reconstruct length
    num_blocks_b = max(1, n_b // block_size)
    num_blocks_r = max(1, n_r // block_size)

    base_rets = []
    base_dds = []
    base_shs = []

    radar_rets = []
    radar_dds = []
    radar_shs = []

    delta_pnls = []
    delta_shs = []
    delta_dds = []

    for _ in range(iterations):
        # 1. Resample Baseline using circular blocks
        sample_b: list[GridSessionResult] = []
        for _ in range(num_blocks_b):
            start_idx = rng.randint(0, n_b - 1)
            for k in range(block_size):
                sample_b.append(baseline_sessions[(start_idx + k) % n_b])

        # 2. Resample Radar using circular blocks
        sample_r: list[GridSessionResult] = []
        for _ in range(num_blocks_r):
            start_idx = rng.randint(0, n_r - 1)
            for k in range(block_size):
                sample_r.append(radar_sessions[(start_idx + k) % n_r])

        m_b = calculate_grid_metrics(sample_b)
        m_r = calculate_grid_metrics(sample_r)

        base_rets.append(m_b.total_return_pct)
        base_dds.append(m_b.max_drawdown_pct)
        base_shs.append(m_b.sharpe_ratio)

        radar_rets.append(m_r.total_return_pct)
        radar_dds.append(m_r.max_drawdown_pct)
        radar_shs.append(m_r.sharpe_ratio)

        # Delta metrics
        d_pnl = m_r.total_net_pnl - m_b.total_net_pnl
        d_sh = m_r.sharpe_ratio - m_b.sharpe_ratio
        # For drawdown, a reduction in DD is positive improvement
        d_dd = m_b.max_drawdown_pct - m_r.max_drawdown_pct

        delta_pnls.append(d_pnl)
        delta_shs.append(d_sh)
        delta_dds.append(d_dd)

    d_pnl_arr = np.array(delta_pnls)
    d_sh_arr = np.array(delta_shs)

    p_sup_pnl = float(np.mean(d_pnl_arr > 0))
    p_sup_sh = float(np.mean(d_sh_arr > 0))
    p_val_pnl = float(np.mean(d_pnl_arr <= 0))

    dist_pnl = _build_dist(delta_pnls)
    is_sig = (p_val_pnl < 0.05) and (dist_pnl.ci_lower_95 > 0)

    return GridBootstrapReport(
        iterations=iterations,
        block_size=block_size,
        baseline_return=_build_dist(base_rets),
        baseline_max_dd=_build_dist(base_dds),
        baseline_sharpe=_build_dist(base_shs),
        radar_return=_build_dist(radar_rets),
        radar_max_dd=_build_dist(radar_dds),
        radar_sharpe=_build_dist(radar_shs),
        delta_net_pnl=dist_pnl,
        delta_sharpe=_build_dist(delta_shs),
        delta_max_dd=_build_dist(delta_dds),
        p_superior_pnl=p_sup_pnl,
        p_superior_sharpe=p_sup_sh,
        p_value_superiority=p_val_pnl,
        is_statistically_significant=is_sig,
    )
