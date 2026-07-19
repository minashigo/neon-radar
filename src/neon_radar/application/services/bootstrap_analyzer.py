"""Block Bootstrap Validation service."""

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from neon_radar.domain.trading.backtest import BacktestReport, Trade
from neon_radar.domain.trading.bootstrap import BootstrapMetricDistribution, BootstrapReport


class BootstrapAnalyzer:
    """Analyzes statistical stability of a trading strategy using Block Bootstrap.

    Instead of recalculating metrics, it uses an injected calculator (like TradeAnalyzer)
    to compute metrics for each bootstrapped sequence of trades.
    """

    def __init__(self, metric_calculator: Callable[[Sequence[Trade]], Any]) -> None:
        """Initialize with a metric calculator function.
        
        Args:
            metric_calculator: A function that takes a sequence of Trades and returns
                an object that has properties for the metrics we want to track.
                Typically, this is `TradeAnalyzer().analyze`.
        """
        self._metric_calculator = metric_calculator

    def run(
        self,
        trades: Sequence[Trade],
        block_size: int = 20,
        iterations: int = 1000,
        random_seed: int = 42,
    ) -> BootstrapReport | None:
        """Run Block Bootstrap analysis on a sequence of trades."""
        total_trades = len(trades)
        if total_trades == 0:
            return None

        # Fix the seed for reproducibility using standard library random
        rng = random.Random(random_seed)

        # Pre-allocate lists for each metric
        metric_values: dict[str, list[float]] = {
            "Profit Factor": [],
            "Expectancy (%)": [],
            "Win Rate (%)": [],
            "Sharpe Ratio": [],
            "Max Drawdown (%)": [],
        }

        # Calculate number of blocks needed to match the original sample size
        num_blocks = math.ceil(total_trades / block_size)

        for _ in range(iterations):
            # Sample blocks with replacement
            sampled_trades: list[Trade] = []
            for _ in range(num_blocks):
                # Pick a random starting index for the block
                # Max start index ensures we don't go out of bounds
                max_start_idx = max(0, total_trades - block_size)
                start_idx = rng.randint(0, max_start_idx)

                # Extract the block and add it to our sample
                block = trades[start_idx : start_idx + block_size]
                sampled_trades.extend(block)

            # Trim the sampled trades to exactly match the original length
            sampled_trades = sampled_trades[:total_trades]

            # Compute metrics for this bootstrapped sample using the injected calculator
            report = self._metric_calculator(sampled_trades)

            # We assume the calculator returns an object matching BacktestReport interface
            if isinstance(report, BacktestReport):
                pf = report.net_profit_factor
                exp = report.net_expectancy * 100
                wr = report.win_rate * 100
                sharpe = report.net_sharpe_ratio
                dd = report.max_drawdown_pct * 100
            else:
                # If a different calculator is passed, it needs to provide these attributes
                # We use getattr with fallback to handle generic ducks
                pf = getattr(report, "net_profit_factor", 0.0)
                exp = getattr(report, "net_expectancy", 0.0) * 100
                wr = getattr(report, "win_rate", 0.0) * 100
                sharpe = getattr(report, "net_sharpe_ratio", 0.0)
                dd = getattr(report, "max_drawdown_pct", 0.0) * 100

            metric_values["Profit Factor"].append(pf)
            metric_values["Expectancy (%)"].append(exp)
            metric_values["Win Rate (%)"].append(wr)
            metric_values["Sharpe Ratio"].append(sharpe)
            metric_values["Max Drawdown (%)"].append(dd)

        # Compute summary statistics for each metric
        distributions = {}
        for name, values in metric_values.items():
            arr = np.array(values, dtype=float)

            # Handle inf values which might appear in Profit Factor
            arr_clean = arr[~np.isinf(arr)]
            if len(arr_clean) == 0:
                arr_clean = np.array([0.0])

            distributions[name] = BootstrapMetricDistribution(
                mean=float(np.mean(arr_clean)),
                median=float(np.median(arr_clean)),
                std_dev=float(np.std(arr_clean, ddof=1) if len(arr_clean) > 1 else 0.0),
                min_val=float(np.min(arr_clean)),
                max_val=float(np.max(arr_clean)),
                ci_lower_95=float(np.percentile(arr_clean, 2.5)),
                ci_upper_95=float(np.percentile(arr_clean, 97.5)),
                values=tuple(float(v) for v in values),
            )

        return BootstrapReport(
            iterations=iterations,
            block_size=block_size,
            metrics=distributions,
        )
