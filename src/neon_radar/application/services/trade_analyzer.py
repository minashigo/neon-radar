"""Service for analyzing backtest results."""

from collections.abc import Iterable

from neon_radar.domain.trading.backtest import BacktestReport, StatisticalValidationReport, Trade


class TradeAnalyzer:
    """Analyzes a series of trades and generates a BacktestReport.

    This service encapsulates all the mathematical formulas for trading
    metrics, keeping the simulation engine and domain models clean.
    """

    def analyze(self, trades_iter: Iterable[Trade]) -> BacktestReport:
        """Compute metrics from an iterable of completed trades."""
        trades = tuple(trades_iter)
        total = len(trades)

        if total == 0:
            return BacktestReport(
                total_trades=0,
                win_rate=0.0,
                wins=0,
                losses=0,
                avg_win_pct=0.0,
                avg_loss_pct=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                avg_holding_time_ms=0.0,
                validation=StatisticalValidationReport(False, 1.0, 0.0, 0.0, 0.0, 1.0),
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
        loss_rate = n_losses / total

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        max_cons_wins = 0
        max_cons_losses = 0
        curr_cons_wins = 0
        curr_cons_losses = 0

        total_holding_time = 0.0
        closed_trades_count = 0

        for t in trades:
            if t.pnl_pct > 0:
                curr_cons_wins += 1
                curr_cons_losses = 0
                if curr_cons_wins > max_cons_wins:
                    max_cons_wins = curr_cons_wins
            elif t.pnl_pct < 0:
                curr_cons_losses += 1
                curr_cons_wins = 0
                if curr_cons_losses > max_cons_losses:
                    max_cons_losses = curr_cons_losses
            else:
                # Break-even stops streaks
                curr_cons_wins = 0
                curr_cons_losses = 0

            if t.exit_time is not None:
                total_holding_time += t.exit_time - t.entry_time
                closed_trades_count += 1

        avg_holding_time = (
            total_holding_time / closed_trades_count if closed_trades_count > 0 else 0.0
        )

        validation = self.calculate_statistical_validation(trades)

        return BacktestReport(
            total_trades=total,
            win_rate=win_rate,
            wins=n_wins,
            losses=n_losses,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_consecutive_wins=max_cons_wins,
            max_consecutive_losses=max_cons_losses,
            avg_holding_time_ms=avg_holding_time,
            validation=validation,
            trades=trades,
        )

    def calculate_statistical_validation(
        self, trades: tuple[Trade, ...], mc_simulations: int = 10_000
    ) -> StatisticalValidationReport:
        """Calculate statistical significance and Monte Carlo bootstrap metrics."""
        import math

        import numpy as np

        from neon_radar.domain.trading.backtest import StatisticalValidationReport

        if len(trades) < 2:
            return StatisticalValidationReport(False, 1.0, 0.0, 0.0, 0.0, 1.0)

        pnls = np.array([t.pnl_pct for t in trades], dtype=float)
        n = len(pnls)
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)

        # 1. T-Test (Statistical Significance)
        # We use Normal approximation for the p-value.
        if std_pnl > 0:
            t_statistic = mean_pnl / (std_pnl / math.sqrt(n))
        else:
            t_statistic = 0.0 if mean_pnl == 0 else (float("inf") if mean_pnl > 0 else float("-inf"))

        if math.isinf(t_statistic):
            p_value = 0.0 if t_statistic > 0 else 1.0
        else:
            # One-tailed normal approximation: P(Z > t)
            # CDF of standard normal: 0.5 * (1 + erf(x / sqrt(2)))
            # We want 1 - CDF
            p_value = 0.5 * (1.0 - math.erf(t_statistic / math.sqrt(2.0)))

        # 2. Monte Carlo Bootstrap
        # We resample the PnL array `mc_simulations` times with replacement.
        # Each simulation is an array of size `n`.
        # To save memory and time, we can generate a matrix of indices:
        # shape: (mc_simulations, n)
        # But for large n or mc_simulations, we can do it in batches or just let numpy handle it.
        # Numpy `random.choice` is fast.
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        resampled_indices = rng.integers(0, n, size=(mc_simulations, n))
        resampled_pnls = pnls[resampled_indices]

        # Calculate expectancy (mean) for each simulation
        sim_expectancies = np.mean(resampled_pnls, axis=1)

        # 95% Confidence Interval (2.5th and 97.5th percentiles)
        ci_lower = float(np.percentile(sim_expectancies, 2.5))
        ci_upper = float(np.percentile(sim_expectancies, 97.5))

        # Probability of loss (fraction of simulations with expectancy < 0)
        prob_loss = float(np.sum(sim_expectancies < 0) / mc_simulations)

        return StatisticalValidationReport(
            is_valid=True,
            p_value=float(p_value),
            t_statistic=float(t_statistic),
            mc_expectancy_95_ci_lower=ci_lower,
            mc_expectancy_95_ci_upper=ci_upper,
            mc_probability_of_loss=prob_loss,
        )
