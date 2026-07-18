"""Service for analyzing backtest results."""

import math
from collections.abc import Iterable

import numpy as np

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
                gross_avg_win_pct=0.0,
                gross_avg_loss_pct=0.0,
                gross_profit_factor=0.0,
                gross_expectancy=0.0,
                net_profit_pct=0.0,
                net_avg_win_pct=0.0,
                net_avg_loss_pct=0.0,
                net_profit_factor=0.0,
                net_expectancy=0.0,
                net_sharpe_ratio=0.0,
                avg_trade_cost_pct=0.0,
                avg_slippage_pct=0.0,
                total_fees_pct=0.0,
                total_funding_pct=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                avg_holding_time_ms=0.0,
                validation=StatisticalValidationReport(False, 1.0, 0.0, 0.0, 0.0, 1.0),
                trades=(),
            )

        # Net is the primary determiner for Win/Loss state now
        # but to keep `wins` consistent with old definition, we can use net_pnl_pct
        wins = [t for t in trades if t.net_pnl_pct > 0]
        losses = [t for t in trades if t.net_pnl_pct < 0]

        n_wins = len(wins)
        n_losses = len(losses)

        # Gross metrics
        gross_wins = [t for t in trades if t.gross_pnl_pct > 0]
        gross_losses = [t for t in trades if t.gross_pnl_pct < 0]
        sum_gross_wins = sum(t.gross_pnl_pct for t in gross_wins)
        sum_gross_losses = sum(abs(t.gross_pnl_pct) for t in gross_losses)
        gross_avg_win = sum_gross_wins / len(gross_wins) if gross_wins else 0.0
        gross_avg_loss = sum_gross_losses / len(gross_losses) if gross_losses else 0.0
        gross_profit_factor = sum_gross_wins / sum_gross_losses if sum_gross_losses > 0 else float("inf")
        if len(gross_losses) == 0 and len(gross_wins) == 0:
            gross_profit_factor = 0.0
        gross_expectancy = (sum_gross_wins - sum_gross_losses) / total if total > 0 else 0.0

        # Net metrics
        sum_net_wins = sum(t.net_pnl_pct for t in wins)
        sum_net_losses = sum(abs(t.net_pnl_pct) for t in losses)
        net_avg_win = sum_net_wins / n_wins if n_wins > 0 else 0.0
        net_avg_loss = sum_net_losses / n_losses if n_losses > 0 else 0.0
        net_profit_factor = sum_net_wins / sum_net_losses if sum_net_losses > 0 else float("inf")
        if n_losses == 0 and n_wins == 0:
            net_profit_factor = 0.0

        win_rate = n_wins / total
        loss_rate = n_losses / total
        net_expectancy = (win_rate * net_avg_win) - (loss_rate * net_avg_loss)
        net_profit_pct = sum(t.net_pnl_pct for t in trades)

        # Cost metrics
        total_costs = 0.0
        total_fees = 0.0
        total_slippage = 0.0
        total_funding = 0.0

        for t in trades:
            if t.costs:
                total_costs += t.costs.total_costs_pct
                total_fees += t.costs.fees_pct
                total_slippage += t.costs.slippage_pct
                total_funding += t.costs.funding_pct

        avg_trade_cost = total_costs / total
        avg_slippage = total_slippage / total

        # Consecutive stats & holding time
        max_cons_wins = 0
        max_cons_losses = 0
        curr_cons_wins = 0
        curr_cons_losses = 0

        total_holding_time = 0.0
        closed_trades_count = 0

        for t in trades:
            if t.net_pnl_pct > 0:
                curr_cons_wins += 1
                curr_cons_losses = 0
                if curr_cons_wins > max_cons_wins:
                    max_cons_wins = curr_cons_wins
            elif t.net_pnl_pct < 0:
                curr_cons_losses += 1
                curr_cons_wins = 0
                if curr_cons_losses > max_cons_losses:
                    max_cons_losses = curr_cons_losses
            else:
                curr_cons_wins = 0
                curr_cons_losses = 0

            if t.exit_time is not None:
                total_holding_time += t.exit_time - t.entry_time
                closed_trades_count += 1

        avg_holding_time = (
            total_holding_time / closed_trades_count if closed_trades_count > 0 else 0.0
        )

        if closed_trades_count > 1:
            pnls = np.array([t.net_pnl_pct for t in trades if t.exit_time is not None], dtype=float)
            if len(pnls) > 1:
                mean_pnl = np.mean(pnls)
                std_pnl = np.std(pnls, ddof=1)
                net_sharpe_ratio = float(mean_pnl / std_pnl) if std_pnl > 0 else 0.0
            else:
                net_sharpe_ratio = 0.0
        else:
            net_sharpe_ratio = 0.0

        validation = self.calculate_statistical_validation(trades)

        return BacktestReport(
            total_trades=total,
            win_rate=win_rate,
            wins=n_wins,
            losses=n_losses,
            gross_avg_win_pct=gross_avg_win,
            gross_avg_loss_pct=gross_avg_loss,
            gross_profit_factor=gross_profit_factor,
            gross_expectancy=gross_expectancy,
            net_profit_pct=net_profit_pct,
            net_avg_win_pct=net_avg_win,
            net_avg_loss_pct=net_avg_loss,
            net_profit_factor=net_profit_factor,
            net_expectancy=net_expectancy,
            net_sharpe_ratio=net_sharpe_ratio,
            avg_trade_cost_pct=avg_trade_cost,
            avg_slippage_pct=avg_slippage,
            total_fees_pct=total_fees,
            total_funding_pct=total_funding,
            max_consecutive_wins=max_cons_wins,
            max_consecutive_losses=max_cons_losses,
            avg_holding_time_ms=avg_holding_time,
            validation=validation,
            trades=trades,
        )

    def calculate_statistical_validation(
        self, trades: tuple[Trade, ...], mc_simulations: int = 10_000
    ) -> StatisticalValidationReport:
        """Calculate statistical significance and Monte Carlo bootstrap metrics based on NET PnL."""
        if len(trades) < 2:
            return StatisticalValidationReport(False, 1.0, 0.0, 0.0, 0.0, 1.0)

        pnls = np.array([t.net_pnl_pct for t in trades], dtype=float)
        n = len(pnls)
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)

        if std_pnl > 0:
            t_statistic = mean_pnl / (std_pnl / math.sqrt(n))
        else:
            t_statistic = 0.0 if mean_pnl == 0 else (float("inf") if mean_pnl > 0 else float("-inf"))

        if math.isinf(t_statistic):
            p_value = 0.0 if t_statistic > 0 else 1.0
        else:
            p_value = 0.5 * (1.0 - math.erf(t_statistic / math.sqrt(2.0)))

        rng = np.random.default_rng(42)
        resampled_indices = rng.integers(0, n, size=(mc_simulations, n))
        resampled_pnls = pnls[resampled_indices]

        sim_expectancies = np.mean(resampled_pnls, axis=1)

        ci_lower = float(np.percentile(sim_expectancies, 2.5))
        ci_upper = float(np.percentile(sim_expectancies, 97.5))

        prob_loss = float(np.sum(sim_expectancies < 0) / mc_simulations)

        return StatisticalValidationReport(
            is_valid=True,
            p_value=float(p_value),
            t_statistic=float(t_statistic),
            mc_expectancy_95_ci_lower=ci_lower,
            mc_expectancy_95_ci_upper=ci_upper,
            mc_probability_of_loss=prob_loss,
        )
