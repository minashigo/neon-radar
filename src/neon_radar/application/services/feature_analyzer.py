"""Service for conducting Ablation Analysis on trading rules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.domain.trading.feature_importance import (
    FeatureImportanceMetrics,
    FeatureImportanceReport,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date

    from neon_radar.application.services.trade_analyzer import TradeAnalyzer
    from neon_radar.config.models import TimeFrame
    from neon_radar.domain.models import Symbol

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Orchestrates ablation analysis to evaluate feature importance."""

    # Weights for the Feature Score (total = 100)
    W_PROFIT_FACTOR = 0.35
    W_EXPECTANCY = 0.30
    W_SHARPE_RATIO = 0.20
    W_WIN_RATE = 0.10
    W_PROBABILITY_OF_LOSS = 0.05

    def __init__(self, analyzer: TradeAnalyzer) -> None:
        self._analyzer = analyzer

    async def analyze(
        self,
        baseline_tester: TradeBacktester,
        start_date: date,
        end_date: date,
        symbols: Iterable[Symbol],
        timeframe: TimeFrame = "1d",
        min_history_candles: int = 50,
    ) -> FeatureImportanceReport:
        """Run full ablation analysis over the given period."""
        symbols = tuple(symbols)
        logger.info("Starting baseline backtest for Feature Analysis...")

        # 1. Run Baseline
        baseline_trades = await baseline_tester.run(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            timeframe=timeframe,
            min_history_candles=min_history_candles,
        )
        baseline_report = self._analyzer.analyze(baseline_trades)

        b_pf = baseline_report.net_profit_factor
        b_exp = baseline_report.net_expectancy
        b_sharpe = baseline_report.net_sharpe_ratio
        b_wr = baseline_report.win_rate
        b_prob_loss = baseline_report.validation.mc_probability_of_loss if baseline_report.validation else 1.0
        b_pval = baseline_report.validation.p_value if baseline_report.validation else 1.0

        # We need the base components to instantiate ablated testers
        exchange = baseline_tester._exchange
        scoring_config = baseline_tester._scoring_config
        all_rules = baseline_tester._rules
        cache = baseline_tester.cache

        features_metrics: list[FeatureImportanceMetrics] = []

        logger.info(f"Running ablation for {len(all_rules)} rules...")

        # 2. Run Ablation for each rule
        for rule in all_rules:
            logger.info(f"  Ablating rule: {rule.name}")
            ablated_rules = tuple(r for r in all_rules if r != rule)

            ablated_tester = TradeBacktester(
                exchange=exchange,
                scoring_config=scoring_config,
                rules=ablated_rules,
                preloaded_series=cache,  # Re-use pre-fetched data
            )

            ablated_trades = await ablated_tester.run(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                timeframe=timeframe,
                min_history_candles=min_history_candles,
            )

            ablated_report = self._analyzer.analyze(ablated_trades)

            a_pf = ablated_report.net_profit_factor
            a_exp = ablated_report.net_expectancy
            a_sharpe = ablated_report.net_sharpe_ratio
            a_wr = ablated_report.win_rate
            a_prob_loss = ablated_report.validation.mc_probability_of_loss if ablated_report.validation else 1.0
            a_pval = ablated_report.validation.p_value if ablated_report.validation else 1.0

            # Calculate Deltas (Positive means removing the rule made it worse, so the rule is good)
            delta_pf = b_pf - a_pf
            delta_exp = b_exp - a_exp
            delta_sharpe = b_sharpe - a_sharpe
            delta_wr = b_wr - a_wr

            # For p_value and prob_loss, lower is better. So if Baseline is better (lower),
            # Ablated is higher. Therefore Ablated - Baseline is positive when rule is good.
            delta_prob_loss = a_prob_loss - b_prob_loss
            delta_pval = a_pval - b_pval

            # Calculate composite Feature Score
            # To normalize somewhat, we map them based on typical scales.
            # Delta PF is absolute. Delta WR and Exp are percentages.
            # Example heuristic normalization for the score:
            pf_score = delta_pf * 1.0             # Delta of 0.1 -> 0.1
            exp_score = delta_exp * 10.0          # Delta of 1% (0.01) -> 0.1
            sharpe_score = delta_sharpe * 0.2     # Delta of 0.5 -> 0.1
            wr_score = delta_wr * 2.0             # Delta of 5% (0.05) -> 0.1
            prob_loss_score = delta_prob_loss * 1.0 # Delta of 10% (0.1) -> 0.1

            score = (
                self.W_PROFIT_FACTOR * pf_score +
                self.W_EXPECTANCY * exp_score +
                self.W_SHARPE_RATIO * sharpe_score +
                self.W_WIN_RATE * wr_score +
                self.W_PROBABILITY_OF_LOSS * prob_loss_score
            )

            metrics = FeatureImportanceMetrics(
                rule_name=rule.name,
                delta_profit_factor=delta_pf,
                delta_expectancy=delta_exp,
                delta_sharpe_ratio=delta_sharpe,
                delta_win_rate=delta_wr,
                delta_probability_of_loss=delta_prob_loss,
                delta_p_value=delta_pval,
                feature_score=score,
            )
            features_metrics.append(metrics)

        # Sort by feature score descending
        features_metrics.sort(key=lambda x: x.feature_score, reverse=True)

        return FeatureImportanceReport(
            baseline=baseline_report,
            features=tuple(features_metrics)
        )
