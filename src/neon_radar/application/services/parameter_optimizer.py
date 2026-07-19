"""Parameter optimization for Walk-Forward Analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from neon_radar.application.services.trade_analyzer import TradeAnalyzer
from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date

    from neon_radar.config.scoring_models import ScoringRulesConfig
    from neon_radar.domain.models import Symbol
    from neon_radar.domain.trading.backtest import BacktestReport

logger = get_logger(__name__)


class ParameterOptimizer(Protocol):
    """Protocol for a walk-forward parameter optimizer.
    
    Given a Backtester (which holds the cache, exchange, base config),
    finds the best ScoringRulesConfig on the given historical period.
    """

    async def optimize(
        self,
        base_backtester: TradeBacktester,
        start_date: date,
        end_date: date,
        symbols: Iterable[Symbol],
        timeframe: str,
    ) -> tuple[ScoringRulesConfig, BacktestReport]:
        """Run optimization and return the best config and its IS report."""
        ...


class GridSearchOptimizer:
    """Basic grid search optimizer for V1.
    
    Optimizes ONLY the `min_confidence` parameter to isolate its effect
    and prevent curve fitting over multiple dimensions.
    """

    def __init__(self, min_confidence_grid: list[float] | None = None) -> None:
        if min_confidence_grid is None:
            self.min_confidence_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        else:
            self.min_confidence_grid = min_confidence_grid

        self._analyzer = TradeAnalyzer()

    async def optimize(
        self,
        base_backtester: TradeBacktester,
        start_date: date,
        end_date: date,
        symbols: Iterable[Symbol],
        timeframe: str,
    ) -> tuple[ScoringRulesConfig, BacktestReport]:
        symbols = tuple(symbols)
        best_config: ScoringRulesConfig | None = None
        best_report: BacktestReport | None = None

        # User defined priority: Max Expectancy -> Max Profit Factor -> Max min_confidence (most conservative)
        best_expectancy = float("-inf")
        best_profit_factor = float("-inf")
        best_confidence = -1.0

        for conf in self.min_confidence_grid:
            new_cfg = base_backtester._scoring_config.model_copy(update={"min_confidence": conf})

            # Recreate backtester with the new config but sharing the cache and rules
            # We don't rebuild rules because their parameters are unchanged.
            tester = TradeBacktester(
                exchange=base_backtester._exchange,
                scoring_config=new_cfg,
                rules=base_backtester._rules,
                funding_provider=base_backtester._funding_provider,
                preloaded_series=base_backtester.cache,
                cost_model=base_backtester._cost_model,
            )

            trades = await tester.run(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                timeframe=timeframe,
            )

            report = self._analyzer.analyze(trades)

            # Selection logic
            is_better = False
            if report.net_expectancy > best_expectancy:
                is_better = True
            elif abs(report.net_expectancy - best_expectancy) < 1e-6:
                if report.net_profit_factor > best_profit_factor:
                    is_better = True
                elif abs(report.net_profit_factor - best_profit_factor) < 1e-6:
                    if conf > best_confidence:
                        is_better = True

            if is_better or best_config is None:
                best_expectancy = report.net_expectancy
                best_profit_factor = report.net_profit_factor
                best_confidence = conf
                best_config = new_cfg
                best_report = report

        assert best_config is not None and best_report is not None
        return best_config, best_report
