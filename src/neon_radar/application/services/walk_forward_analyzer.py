"""Walk-Forward Analysis service."""

from __future__ import annotations

import calendar
from datetime import date
from typing import TYPE_CHECKING

from neon_radar.application.services.trade_analyzer import TradeAnalyzer
from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.domain.trading.walk_forward import WalkForwardCycle, WalkForwardReport
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neon_radar.application.services.parameter_optimizer import ParameterOptimizer
    from neon_radar.domain.models import Symbol

logger = get_logger(__name__)


def _add_months(d: date, months: int) -> date:
    """Safely add months to a date."""
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


class WalkForwardAnalyzer:
    """Coordinates the rolling Walk-Forward Analysis over historical data.
    
    For each step:
      1. Define IS (In-Sample) and OOS (Out-of-Sample) windows.
      2. Ask ParameterOptimizer to find the best configuration on the IS window.
      3. Evaluate the OOS window using the chosen configuration.
      4. Step forward.
    """

    def __init__(
        self,
        optimizer: ParameterOptimizer,
    ) -> None:
        self._optimizer = optimizer
        self._analyzer = TradeAnalyzer()

    async def run(
        self,
        base_backtester: TradeBacktester,
        start_date: date,
        end_date: date,
        symbols: Iterable[Symbol],
        timeframe: str,
        is_window_months: int = 12,
        oos_window_months: int = 3,
        step_months: int = 3,
    ) -> WalkForwardReport:
        symbols = tuple(symbols)
        cycles: list[WalkForwardCycle] = []

        # Calculate total windows
        total_cycles = 0
        tmp_is_start = start_date
        while True:
            tmp_is_end = _add_months(tmp_is_start, is_window_months)
            tmp_oos_start = tmp_is_end
            tmp_oos_end = _add_months(tmp_oos_start, oos_window_months)
            if tmp_oos_start >= end_date:
                break
            total_cycles += 1
            tmp_is_start = _add_months(tmp_is_start, step_months)

        logger.info(f"Starting WFA with {total_cycles} total windows...")

        # Generate windows
        current_is_start = start_date

        # We need to prefetch data for the ENTIRE range first so that optimizing is fast.
        logger.info(f"Prefetching data from {start_date} to {end_date} for Walk-Forward Analysis...")
        await base_backtester._prefetch(symbols, timeframe, start_date, end_date)

        cycle_idx = 1
        while True:
            current_is_end = _add_months(current_is_start, is_window_months)
            current_oos_start = current_is_end
            current_oos_end = _add_months(current_oos_start, oos_window_months)

            if current_oos_start >= end_date:
                break

            if current_oos_end > end_date:
                current_oos_end = end_date

            logger.info(
                f"WFA Cycle {cycle_idx}/{total_cycles}: IS [{current_is_start} to {current_is_end}] -> "
                f"OOS [{current_oos_start} to {current_oos_end}]"
            )

            # 1. Optimize on IS
            best_config, is_report = await self._optimizer.optimize(
                base_backtester=base_backtester,
                start_date=current_is_start,
                end_date=current_is_end,
                symbols=symbols,
                timeframe=timeframe,
            )

            logger.info(f"  Selected min_confidence: {best_config.min_confidence:.2f} "
                        f"(IS Expectancy: {is_report.net_expectancy:.2%}, IS Profit Factor: {is_report.net_profit_factor:.2f})")

            # 2. Evaluate on OOS with the best config
            oos_tester = TradeBacktester(
                exchange=base_backtester._exchange,
                scoring_config=best_config,
                rules=base_backtester._rules,
                funding_provider=base_backtester._funding_provider,
                preloaded_series=base_backtester.cache,
                cost_model=base_backtester._cost_model,
            )

            oos_trades = await oos_tester.run(
                start_date=current_oos_start,
                end_date=current_oos_end,
                symbols=symbols,
                timeframe=timeframe,
            )

            oos_report = self._analyzer.analyze(oos_trades)

            logger.info(f"  OOS Result: Expectancy {oos_report.net_expectancy:.2%}, "
                        f"Profit Factor {oos_report.net_profit_factor:.2f}")

            # 3. Save Cycle
            cycles.append(
                WalkForwardCycle(
                    is_start=current_is_start,
                    is_end=current_is_end,
                    oos_start=current_oos_start,
                    oos_end=current_oos_end,
                    optimized_config=best_config,
                    is_report=is_report,
                    oos_report=oos_report,
                )
            )

            # Step forward
            current_is_start = _add_months(current_is_start, step_months)
            cycle_idx += 1

        logger.info(f"Walk-Forward Analysis complete. Generated {len(cycles)} cycles.")
        return WalkForwardReport(cycles=tuple(cycles))
