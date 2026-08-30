"""Walk-Forward Analysis for DVOL Regime Factor Rule."""

import asyncio
import itertools
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
from dateutil.relativedelta import relativedelta

from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.config.loader import ConfigLoader
from neon_radar.config.models import TimeFrame
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.models import Symbol

sys.path.append(str(Path(__file__).parent))
from dvol_regime_rule import DvolRegimeFactorRule

from neon_radar.infrastructure.exchanges.binance import BinanceClient
from neon_radar.infrastructure.storage.intelligence_store import HistoricalIntelligenceStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_trades(trades):
    from neon_radar.application.services.trade_analyzer import TradeAnalyzer
    analyzer = TradeAnalyzer()
    return analyzer.analyze(trades)

async def run_optimization():
    config = ConfigLoader(Path("config.json")).load()
    import json

    from neon_radar.config.loader import _strip_meta
    raw = json.loads(Path("scoring_rules.json").read_text(encoding="utf-8"))
    scoring_config = ScoringRulesConfig.model_validate(_strip_meta(raw))

    # We must ensure regime_filter is enabled for the DVOL rule to see the regime
    regime_dict = dict(scoring_config.regime_filter)
    regime_dict["enabled"] = True
    scoring_config = scoring_config.model_copy(update={"regime_filter": regime_dict})

    # Load the base rules directly
    from neon_radar.config.scoring_loader import load_rules
    base_rules = tuple(load_rules(Path("scoring_rules.json")))

    binance_client = BinanceClient(config.api)
    intelligence_store = HistoricalIntelligenceStore(Path("data/intelligence"))

    # Instantiate Baseline BT (With configured rules)
    baseline_bt = TradeBacktester(
        exchange=binance_client,
        scoring_config=scoring_config,
        rules=base_rules,
        intelligence_store=intelligence_store
    )

    symbol = Symbol("BTCUSDT")
    timeframe = TimeFrame("1d")

    # We need data from a bit earlier to prime indicators
    start_date = date(2021, 1, 1)
    end_date = date(2026, 8, 1)

    logger.info("Prefetching market data...")
    await baseline_bt._prefetch([symbol], timeframe, start_date, end_date)

    # WFA Configuration
    is_length = relativedelta(months=6)
    oos_length = relativedelta(months=2)
    current_start = date(2023, 1, 1)  # Ensure enough MI history

    # Parameter Grid
    behaviors = ["directional", "confidence", "filter"]
    thresholds = [1.0, 1.5, 2.0]

    stitched_oos_trades_baseline = []
    stitched_oos_trades_dvol = []
    oos_results = []

    window_idx = 1

    while True:
        is_end = current_start + is_length
        oos_end = is_end + oos_length

        if oos_end > end_date:
            break

        logger.info(f"--- WFA Window {window_idx} ---")
        logger.info(f"IS: {current_start} to {is_end}")
        logger.info(f"OOS: {is_end} to {oos_end}")

        # 1. In-Sample Optimization
        best_param = None
        best_expectancy = -float('inf')

        for behavior, threshold in itertools.product(behaviors, thresholds):
            rule = DvolRegimeFactorRule(
                name="dvol_regime",
                behavior=behavior,
                z_score_threshold=threshold,
                confidence_boost=0.8,
                confidence_penalty=0.1
            )

            test_bt = TradeBacktester(
                exchange=binance_client,
                scoring_config=scoring_config,
                rules=base_rules + (rule,),
                intelligence_store=intelligence_store
            )
            test_bt._series_cache = baseline_bt._series_cache
            test_bt._context_cache = baseline_bt._context_cache
            test_bt._raw_mi_map = baseline_bt._raw_mi_map

            trades = await test_bt.run(
                start_date=current_start,
                end_date=is_end,
                symbols=[symbol],
                timeframe=timeframe,
                min_history_candles=50
            )

            report = evaluate_trades(trades)
            exp = report.net_expectancy

            if exp > best_expectancy:
                best_expectancy = exp
                best_param = (behavior, threshold)

        if not best_param:
            logger.warning("No profitable params found in IS. Using default/neutral for OOS.")
            best_param = ("confidence", 1.5)

        logger.info(f"Optimal IS Params: Behavior={best_param[0]}, Threshold={best_param[1]} (Exp: {best_expectancy:.4f})")

        # 2. Out-of-Sample Verification
        best_rule = DvolRegimeFactorRule(
            name="dvol_regime",
            behavior=best_param[0],
            z_score_threshold=best_param[1],
            confidence_boost=0.8,
            confidence_penalty=0.1
        )

        oos_bt = TradeBacktester(
            exchange=binance_client,
            scoring_config=scoring_config,
            rules=base_rules + (best_rule,),
            intelligence_store=intelligence_store
        )
        oos_bt._series_cache = baseline_bt._series_cache
        oos_bt._context_cache = baseline_bt._context_cache
        oos_bt._raw_mi_map = baseline_bt._raw_mi_map

        oos_trades_dvol = await oos_bt.run(
            start_date=is_end,
            end_date=oos_end,
            symbols=[symbol],
            timeframe=timeframe,
            min_history_candles=50
        )

        oos_trades_base = await baseline_bt.run(
            start_date=is_end,
            end_date=oos_end,
            symbols=[symbol],
            timeframe=timeframe,
            min_history_candles=50
        )

        stitched_oos_trades_dvol.extend(oos_trades_dvol)
        stitched_oos_trades_baseline.extend(oos_trades_base)

        dvol_rep = evaluate_trades(oos_trades_dvol)
        base_rep = evaluate_trades(oos_trades_base)

        oos_results.append({
            "window": window_idx,
            "dvol_pf": dvol_rep.net_profit_factor,
            "base_pf": base_rep.net_profit_factor,
            "dvol_exp": dvol_rep.net_expectancy,
            "trades": dvol_rep.total_trades
        })

        logger.info(f"OOS Result - DVOL PF: {dvol_rep.net_profit_factor:.2f}, Base PF: {base_rep.net_profit_factor:.2f}, Trades: {dvol_rep.total_trades}")

        current_start += oos_length
        window_idx += 1

    # Final Analysis of Stitched OOS
    logger.info("="*60)
    logger.info("STITCHED OOS RESULTS")
    logger.info("="*60)

    base_final = evaluate_trades(stitched_oos_trades_baseline)
    dvol_final = evaluate_trades(stitched_oos_trades_dvol)

    logger.info(f"Baseline PF: {base_final.net_profit_factor:.2f}")
    logger.info(f"DVOL Rule PF: {dvol_final.net_profit_factor:.2f}")
    logger.info(f"Baseline Expectancy: {base_final.net_expectancy:.4f}")
    logger.info(f"DVOL Rule Expectancy: {dvol_final.net_expectancy:.4f}")
    logger.info(f"Baseline Sharpe: {base_final.net_sharpe_ratio:.2f}")
    logger.info(f"DVOL Rule Sharpe: {dvol_final.net_sharpe_ratio:.2f}")
    logger.info(f"Baseline Max DD: {base_final.max_drawdown_pct:.2%}")
    logger.info(f"DVOL Rule Max DD: {dvol_final.max_drawdown_pct:.2%}")

    total_trades = dvol_final.total_trades
    logger.info(f"Total DVOL Rule Trades in OOS: {total_trades}")
    if total_trades < 10:
        logger.warning("Insufficient evidence: Total trades < 10")

    # Bootstrap
    logger.info("\nRunning Bootstrap on Stitched DVOL Trades...")
    if total_trades >= 5:
        # Actually expectancy is mean of net_profit / position_size. We'll bootstrap mean of net_profit %
        # A simple estimation:
        pct_returns = [(t.exit_price - t.entry_price)/t.entry_price if t.direction.value == 1 else (t.entry_price - t.exit_price)/t.entry_price for t in stitched_oos_trades_dvol]

        boot_means = []
        for _ in range(1000):
            sample = np.random.choice(pct_returns, size=len(pct_returns), replace=True)
            boot_means.append(np.mean(sample))

        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)
        logger.info(f"Bootstrap 95% CI for Expectancy: [{ci_lower:.4f}, {ci_upper:.4f}]")
    else:
        logger.info("Not enough trades to bootstrap.")

if __name__ == "__main__":
    asyncio.run(run_optimization())
