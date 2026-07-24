import asyncio
import json
import logging
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any

from neon_radar.application.services.backtester import WalkForwardBacktester
from neon_radar.application.services.market_context.cache import ContextCache
from neon_radar.application.services.market_context.history_service import (
    MarketContextHistoryService,
)
from neon_radar.application.services.parameter_optimizer import GridSearchOptimizer
from neon_radar.application.services.trade_analyzer import TradeAnalyzer
from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.application.services.walk_forward_analyzer import WalkForwardAnalyzer
from neon_radar.config.scoring_loader import load_rules
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.models import Symbol
from neon_radar.infrastructure.exchanges.binance.client import BinanceClient
from neon_radar.infrastructure.exchanges.binance_transport import BinanceTransport
from neon_radar.infrastructure.providers.binance_context import BinanceContextProviders

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RC4_Validation")

# Spearman Rank Correlation
def spearmanr(x: list[float], y: list[float]) -> float:
    def rank(seq: list[float]) -> list[float]:
        sorted_seq = sorted(seq)
        return [sorted_seq.index(v) + 1 for v in seq]

    n = len(x)
    if n == 0:
        return 0.0
    rank_x = rank(x)
    rank_y = rank(y)

    # Pearson on ranks
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n
    num = sum((rx - mean_x) * (ry - mean_y) for rx, ry in zip(rank_x, rank_y, strict=True))
    den_x = sum((rx - mean_x) ** 2 for rx in rank_x)
    den_y = sum((ry - mean_y) ** 2 for ry in rank_y)
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / ((den_x * den_y) ** 0.5)

async def run_correlation(start_date: date, end_date: date, symbols: tuple[Symbol, ...], rules: tuple, exchange: BinanceClient, history_service: MarketContextHistoryService) -> dict:
    logger.info("Running Correlation Analyzer...")
    scoring_cfg = ScoringRulesConfig(rules=[])

    tester = WalkForwardBacktester(exchange=exchange, scoring_config=scoring_cfg, rules=rules, history_service=history_service)
    await tester._prefetch(symbols, "1d", start_date, end_date, (1,))
    evaluations = tester._evaluate_all(
        symbols=symbols,
        timeframe="1d",
        start_date=start_date,
        end_date=end_date,
        horizons=(1,),
        min_history_candles=50
    )

    from neon_radar.domain.scoring.backtest import BacktestConfig
    config = BacktestConfig(
        start_date=start_date, end_date=end_date, timeframe="1d",
        symbols=tuple(str(s) for s in symbols), horizons=(1,),
        min_confidence=scoring_cfg.min_confidence,
        confluence_bonus=scoring_cfg.confluence_bonus,
        confluence_penalty=scoring_cfg.confluence_penalty,
        max_confidence_boost=scoring_cfg.max_confidence_boost
    )
    result = tester._aggregate(config=config, evaluations=evaluations, horizons=(1,))

    if not result.correlation:
        return {}

    rule_names = result.correlation.rule_names

    # Extract raw signal values for Spearman & Direction
    signal_series: dict[str, list[float]] = {r: [] for r in rule_names}
    for e in evaluations:
        if e.horizon_days != 1:
            continue
        
        # Create a dict of just this evaluation's rule values
        val_map = {name: val for name, val in e.rule_values}
        for r_name in rule_names:
            signal_series[r_name].append(val_map.get(r_name, 0.0))

    correlation_data: list[dict[str, Any]] = []

    for r1, r2 in combinations(rule_names, 2):
        s1 = signal_series[r1]
        s2 = signal_series[r2]

        # Direction Agreement: % where both are non-zero and share same sign
        # Simultaneous: % where both are non-zero
        total_evals = len(s1)
        simultaneous = sum(1 for a, b in zip(s1, s2, strict=True) if a != 0 and b != 0)
        agreement = sum(1 for a, b in zip(s1, s2, strict=True) if a != 0 and b != 0 and (a > 0) == (b > 0))

        dir_agree = agreement / simultaneous if simultaneous > 0 else 0.0
        simult_pct = simultaneous / total_evals if total_evals > 0 else 0.0

        p = result.correlation.get(r1, r2)
        s = spearmanr(s1, s2)

        correlation_data.append({
            "rule_1": r1,
            "rule_2": r2,
            "pearson": p,
            "spearman": s,
            "direction_agreement": dir_agree,
            "simultaneous_pct": simult_pct
        })

    return {"matrix": correlation_data}

async def run_wfa(tester: TradeBacktester, start_date: date, end_date: date, symbols: tuple[Symbol, ...]) -> dict:
    optimizer = GridSearchOptimizer(min_confidence_grid=[0.3, 0.4, 0.5])
    wfa = WalkForwardAnalyzer(optimizer)
    report = await wfa.run(
        base_backtester=tester,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        timeframe="1d",
        is_window_months=6,
        oos_window_months=1,
        step_months=1
    )

    # Aggregate OOS trades
    all_oos_trades = []
    for cycle in report.cycles:
        all_oos_trades.extend(cycle.oos_report.trades)

    analyzer = TradeAnalyzer()
    if not all_oos_trades:
        return {"pf": 0.0, "exp": 0.0, "wr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "trade_count": 0}

    final_report = analyzer.analyze(tuple(all_oos_trades))
    return {
        "pf": final_report.net_profit_factor,
        "exp": final_report.net_expectancy,
        "wr": final_report.win_rate,
        "sharpe": final_report.net_sharpe_ratio,
        "max_dd": final_report.max_drawdown_pct,
        "trade_count": final_report.total_trades,
    }

async def run_ablation(start_date: date, end_date: date, symbols: tuple[Symbol, ...], rules: tuple, exchange: BinanceClient, history_service: MarketContextHistoryService) -> dict:
    logger.info("Running Feature Importance (Ablation) via WFA...")
    scoring_cfg = ScoringRulesConfig(rules=[])

    base_tester = TradeBacktester(
        exchange=exchange,
        scoring_config=scoring_cfg,
        rules=rules,
        history_service=history_service,
    )
    await base_tester._prefetch(symbols, "1d", start_date, end_date)

    baseline_metrics = await run_wfa(base_tester, start_date, end_date, symbols)

    ablation_results = []
    for rule in rules:
        logger.info(f"Ablating rule: {rule.name}")
        ablated_rules = tuple(r for r in rules if r != rule)
        ablated_tester = TradeBacktester(
            exchange=exchange,
            scoring_config=scoring_cfg,
            rules=ablated_rules,
            history_service=history_service,
            preloaded_series=base_tester.cache,
            preloaded_context=base_tester._context_cache,
        )
        ablated_metrics = await run_wfa(ablated_tester, start_date, end_date, symbols)

        ablation_results.append({
            "rule": rule.name,
            "delta_pf": baseline_metrics["pf"] - ablated_metrics["pf"],
            "delta_exp": baseline_metrics["exp"] - ablated_metrics["exp"],
            "delta_wr": baseline_metrics["wr"] - ablated_metrics["wr"],
            "delta_sharpe": baseline_metrics["sharpe"] - ablated_metrics["sharpe"],
            "delta_max_dd": baseline_metrics["max_dd"] - ablated_metrics["max_dd"],
        })

    return {"baseline": baseline_metrics, "features": ablation_results}

async def run_confluence_validation(start_date: date, end_date: date, symbols: tuple[Symbol, ...], rules: tuple, exchange: BinanceClient, history_service: MarketContextHistoryService) -> dict:
    logger.info("Running Confluence Validation...")

    cfg_with = ScoringRulesConfig(rules=[], confluence_bonus=0.20, confluence_penalty=0.15, max_confidence_boost=0.40)
    cfg_without = ScoringRulesConfig(rules=[], confluence_bonus=0.0, confluence_penalty=0.0, max_confidence_boost=0.0)

    tester_with = TradeBacktester(exchange=exchange, scoring_config=cfg_with, rules=rules, history_service=history_service)
    await tester_with._prefetch(symbols, "1d", start_date, end_date)

    metrics_with = await run_wfa(tester_with, start_date, end_date, symbols)

    tester_without = TradeBacktester(exchange=exchange, scoring_config=cfg_without, rules=rules, history_service=history_service, preloaded_series=tester_with.cache, preloaded_context=tester_with._context_cache)
    metrics_without = await run_wfa(tester_without, start_date, end_date, symbols)

    return {
        "with_confluence": metrics_with,
        "without_confluence": metrics_without
    }

async def run_sensitivity(start_date: date, end_date: date, symbols: tuple[Symbol, ...], rules: tuple, exchange: BinanceClient, history_service: MarketContextHistoryService) -> dict:
    logger.info("Running Sensitivity Analysis...")
    results = []

    bonuses = [0.10, 0.20, 0.30]
    penalties = [0.10, 0.15, 0.20]

    base_tester = TradeBacktester(exchange=exchange, scoring_config=ScoringRulesConfig(rules=[]), rules=rules, history_service=history_service)
    await base_tester._prefetch(symbols, "1d", start_date, end_date)

    for bonus in bonuses:
        for penalty in penalties:
            cfg = ScoringRulesConfig(rules=[], confluence_bonus=bonus, confluence_penalty=penalty, max_confidence_boost=bonus * 2)
            tester = TradeBacktester(exchange=exchange, scoring_config=cfg, rules=rules, history_service=history_service, preloaded_series=base_tester.cache, preloaded_context=base_tester._context_cache)
            metrics = await run_wfa(tester, start_date, end_date, symbols)
            results.append({
                "bonus": bonus,
                "penalty": penalty,
                "pf": metrics["pf"],
                "exp": metrics["exp"],
                "sharpe": metrics["sharpe"]
            })

    return {"grid": results}

async def main():
    start_date = date(2023, 7, 1)
    end_date = date(2024, 7, 1) # 12 months
    symbols = (Symbol("BTCUSDT"), Symbol("ETHUSDT"), Symbol("SOLUSDT"))

    rules = load_rules(Path("config/scoring.example.json"))

    from neon_radar.config.loader import load_config
    cfg = load_config()

    async with BinanceClient(cfg.api) as exchange:
        transport = BinanceTransport(base_url=cfg.api.base_url, rate_limit_per_minute=cfg.api.rate_limit_per_minute)
        try:
            cache = ContextCache(directory=Path(".cache/context"))
            providers = [
                BinanceContextProviders(transport, cache)
            ]
            history_service = MarketContextHistoryService(providers)

            correlation = await run_correlation(start_date, end_date, symbols, rules, exchange, history_service)
            ablation = await run_ablation(start_date, end_date, symbols, rules, exchange, history_service)
            confluence = await run_confluence_validation(start_date, end_date, symbols, rules, exchange, history_service)
            sensitivity = await run_sensitivity(start_date, end_date, symbols, rules, exchange, history_service)

            final_data = {
                "correlation": correlation,
                "ablation": ablation,
                "confluence": confluence,
                "sensitivity": sensitivity
            }

            output_path = r"C:\Users\orphan\.gemini\antigravity\brain\581231c9-0c19-4477-b91a-7ab1a96c82a4\rc4_validation_results_with_context.json"
            with open(output_path, "w") as f:
                json.dump(final_data, f, indent=2)

            logger.info(f"Validation data dumped to {output_path}")
        finally:
            await transport.close()

if __name__ == "__main__":
    asyncio.run(main())
