import asyncio
import json
import os
from pathlib import Path

from neon_radar.application.services.trade_analyzer import TradeAnalyzer
from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.application.services.parameter_optimizer import GridSearchOptimizer
from neon_radar.application.services.walk_forward_analyzer import WalkForwardAnalyzer
from neon_radar.config.loader import load_config
from neon_radar.config.scoring_loader import load_rules
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.infrastructure.exchanges.binance.client import BinanceClient
from datetime import date

PERIODS = {
    "Bull 1 (2020-2021)": ("2020-10-01", "2021-05-01"),
    "Bear (2021-2022)": ("2021-11-01", "2022-12-31"),
    "Chop (2023)": ("2023-04-01", "2023-09-30"),
    "Bull 2 (2023-2024)": ("2023-10-01", "2024-03-31"),
    "COVID Crash (2020)": ("2020-02-15", "2020-04-15"),
    "FTX Crash (2022)": ("2022-10-15", "2022-12-15"),
}

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
MIN_HISTORY = 100

async def main():
    print("Loading config...")
    cfg = load_config()
    from neon_radar.config.loader import _strip_meta
    scoring_raw = json.loads(Path("scoring_rules.json").read_text(encoding="utf-8"))
    scoring_cfg = ScoringRulesConfig.model_validate(_strip_meta(scoring_raw))
    rules = tuple(load_rules(Path("scoring_rules.json")))

    analyzer = TradeAnalyzer()
    optimizer = GridSearchOptimizer()
    wfa = WalkForwardAnalyzer(optimizer)

    os.makedirs("results", exist_ok=True)
    summary = {}

    async with BinanceClient(cfg.api) as client:
        backtester = TradeBacktester(
            exchange=client,
            scoring_config=scoring_cfg,
            rules=rules
        )

        global_trades = {"1d": [], "4h": []}
        global_reports = {}

        for tf in ["1d", "4h"]:
            print(f"Running global WFA for {tf}...")
            try:
                report = await wfa.run(
                    base_backtester=backtester,
                    start_date=date.fromisoformat("2019-08-01"),
                    end_date=date.fromisoformat("2024-04-01"),
                    symbols=tuple(SYMBOLS),
                    timeframe=tf,
                    is_window_months=6,
                    oos_window_months=2,
                    step_months=2,
                )
                global_reports[tf] = report
                
                # Aggregate all OOS trades
                for cycle in report.cycles:
                    global_trades[tf].extend(cycle.oos_report.trades)

            except Exception as e:
                print(f"Failed to run WFA on {tf}: {e}")

        # Now bucket trades into regimes
        for name, (start_str, end_str) in PERIODS.items():
            summary[name] = {}
            # Convert to ms
            from datetime import datetime, timezone
            start_ms = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
            end_ms = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000

            for tf in ["1d", "4h"]:
                if tf not in global_trades:
                    continue
                
                # Filter trades for this regime
                regime_trades = [
                    t for t in global_trades[tf] 
                    if start_ms <= t.entry_time <= end_ms
                ]
                
                # Analyze them
                report = analyzer.analyze(regime_trades)
                
                summary[name][tf] = {
                    "total_trades": report.total_trades,
                    "win_rate": report.win_rate,
                    "net_profit_factor": report.net_profit_factor,
                    "net_expectancy": report.net_expectancy,
                    "net_sharpe_ratio": report.net_sharpe_ratio,
                    "max_drawdown_pct": report.max_drawdown_pct,
                }

        # Overall summary
        summary["Overall"] = {}
        for tf in ["1d", "4h"]:
            if tf in global_trades:
                report = analyzer.analyze(global_trades[tf])
                summary["Overall"][tf] = {
                    "total_trades": report.total_trades,
                    "win_rate": report.win_rate,
                    "net_profit_factor": report.net_profit_factor,
                    "net_expectancy": report.net_expectancy,
                    "net_sharpe_ratio": report.net_sharpe_ratio,
                    "max_drawdown_pct": report.max_drawdown_pct,
                }

        with open("results/phase1_wfa_regime_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("WFA regime analysis completed. Results saved to results/phase1_wfa_regime_summary.json")

if __name__ == "__main__":
    asyncio.run(main())
