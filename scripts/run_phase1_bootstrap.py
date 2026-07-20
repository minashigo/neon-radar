import asyncio
import json
import os
from pathlib import Path

from neon_radar.application.services.bootstrap_analyzer import BootstrapAnalyzer
from neon_radar.application.services.trade_analyzer import TradeAnalyzer
from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.config.loader import load_config
from neon_radar.config.scoring_loader import load_rules
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.infrastructure.exchanges.binance.client import BinanceClient

PERIODS = {
    "Bull 1 (2020-2021)": ("2020-10-01", "2021-05-01"),
    "Bear (2021-2022)": ("2021-11-01", "2022-12-31"),
    "Chop (2023)": ("2023-04-01", "2023-09-30"),
    "Bull 2 (2023-2024)": ("2023-10-01", "2024-03-31"),
    "COVID Crash (2020)": ("2020-02-15", "2020-04-15"),
    "FTX Crash (2022)": ("2022-10-15", "2022-12-15"),
}

TIMEFRAMES = ["1d", "4h"]
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
    boot_analyzer = BootstrapAnalyzer(analyzer.analyze)

    os.makedirs("results", exist_ok=True)
    summary = {}

    async with BinanceClient(cfg.api) as client:
        backtester = TradeBacktester(
            exchange=client,
            scoring_config=scoring_cfg,
            rules=rules
        )

        for name, (start_str, end_str) in PERIODS.items():
            from datetime import date
            start = date.fromisoformat(start_str)
            end = date.fromisoformat(end_str)
            summary[name] = {}
            for tf in TIMEFRAMES:
                print(f"Running Bootstrap for {name} on {tf}...")
                try:
                    trades = await backtester.run(
                        start_date=start,
                        end_date=end,
                        symbols=tuple(SYMBOLS),
                        timeframe=tf,
                        min_history_candles=MIN_HISTORY,
                    )
                except Exception as e:
                    print(f"Failed to run backtest for {name} on {tf}: {e}")
                    continue

                report = analyzer.analyze(trades)
                boot_report = boot_analyzer.run(trades, block_size=20, iterations=1000)

                # Convert backtest report to dict
                b_dict = {
                    "net_profit_factor": report.net_profit_factor,
                    "net_expectancy": report.net_expectancy,
                    "win_rate": report.win_rate,
                    "net_sharpe_ratio": report.net_sharpe_ratio,
                    "max_drawdown_pct": report.max_drawdown_pct,
                }

                # Convert boot report metrics to dict
                boot_dict = {"metrics": {}}
                if boot_report:
                    for m_name, dist in boot_report.metrics.items():
                        boot_dict["metrics"][m_name] = {
                            "mean": dist.mean,
                            "median": dist.median,
                            "ci_lower_95": dist.ci_lower_95,
                            "ci_upper_95": dist.ci_upper_95
                        }

                summary[name][tf] = {
                    "backtest_report": b_dict,
                    "boot_report": boot_dict
                }

                with open("results/phase1_bootstrap_raw.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

    print("Bootstrap analysis completed. Results saved to results/phase1_bootstrap_raw.json")

if __name__ == "__main__":
    asyncio.run(main())
