"""Script for extracting Point-in-Time MI features and Forward Outcomes."""

import asyncio
import csv
import logging
from datetime import UTC, date, datetime
from pathlib import Path

from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.config.models import TimeFrame
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.models import Symbol
from neon_radar.infrastructure.exchanges.binance import BinanceClient
from neon_radar.infrastructure.storage.intelligence_store import HistoricalIntelligenceStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_extraction():
    # 1. Setup minimal dependencies
    from neon_radar.config.loader import ConfigLoader

    config = ConfigLoader(Path("config.json")).load()
    scoring_config = ScoringRulesConfig(min_confidence=0.0, rules=[], regime_filter={"enabled": True})

    binance_client = BinanceClient(config.api)
    intelligence_store = HistoricalIntelligenceStore(Path("data/intelligence"))

    bt = TradeBacktester(
        exchange=binance_client,
        scoring_config=scoring_config,
        rules=(),
        intelligence_store=intelligence_store
    )

    symbol = Symbol("BTCUSDT")
    timeframe = TimeFrame("1d")
    start_date = date(2021, 1, 1)
    end_date = date(2026, 8, 1)

    logger.info("Prefetching market data...")
    await bt._prefetch([symbol], timeframe, start_date, end_date)

    series = bt.cache.get((symbol, timeframe))
    if not series:
        logger.error("No data fetched.")
        return

    records = []

    # Intercept build_setup instead of evaluate to access the MarketState
    original_build_setup = bt._pipeline.setup_engine.build_setup
    last_regime_info = {"regime": "UNKNOWN", "reason": "No setup"}

    def intercepted_build_setup(state, analysis_result):
        # The engine will classify the regime before returning, but we can just do it here
        if bt._regime_classifier:
            classification = bt._regime_classifier.classify(state)
            last_regime_info["regime"] = classification.regime.value
            last_regime_info["reason"] = classification.reason

        return original_build_setup(state, analysis_result)

    bt._pipeline.setup_engine.build_setup = intercepted_build_setup

    original_evaluate = bt._pipeline.evaluate
    def intercepted_evaluate(*args, **kwargs):
        # Reset
        last_regime_info["regime"] = "UNKNOWN"
        last_regime_info["reason"] = "No setup"

        setup = original_evaluate(*args, **kwargs)
        intelligence = kwargs.get("intelligence")
        history = args[0]
        current_candle = history.candles[-1]

        record = {
            "timestamp": int(current_candle.open_time),
            "date": str(datetime.fromtimestamp(current_candle.open_time / 1000, UTC).date()),
            "close": current_candle.close,
            "regime": last_regime_info["regime"],
            "regime_reason": last_regime_info["reason"],
        }

        if intelligence:
            record.update({
                "fng_value": getattr(intelligence, "fng_value", None),
                "fng_z_score_30d": getattr(intelligence, "fng_z_score_30d", None),
                "fng_percentile_30d": getattr(intelligence, "fng_percentile_30d", None),
                "dvol_value": getattr(intelligence, "dvol_value", None),
                "dvol_z_score_30d": getattr(intelligence, "dvol_z_score_30d", None),
                "dvol_percentile_30d": getattr(intelligence, "dvol_percentile_30d", None),
            })

        records.append(record)
        return setup

    bt._pipeline.evaluate = intercepted_evaluate

    logger.info("Running simulation for extraction...")
    bt._simulate_symbol(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        min_history_candles=50
    )

    # Now calculate forward outcomes (N=1, 3, 7, 14, 30)
    logger.info("Calculating forward outcomes...")

    candle_map = {int(c.open_time): c for c in series.candles}
    sorted_timestamps = sorted(candle_map.keys())

    final_records = []

    for rec in records:
        t = rec["timestamp"]
        idx = sorted_timestamps.index(t)

        if idx + 1 >= len(sorted_timestamps):
            continue

        entry_price = candle_map[sorted_timestamps[idx + 1]].open

        for n in [1, 3, 7, 14, 30]:
            if idx + 1 + n >= len(sorted_timestamps):
                rec[f"fwd_ret_{n}d"] = None
                rec[f"mfe_{n}d"] = None
                rec[f"mae_{n}d"] = None
                continue

            exit_price = candle_map[sorted_timestamps[idx + 1 + n]].close
            ret = (exit_price - entry_price) / entry_price

            period_candles = [candle_map[sorted_timestamps[i]] for i in range(idx + 1, idx + 1 + n + 1)]
            highs = [c.high for c in period_candles]
            lows = [c.low for c in period_candles]

            mfe = (max(highs) - entry_price) / entry_price
            mae = (min(lows) - entry_price) / entry_price

            rec[f"fwd_ret_{n}d"] = ret
            rec[f"mfe_{n}d"] = mfe
            rec[f"mae_{n}d"] = mae

        final_records.append(rec)

    out_dir = Path(r"C:\Users\orphan\.gemini\antigravity\brain\fb6f9fda-e97a-4713-83ed-9f3f4425bb3d\scratch")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "mi_features_outcomes.csv"

    if final_records:
        keys = final_records[0].keys()
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(final_records)

        logger.info(f"Successfully extracted {len(final_records)} records to {out_path}")
    else:
        logger.warning("No records extracted.")

if __name__ == "__main__":
    asyncio.run(run_extraction())
