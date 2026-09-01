import asyncio
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.config.loader import ConfigLoader, _strip_meta
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.models import Symbol
from neon_radar.infrastructure.exchanges.binance import BinanceClient

logging.basicConfig(level=logging.WARNING)

UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "MATICUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "SOLUSDT",
    "UNIUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "ATOMUSDT",
]


async def audit_data():
    config = ConfigLoader(Path("config.json")).load()
    raw = json.loads(Path("scoring_rules.json").read_text(encoding="utf-8"))
    scoring_config = ScoringRulesConfig.model_validate(_strip_meta(raw))
    from neon_radar.config.scoring_loader import load_rules

    rules = tuple(load_rules(Path("scoring_rules.json")))
    binance_client = BinanceClient(config.api)
    symbols = [Symbol(s) for s in UNIVERSE]

    start_date = date(2023, 1, 1)
    end_date = date(2026, 8, 1)

    bt_base = TradeBacktester(
        exchange=binance_client,
        scoring_config=scoring_config,
        rules=rules,
    )

    print("Prefetching data...")
    await bt_base._prefetch(symbols, "4h", start_date, end_date)

    res = {}
    for sym in symbols:
        series = bt_base._series_cache.get((str(sym), "4h"))
        if series and not series.is_empty:
            st = pd.to_datetime(series.candles[0].open_time, unit="ms")
            et = pd.to_datetime(series.candles[-1].open_time, unit="ms")
            res[str(sym)] = {"start": st, "end": et, "count": len(series.candles)}

    print("\nData Window Audit:")
    for sym, val in res.items():
        print(f"{sym}: Start {val['start']}, End {val['end']}, Count {val['count']}")

    df = pd.DataFrame(res).T
    print("\nOverall Earliest:", df["start"].min())
    print("Overall Latest:", df["end"].max())


if __name__ == "__main__":
    asyncio.run(audit_data())
