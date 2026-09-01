import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

START_DT = datetime(2023, 1, 1, tzinfo=UTC)
END_DT = datetime(2026, 8, 1, tzinfo=UTC)
INTERVAL = "4h"


async def fetch_symbol(client: httpx.AsyncClient, symbol: str, start_ms: int, end_ms: int):
    all_candles = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": 1500,
            "startTime": current_start,
            "endTime": end_ms,
        }
        resp = await client.get("https://fapi.binance.com/fapi/v1/klines", params=params)

        if resp.status_code == 400 and "Invalid symbol" in resp.text:
            logger.warning(
                f"{symbol} might be delisted or invalid at {current_start}. Stopping fetch."
            )
            break

        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_candles.extend(data)

        last_open_time = data[-1][0]
        # 4h = 4 * 60 * 60 * 1000 = 14400000 ms
        next_start = last_open_time + 14400000

        if next_start <= current_start:
            break

        current_start = next_start
        await asyncio.sleep(0.1)  # Rate limit protection

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_candles,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


async def main():
    start_ms = int(START_DT.timestamp() * 1000)
    end_ms = int(END_DT.timestamp() * 1000)

    out_dir = Path("research/experiments/rc7_cross_sectional_momentum/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient() as client:
        for sym in UNIVERSE:
            logger.info(f"Fetching {sym}...")
            df = await fetch_symbol(client, sym, start_ms, end_ms)
            if not df.empty:
                logger.info(
                    f"Fetched {len(df)} candles for {sym}. Range: {df['open_time'].min()} to {df['open_time'].max()}"
                )
                df.to_csv(out_dir / f"{sym}.csv", index=False)
            else:
                logger.warning(f"No data fetched for {sym}")


if __name__ == "__main__":
    asyncio.run(main())
