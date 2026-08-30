"""Script to fetch historical MI data."""

import asyncio
import time
from pathlib import Path

from neon_radar.config.intelligence import ProviderConfig
from neon_radar.infrastructure.providers.alternative_me.provider import AlternativeMeProvider
from neon_radar.infrastructure.providers.deribit.provider import DeribitProvider
from neon_radar.infrastructure.storage.intelligence_store import HistoricalIntelligenceStore


async def main():
    store = HistoricalIntelligenceStore(Path("data/intelligence"))
    start_time = 0
    end_time = int(time.time() * 1000)

    cfg1 = ProviderConfig(enabled=True)
    alt_provider = AlternativeMeProvider(cfg1)

    print("Fetching Alternative.me (F&G)...")
    obs1 = await alt_provider.fetch_historical_signals(start_time, end_time)
    print(f"Fetched {len(obs1)} F&G observations.")
    store.append_series("fear_and_greed", list(obs1))

    cfg2 = ProviderConfig(enabled=True)
    deribit_provider = DeribitProvider(cfg2)

    print("Fetching Deribit (DVOL)...")
    obs2 = await deribit_provider.fetch_historical_signals(start_time, end_time)
    print(f"Fetched {len(obs2)} DVOL observations.")
    store.append_series("dvol", list(obs2))

    print("Historical MI update complete.")

if __name__ == "__main__":
    asyncio.run(main())
