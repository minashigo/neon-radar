import asyncio
import time
from pathlib import Path
from neon_radar.domain.models import Symbol
from neon_radar.infrastructure.exchanges.binance_transport import BinanceTransport
from neon_radar.application.services.market_context.cache import ContextCache
from neon_radar.infrastructure.providers.binance_context import BinanceContextProviders
from neon_radar.application.services.market_context.history_service import MarketContextHistoryService
from neon_radar.utils.logging import get_logger

logger = get_logger(__name__)

async def main():
    symbol = Symbol("BTCUSDT")
    end_time = int(time.time() * 1000)
    start_time = end_time - 10 * 3600 * 1000  # 10 hours ago

    transport = BinanceTransport(base_url="https://fapi.binance.com")
    cache = ContextCache(directory=Path(".cache/market_context"))
    provider = BinanceContextProviders(transport, cache)
    service = MarketContextHistoryService(providers=[provider])

    print(f"Fetching Historical Context for {symbol}...")
    
    historical_context = await service.get_historical_context(
        symbol=symbol,
        timestamp=end_time,
        start_time=start_time,
        end_time=end_time,
        limit=500
    )

    if historical_context.funding_history:
        print(f"Funding Series count: {len(historical_context.funding_history.items)}")
        print(f"Latest Funding: {historical_context.funding_history.latest()}")
    else:
        print("No Funding History found.")

    if historical_context.open_interest_history:
        print(f"OI Series count: {len(historical_context.open_interest_history.items)}")
        print(f"Latest OI: {historical_context.open_interest_history.latest()}")
    else:
        print("No OI History found.")

    if historical_context.ls_ratio_history:
        print(f"Long/Short Series count: {len(historical_context.ls_ratio_history.items)}")
        print(f"Latest LS Ratio: {historical_context.ls_ratio_history.latest()}")
    else:
        print("No LS Ratio History found.")

    if historical_context.taker_flow_history:
        print(f"Taker Flow Series count: {len(historical_context.taker_flow_history.items)}")
        print(f"Latest Taker Flow: {historical_context.taker_flow_history.latest()}")
    else:
        print("No Taker Flow History found.")

    await transport.close()

if __name__ == "__main__":
    asyncio.run(main())
