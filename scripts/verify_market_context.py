"""Manual verification script for Market Context Providers."""

import asyncio
import time
from neon_radar.domain.models import Symbol
from neon_radar.infrastructure.exchanges.binance_transport import BinanceTransport
from neon_radar.application.services.market_context.cache import ContextCache
from neon_radar.infrastructure.providers.binance_context import BinanceContextProviders

async def main():
    transport = BinanceTransport(base_url="https://fapi.binance.com")
    cache = ContextCache()
    providers = BinanceContextProviders(transport, cache)
    
    symbol = Symbol("BTCUSDT")
    current_time = int(time.time() * 1000)
    
    print(f"Fetching Market Context for {symbol}...")
    
    # 1. Fetch Funding
    funding = await providers.get_funding(symbol, current_time)
    print("\n--- Funding Context ---")
    if funding:
        print(f"Raw Funding:      {funding.raw_funding:.6f}")
        print(f"Annualized APR:   {funding.annualized_apr:.4f}")
        print(f"Mark Price:       {funding.mark_price:.2f}")
    else:
        print("Failed to fetch Funding")

    # 2. Fetch Open Interest
    oi = await providers.get_open_interest(symbol, current_time)
    print("\n--- Open Interest Context ---")
    if oi:
        print(f"OI (Coin):        {oi.oi_coin:.2f}")
        print(f"OI (USD):         {oi.oi_usd_notional:.2f}")
    else:
        print("Failed to fetch Open Interest")

    # 3. Fetch Long/Short Ratio
    ls_ratio = await providers.get_long_short_ratio(symbol, current_time)
    print("\n--- Long/Short Ratio Context ---")
    if ls_ratio:
        print(f"Long %:           {ls_ratio.long_pct:.4f}")
        print(f"Short %:          {ls_ratio.short_pct:.4f}")
        print(f"L/S Ratio:        {ls_ratio.ls_ratio:.4f}")
    else:
        print("Failed to fetch Long/Short Ratio")

    # 4. Fetch Taker Flow
    taker = await providers.get_taker_flow(symbol, current_time)
    print("\n--- Taker Flow Context ---")
    if taker:
        print(f"Buy Volume:       {taker.buy_volume:.2f}")
        print(f"Sell Volume:      {taker.sell_volume:.2f}")
        print(f"Net Buy Volume:   {taker.net_buy_volume:.2f}")
    else:
        print("Failed to fetch Taker Flow")

    await transport.close()

if __name__ == "__main__":
    asyncio.run(main())
