"""Binance Futures exchange client.

Public REST API only — no API key required.

Endpoints used:

* ``GET /fapi/v1/exchangeInfo`` — for ``ExchangeClient.info``
* ``GET /fapi/v1/klines`` — for :meth:`BinanceClient.get_klines`
* ``GET /fapi/v1/ticker/24hr`` — for :meth:`BinanceClient.get_ticker`
* ``GET /fapi/v1/premiumIndex`` — for :meth:`BinanceClient.get_funding_rate`
* ``GET /fapi/v1/openInterest`` — for :meth:`BinanceClient.get_open_interest`

See :class:`BinanceClient` for the implementation.
"""

from neon_radar.infrastructure.exchanges.binance.client import BinanceClient
from neon_radar.infrastructure.exchanges.binance.rate_limiter import (
    RateLimiterConfig,
    TokenBucketRateLimiter,
)

__all__ = ["BinanceClient", "RateLimiterConfig", "TokenBucketRateLimiter"]
