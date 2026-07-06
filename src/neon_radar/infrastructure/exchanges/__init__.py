"""Exchange client abstraction.

Concrete exchange clients (Binance, Bybit, OKX, Hyperliquid, …) all
implement :class:`ExchangeClient`. Application services depend only on
the abstract interface, which means:

* Swapping Binance for another exchange is a one-line change.
* Testing the application layer does not require any real exchange.
* The presentation layer never sees exchange-specific types.
"""

from neon_radar.infrastructure.exchanges.base import (
    ExchangeClient,
    ExchangeInfo,
)

__all__ = ["ExchangeClient", "ExchangeInfo"]
