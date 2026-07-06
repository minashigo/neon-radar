"""Bybit exchange client (placeholder).

Bybit's unified V5 API exposes equivalent endpoints for klines, ticker
24h, funding rate, and open interest. The actual implementation is a
future roadmap item; the package exists so ``exchanges.bybit.BybitClient``
is a valid import target.

Implementation notes for the future:

* Base URL: ``https://api.bybit.com`` (or ``api.bybit.com/v5``).
* Kline endpoint: ``GET /v5/market/kline`` with ``category=linear``.
* Funding endpoint: ``GET /v5/market/funding/history``.
* Open interest: ``GET /v5/market/open-interest``.
"""
