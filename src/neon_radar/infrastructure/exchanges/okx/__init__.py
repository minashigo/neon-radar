"""OKX exchange client (placeholder).

OKX exposes a single public REST API for both spot and derivatives.
Relevant endpoints for our use case:

* ``GET /api/v5/market/candles`` — klines
* ``GET /api/v5/market/ticker`` — 24h ticker
* ``GET /api/v5/public/funding-rate`` — current funding
* ``GET /api/v5/rubik/stat/contracts/open-interest-history`` — OI

Implementation lands later. The package exists to make
``exchanges.okx.OkxClient`` a valid import target.
"""
