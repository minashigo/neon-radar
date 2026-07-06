"""Infrastructure layer.

External integrations, abstracted through interfaces so the rest of the
application is exchange-agnostic.

* :mod:`neon_radar.infrastructure.exchanges` — exchange clients
  (Binance, Bybit, OKX, Hyperliquid) behind a single
  :class:`~neon_radar.infrastructure.exchanges.ExchangeClient` ABC.

This layer adapts the outside world to the domain.

Nothing in :mod:`neon_radar.domain` should import from here.
"""
