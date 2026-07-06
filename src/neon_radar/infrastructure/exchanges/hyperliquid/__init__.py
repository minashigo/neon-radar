"""Hyperliquid exchange client (placeholder).

Hyperliquid is a decentralised perpetual exchange. Its data layer is
HTTP-RPC, not REST, which makes it structurally different from CEX
exchanges but functionally equivalent for our purposes. Relevant
endpoints:

* ``POST /info`` with ``{"type": "candleSnapshot", …}``
* ``POST /info`` with ``{"type": "ticker", …}``
* ``POST /info`` with ``{"type": "metaAndAssetCtxs", …}`` (funding + OI)

Implementation lands later.
"""
