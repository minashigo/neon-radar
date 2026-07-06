"""Funding rate and open interest value objects.

These two pieces of derivatives-market data are **not derivable from
candles** — they live on a separate API surface. Putting them in the
domain here means:

* The scoring engine can treat them as factors (requirement #4).
* They participate in immutability guarantees.
* Exchange mappers convert raw API responses into these types in one place.

Design notes
------------
* All values are immutable, slots-based dataclasses — same rules as
  :mod:`neon_radar.domain.models`.
* ``FundingRate.rate`` is stored as a **decimal** (e.g. ``0.0001`` for
  0.01%). We do not multiply by 100 to get "percent" — that conversion
  belongs in the presentation layer where formatting lives.
* ``OpenInterest`` carries both base and quote representations when the
  exchange provides them; otherwise only ``value`` is populated.
* Timestamps are Unix milliseconds (UTC), matching :class:`OHLCV`.
"""

from __future__ import annotations

from dataclasses import dataclass

from neon_radar.domain.models import Symbol


@dataclass(slots=True, frozen=True)
class FundingRate:
    """Current funding rate for a perpetual futures symbol.

    Attributes
    ----------
    symbol
        The trading pair (e.g. ``BTCUSDT``).
    rate
        Raw funding rate as a decimal (e.g. ``0.0001`` = 1 bps = 0.01%).
        Positive = longs pay shorts; negative = shorts pay longs.
    mark_price
        Mark price used for funding calculation, if provided by the exchange.
    next_funding_time
        Unix ms when the next funding settlement happens.
    timestamp
        Unix ms when this snapshot was taken.
    """

    symbol: Symbol
    rate: float
    mark_price: float | None = None
    next_funding_time: int | None = None
    timestamp: int | None = None

    def __post_init__(self) -> None:
        # Tolerate string inputs — Symbol normalises them.
        if not isinstance(self.symbol, Symbol):
            object.__setattr__(self, "symbol", Symbol(self.symbol))

    @property
    def is_positive(self) -> bool:
        """True if longs are paying shorts (typically a bullish-crowd signal)."""
        return self.rate > 0

    @property
    def annualized_pct(self) -> float:
        """Approximate annualised funding in percent (3 settlements/day x 365).

        Useful for comparing funding cost vs holding spot. The formula is
        ``rate * 3 * 365 * 100`` - assumes 8-hour funding intervals, which
        is Binance / Bybit / OKX default.
        """
        return self.rate * 3 * 365 * 100


@dataclass(slots=True, frozen=True)
class OpenInterest:
    """Outstanding contracts/open interest for a futures symbol.

    Attributes
    ----------
    symbol
        Trading pair.
    value
        Open interest in **base asset** (e.g. BTC). Some exchanges also
        report this in contracts; that detail is dropped here.
    value_quote
        Open interest in **quote asset** (e.g. USDT). Optional - not all
        exchanges compute it server-side.
    timestamp
        Unix ms when this snapshot was taken.
    """

    symbol: Symbol
    value: float
    value_quote: float | None = None
    timestamp: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, Symbol):
            object.__setattr__(self, "symbol", Symbol(self.symbol))
        if self.value < 0:
            raise ValueError(f"OpenInterest.value must be non-negative, got {self.value}")
        if self.value_quote is not None and self.value_quote < 0:
            raise ValueError(
                f"OpenInterest.value_quote must be non-negative, got {self.value_quote}"
            )
