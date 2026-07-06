"""Abstract exchange client interface.

This is the single contract every concrete exchange (Binance, Bybit,
OKX, Hyperliquid, …) must implement. Application services depend on
this interface only — never on a concrete client.

Design notes
------------
* The interface is intentionally **narrow**: it exposes only what the
  scoring engine actually needs. Adding exchange-specific endpoints
  (margin info, liquidation feed, …) means adding new abstract
  methods here **and** updating every concrete implementation — a
  deliberate tax that prevents drift.
* ``get_funding_rate`` and ``get_open_interest`` are part of the
  mandatory surface because requirement #4 calls for funding rate and
  open interest as scoring factors. A spot-only exchange must
  implement them by raising :class:`ExchangeError` (see
  :class:`neon_radar.domain.exceptions.ExchangeError`).
* Methods are ``async`` so concrete implementations can use ``httpx``
  or ``websockets`` without conversion. The application layer awaits
  them from inside an asyncio loop running in a QThread.
* No retry, rate-limit, or caching logic lives here — those are
  cross-cutting concerns handled by a decorator/wrapper in the
  application layer. Keeps this interface simple and easy to fake
  in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.config.models import TimeFrame
    from neon_radar.domain.funding import FundingRate, OpenInterest
    from neon_radar.domain.models import KlineSeries, Symbol, TickerStats


@dataclass(slots=True, frozen=True)
class ExchangeInfo:
    """Static metadata about an exchange.

    Returned by :meth:`ExchangeClient.info` so the UI can show the user
    which exchange is currently active.
    """

    name: str
    display_name: str
    website: str
    supports_funding: bool
    supports_open_interest: bool


class ExchangeClient(ABC):
    """Abstract interface every exchange client must implement.

    Lifetime
    --------
    Instances are designed to be **long-lived**. Construct one at
    startup, share across the application, close on shutdown via
    :meth:`close`. Implementations are responsible for keeping the
    underlying HTTP client alive.
    """

    #: Human-readable name shown in the UI ("Binance", "Bybit", …).
    name: str = ""

    @abstractmethod
    async def info(self) -> ExchangeInfo:
        """Return static metadata about the exchange."""

    @abstractmethod
    async def get_klines(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        *,
        limit: int = 500,
        end_time: int | None = None,
    ) -> KlineSeries:
        """Fetch historical OHLCV candles.

        Parameters
        ----------
        symbol
            Trading pair, e.g. ``BTCUSDT``.
        timeframe
            Candle interval.
        limit
            Maximum number of candles to return. Implementations may
            cap this at an exchange-defined maximum and document it.
        end_time
            Optional Unix-ms upper bound; if ``None``, fetch the most
            recent candles.

        Returns
        -------
        KlineSeries
            Immutable series sorted ascending by ``open_time``. Empty
            series if no data is available.
        """

    @abstractmethod
    async def get_ticker(self, symbol: Symbol) -> TickerStats:
        """Fetch 24h ticker statistics for ``symbol``.

        Raises
        ------
        ExchangeError
            If the exchange does not provide this endpoint.
        """

    async def get_funding_rate(self, symbol: Symbol) -> FundingRate:
        """Fetch the current funding rate for ``symbol``.

        Default implementation raises :class:`ExchangeError` so spot-
        only exchanges do not have to implement it explicitly.
        """
        raise _not_implemented(self.name, "get_funding_rate")

    async def get_open_interest(self, symbol: Symbol) -> OpenInterest:
        """Fetch the current open interest for ``symbol``.

        Default implementation raises :class:`ExchangeError` — same
        reasoning as :meth:`get_funding_rate`.
        """
        raise _not_implemented(self.name, "get_open_interest")

    async def close(self) -> None:
        """Release any resources held by the client.

        Default is a no-op so simple implementations do not have to
        override it.
        """
        return None


def _not_implemented(exchange_name: str, method: str) -> Exception:
    """Helper: build an ExchangeError for unsupported operations.

    Imported lazily to avoid a circular dependency with the exceptions
    module — both ``ExchangeClient`` and ``ExchangeError`` live in the
    domain layer, but the import order at module load time would
    otherwise be tricky.
    """
    from neon_radar.domain.exceptions import ExchangeError

    return ExchangeError(
        f"Exchange '{exchange_name}' does not support {method}()"
    )
