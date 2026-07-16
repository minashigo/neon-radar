"""Market-data orchestration service.

This is the only class the UI talks to for fetching data. It hides:

* the choice of exchange (Binance now, Bybit later)
* the cache layer (callers see the same data whether it came from
  disk or the network)
* the threading model (callers just connect to signals)

Lifecycle::

    service = MarketDataService(exchange=client, cache=cache)
    service.start()                                  # spawn worker thread
    service.request_klines.emit(Symbol("BTCUSDT"), TimeFrame.D1, limit=500)
    # ... later, in a slot connected to klines_ready:
    # def on_klines(series: KlineSeries) -> None: ...

Signals
-------
* ``klines_ready(symbol, timeframe, series)`` — successful fetch
* ``ticker_ready(ticker)`` — successful ticker fetch
* ``funding_ready(symbol, funding)`` — successful funding fetch
* ``open_interest_ready(symbol, oi)`` — successful OI fetch
* ``error_occurred(kind, message)`` — any failure
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Signal

from neon_radar.domain.exceptions import NeonRadarError
from neon_radar.domain.models import KlineSeries, Symbol
from neon_radar.utils.async_bridge import AsyncWorker

if TYPE_CHECKING:
    from collections.abc import Callable

    from neon_radar.config.models import TimeFrame
    from neon_radar.infrastructure.cache import KlineCache
    from neon_radar.infrastructure.exchanges.base import ExchangeClient


class MarketDataService(QObject):
    """Orchestrates fetching and caching market data via an :class:`ExchangeClient`."""

    klines_ready = Signal(object, object, object)  # Symbol, TimeFrame, KlineSeries
    ticker_ready = Signal(object)  # TickerStats
    funding_ready = Signal(object, object)  # Symbol, FundingRate
    open_interest_ready = Signal(object, object)  # Symbol, OpenInterest
    error_occurred = Signal(str, str)  # kind, message

    def __init__(
        self,
        *,
        exchange: ExchangeClient,
        cache: KlineCache | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        self._exchange = exchange
        self._cache = cache
        self._worker = AsyncWorker()
        self._clock = clock

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread. Idempotent."""
        if not self._worker.isRunning():
            self._worker.start()

    def stop(self, *, close_exchange: bool = True) -> None:
        """Stop the worker thread and optionally close the exchange."""
        if close_exchange:
            # Worker may already be stopped — swallow the RuntimeError.
            with suppress(RuntimeError):
                self._worker.submit(self._exchange.close())
        self._worker.stop()

    # ------------------------------------------------------------------
    # Public request API — call from main thread
    # ------------------------------------------------------------------

    def request_klines(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        *,
        limit: int = 500,
    ) -> None:
        """Request klines. Emits ``klines_ready`` or ``error_occurred``."""
        self._submit(
            symbol=symbol,
            timeframe=timeframe,
            coro_factory=lambda: self._fetch_klines(symbol, timeframe, limit),
            on_success=lambda result: self.klines_ready.emit(symbol, timeframe, result),
        )

    def request_ticker(self, symbol: Symbol) -> None:
        self._submit(
            symbol=symbol,
            timeframe=None,
            coro_factory=lambda: self._exchange.get_ticker(symbol),
            on_success=self.ticker_ready.emit,
        )

    def request_funding_rate(self, symbol: Symbol) -> None:
        self._submit(
            symbol=symbol,
            timeframe=None,
            coro_factory=lambda: self._exchange.get_funding_rate(symbol),
            on_success=lambda fr: self.funding_ready.emit(symbol, fr),
        )

    def request_open_interest(self, symbol: Symbol) -> None:
        self._submit(
            symbol=symbol,
            timeframe=None,
            coro_factory=lambda: self._exchange.get_open_interest(symbol),
            on_success=lambda oi: self.open_interest_ready.emit(symbol, oi),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _submit(
        self,
        *,
        symbol: Symbol,
        timeframe: TimeFrame | None,
        coro_factory: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        """Cache-aware submission helper.

        * If a cache exists and has a fresh entry, emit it directly
          (synchronously) — no thread hop needed.
        * Otherwise submit the coroutine to the worker. The completion
          callback runs in the **worker thread** and emits the result
          via Qt signal, which Qt marshals back to the main thread.
        """
        if self._cache is not None and timeframe is not None:
            cached = self._cache.get(symbol, timeframe)
            if cached is not None:
                on_success(cached)
                return

        try:
            future = self._worker.submit(coro_factory())
        except RuntimeError as exc:
            # Worker not running yet. Emit as error so the UI can
            # surface the issue (e.g. "service.start() not called").
            self.error_occurred.emit("worker_not_running", str(exc))
            return

        def _done(fut: Any) -> None:
            try:
                result = fut.result()
            except NeonRadarError as exc:
                self.error_occurred.emit(type(exc).__name__, str(exc))
            except Exception as exc:  # last-resort safety net
                self.error_occurred.emit(type(exc).__name__, str(exc))
            else:
                if (
                    self._cache is not None
                    and timeframe is not None
                    and isinstance(result, KlineSeries)
                ):
                    self._cache.put(result)
                on_success(result)

        future.add_done_callback(_done)

    async def _fetch_klines(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        limit: int,
    ) -> KlineSeries:
        return await self._exchange.get_klines(symbol, timeframe, limit=limit)
