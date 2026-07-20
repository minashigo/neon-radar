"""Live Data Fetcher Service.

Periodically polls the exchange for new candle data and feeds it to the Paper Trading Engine.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neon_radar.application.services.paper_trading_engine import PaperTradingEngine
    from neon_radar.config.models import TimeFrame
    from neon_radar.domain.models import Symbol
    from neon_radar.infrastructure.exchanges.base import ExchangeClient

logger = get_logger(__name__)


class LiveDataFetcher:
    """Continuously fetches klines and pushes them to the paper trading engine."""

    def __init__(self, exchange: ExchangeClient, engine: PaperTradingEngine, poll_interval_seconds: int = 60) -> None:
        self.exchange = exchange
        self.engine = engine
        self.poll_interval = poll_interval_seconds
        self._running = False

    async def run(self, symbols: Iterable[Symbol], timeframe: TimeFrame) -> None:
        """Run the continuous fetching loop."""
        symbols = tuple(symbols)
        self._running = True
        logger.info(f"Starting Live Data Fetcher for {len(symbols)} symbols. Polling every {self.poll_interval}s.")

        while self._running:
            for symbol in symbols:
                try:
                    # Fetch klinedata up to now (500 candles is enough for indicator history)
                    series = await self.exchange.get_klines(symbol, timeframe, limit=500)
                    if not series.is_empty:
                        self.engine.process_kline(symbol, series)
                except Exception as exc:
                    logger.error(f"Error fetching live data for {symbol}: {exc}")

            # Wait for next poll interval
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        """Stop the fetching loop."""
        self._running = False
