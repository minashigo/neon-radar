"""Service for assembling HistoricalMarketContext from various providers."""

import asyncio

from neon_radar.application.services.market_context_provider import (
    FundingProvider,
    LiquidationProvider,
    LongShortProvider,
    MarketContextProvider,
    OpenInterestProvider,
    TakerFlowProvider,
)
from neon_radar.domain.market_context import HistoricalMarketContext
from neon_radar.domain.models import Symbol


class MarketContextHistoryService:
    """Orchestrates fetching of historical market context data from multiple providers."""

    def __init__(self, providers: list[MarketContextProvider]) -> None:
        self.providers = providers

    async def get_historical_context(
        self, symbol: Symbol, timestamp: int, start_time: int, end_time: int, limit: int = 500
    ) -> HistoricalMarketContext:
        """Fetch all available historical context and assemble HistoricalMarketContext."""
        context = HistoricalMarketContext(symbol=symbol, timestamp=timestamp)

        # We can fetch concurrently
        tasks = []

        for provider in self.providers:
            if isinstance(provider, FundingProvider):
                tasks.append(self._fetch_funding_history(provider, context, symbol, start_time, end_time, limit))
            if isinstance(provider, OpenInterestProvider):
                tasks.append(self._fetch_open_interest_history(provider, context, symbol, start_time, end_time, limit))
            if isinstance(provider, LongShortProvider):
                tasks.append(self._fetch_long_short_history(provider, context, symbol, start_time, end_time, limit))
            if isinstance(provider, TakerFlowProvider):
                tasks.append(self._fetch_taker_flow_history(provider, context, symbol, start_time, end_time, limit))
            if isinstance(provider, LiquidationProvider):
                tasks.append(self._fetch_liquidations_history(provider, context, symbol, start_time, end_time, limit))

        await asyncio.gather(*tasks)
        return context

    async def _fetch_funding_history(
        self, provider: FundingProvider, context: HistoricalMarketContext, symbol: Symbol, start: int, end: int, limit: int
    ) -> None:
        series = await provider.get_funding_history(symbol, start, end, limit)
        if series:
            # We set using object.__setattr__ because HistoricalMarketContext is frozen
            object.__setattr__(context, "funding_history", series)

    async def _fetch_open_interest_history(
        self, provider: OpenInterestProvider, context: HistoricalMarketContext, symbol: Symbol, start: int, end: int, limit: int
    ) -> None:
        series = await provider.get_open_interest_history(symbol, start, end, limit)
        if series:
            object.__setattr__(context, "open_interest_history", series)

    async def _fetch_long_short_history(
        self, provider: LongShortProvider, context: HistoricalMarketContext, symbol: Symbol, start: int, end: int, limit: int
    ) -> None:
        series = await provider.get_long_short_ratio_history(symbol, start, end, limit)
        if series:
            object.__setattr__(context, "ls_ratio_history", series)

    async def _fetch_taker_flow_history(
        self, provider: TakerFlowProvider, context: HistoricalMarketContext, symbol: Symbol, start: int, end: int, limit: int
    ) -> None:
        series = await provider.get_taker_flow_history(symbol, start, end, limit)
        if series:
            object.__setattr__(context, "taker_flow_history", series)

    async def _fetch_liquidations_history(
        self, provider: LiquidationProvider, context: HistoricalMarketContext, symbol: Symbol, start: int, end: int, limit: int
    ) -> None:
        series = await provider.get_liquidations_history(symbol, start, end, limit)
        if series:
            object.__setattr__(context, "liquidations_history", series)
