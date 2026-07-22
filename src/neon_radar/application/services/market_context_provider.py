"""Abstract interfaces for Market Context Providers."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.market_context import (
        FundingContext,
        FundingSeries,
        LiquidationContext,
        LiquidationSeries,
        LongShortRatioContext,
        LongShortSeries,
        OpenInterestContext,
        OpenInterestSeries,
        TakerFlowContext,
        TakerFlowSeries,
    )
    from neon_radar.domain.models import Symbol


class MarketContextProvider(abc.ABC):
    """Base interface for all Market Context providers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g. 'binance_funding')."""
        pass


class FundingProvider(MarketContextProvider):
    @abc.abstractmethod
    async def get_funding(self, symbol: Symbol, timestamp: int) -> FundingContext | None:
        """Fetch funding context available at the given timestamp."""
        pass

    @abc.abstractmethod
    async def get_funding_history(
        self, symbol: Symbol, start_time: int, end_time: int, limit: int = 500
    ) -> FundingSeries | None:
        """Fetch a historical series of funding context within the given time range."""
        pass


class OpenInterestProvider(MarketContextProvider):
    @abc.abstractmethod
    async def get_open_interest(self, symbol: Symbol, timestamp: int) -> OpenInterestContext | None:
        """Fetch open interest context available at the given timestamp."""
        pass

    @abc.abstractmethod
    async def get_open_interest_history(
        self, symbol: Symbol, start_time: int, end_time: int, limit: int = 500
    ) -> OpenInterestSeries | None:
        """Fetch a historical series of open interest within the given time range."""
        pass


class LongShortProvider(MarketContextProvider):
    @abc.abstractmethod
    async def get_long_short_ratio(self, symbol: Symbol, timestamp: int) -> LongShortRatioContext | None:
        """Fetch long/short ratio context available at the given timestamp."""
        pass

    @abc.abstractmethod
    async def get_long_short_ratio_history(
        self, symbol: Symbol, start_time: int, end_time: int, limit: int = 500
    ) -> LongShortSeries | None:
        """Fetch a historical series of long/short ratio within the given time range."""
        pass


class TakerFlowProvider(MarketContextProvider):
    @abc.abstractmethod
    async def get_taker_flow(self, symbol: Symbol, timestamp: int) -> TakerFlowContext | None:
        """Fetch taker flow context available at the given timestamp."""
        pass

    @abc.abstractmethod
    async def get_taker_flow_history(
        self, symbol: Symbol, start_time: int, end_time: int, limit: int = 500
    ) -> TakerFlowSeries | None:
        """Fetch a historical series of taker flow within the given time range."""
        pass


class LiquidationProvider(MarketContextProvider):
    @abc.abstractmethod
    async def get_liquidations(self, symbol: Symbol, timestamp: int) -> LiquidationContext | None:
        """Fetch liquidations context available at the given timestamp."""
        pass

    @abc.abstractmethod
    async def get_liquidations_history(
        self, symbol: Symbol, start_time: int, end_time: int, limit: int = 500
    ) -> LiquidationSeries | None:
        """Fetch a historical series of liquidations within the given time range."""
        pass
