"""Abstract interfaces for Market Context Providers."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.market_context import (
        FundingContext,
        LiquidationContext,
        LongShortRatioContext,
        OpenInterestContext,
        TakerFlowContext,
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


class OpenInterestProvider(MarketContextProvider):
    @abc.abstractmethod
    async def get_open_interest(self, symbol: Symbol, timestamp: int) -> OpenInterestContext | None:
        """Fetch open interest context available at the given timestamp."""
        pass


class LongShortProvider(MarketContextProvider):
    @abc.abstractmethod
    async def get_long_short_ratio(self, symbol: Symbol, timestamp: int) -> LongShortRatioContext | None:
        """Fetch long/short ratio context available at the given timestamp."""
        pass


class TakerFlowProvider(MarketContextProvider):
    @abc.abstractmethod
    async def get_taker_flow(self, symbol: Symbol, timestamp: int) -> TakerFlowContext | None:
        """Fetch taker flow context available at the given timestamp."""
        pass


class LiquidationProvider(MarketContextProvider):
    @abc.abstractmethod
    async def get_liquidations(self, symbol: Symbol, timestamp: int) -> LiquidationContext | None:
        """Fetch liquidations context available at the given timestamp."""
        pass
