"""Contracts for external intelligence providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neon_radar.domain.market_intelligence.models import SignalEvidence


@runtime_checkable
class IntelligenceProvider(Protocol):
    """Base protocol for all intelligence providers."""

    @property
    def provider_name(self) -> str:
        """Unique name of the provider (e.g., 'Glassnode', 'CoinGlass')."""
        ...

    @property
    def provider_type(self) -> str:
        """Type of the provider (e.g., 'OnChain', 'News')."""
        ...

    async def fetch_signals(self, timestamp: int) -> tuple[SignalEvidence, ...]:
        """Fetch the latest signals available from this provider."""
        ...


@runtime_checkable
class NewsProvider(IntelligenceProvider, Protocol):
    """Protocol for news aggregators and headline sentiment providers."""
    ...


@runtime_checkable
class SocialProvider(IntelligenceProvider, Protocol):
    """Protocol for social media sentiment and trending narrative providers."""
    ...


@runtime_checkable
class OnChainProvider(IntelligenceProvider, Protocol):
    """Protocol for blockchain metrics and whale activity providers."""
    ...


@runtime_checkable
class MacroProvider(IntelligenceProvider, Protocol):
    """Protocol for traditional finance and macroeconomic data providers."""
    ...
