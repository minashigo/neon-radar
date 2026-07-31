"""Registry for dynamically discovering and instantiating intelligence providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from neon_radar.application.intelligence.providers import IntelligenceProvider
    from neon_radar.config.intelligence import ProviderConfig

T = TypeVar("T", bound="IntelligenceProvider")
ProviderFactory = Callable[["ProviderConfig"], "IntelligenceProvider"]


class ProviderRegistry:
    """Registry to manage intelligence providers."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str) -> Callable[[ProviderFactory], ProviderFactory]:
        """Decorator to register a provider factory by name."""

        def decorator(factory: ProviderFactory) -> ProviderFactory:
            self._factories[name] = factory
            return factory

        return decorator

    def register_factory(self, name: str, factory: ProviderFactory) -> None:
        """Register a provider factory explicitly."""
        self._factories[name] = factory

    def create_provider(self, name: str, config: ProviderConfig) -> IntelligenceProvider:
        """Instantiate a provider by name."""
        if name not in self._factories:
            raise ValueError(f"Unknown intelligence provider: {name}")
        return self._factories[name](config)

    def get_registered_names(self) -> tuple[str, ...]:
        """Return a tuple of all registered provider names."""
        return tuple(self._factories.keys())


# Global registry instance
provider_registry = ProviderRegistry()
