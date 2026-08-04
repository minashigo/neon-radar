"""Infrastructure layer for intelligence providers."""

import contextlib

from neon_radar.infrastructure.providers.base import BaseRateLimitedProvider

# Import specific providers here to ensure they are registered in provider_registry
with contextlib.suppress(ImportError):
    from neon_radar.infrastructure.providers.coinglass import CoinGlassProvider
    from neon_radar.infrastructure.providers.alternative_me import AlternativeMeProvider

__all__ = ["BaseRateLimitedProvider", "CoinGlassProvider", "AlternativeMeProvider"]
