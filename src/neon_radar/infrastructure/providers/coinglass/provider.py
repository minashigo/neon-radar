"""CoinGlass provider implementation."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from neon_radar.application.intelligence.registry import provider_registry
from neon_radar.domain.market_intelligence.models import DataQuality, ProviderResult
from neon_radar.infrastructure.providers.base import BaseRateLimitedProvider
from neon_radar.infrastructure.providers.coinglass.config import CoinGlassConfig
from neon_radar.infrastructure.providers.coinglass.mapper import map_long_short_ratio_to_signal
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from neon_radar.config.intelligence import ProviderConfig
    from neon_radar.domain.market_intelligence.models import IntelligenceSignal, PipelineContext

logger = get_logger(__name__)


@provider_registry.register("CoinGlass")
class CoinGlassProvider(BaseRateLimitedProvider):
    """Fetches market intelligence data from CoinGlass."""

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the CoinGlass provider."""
        # Convert the untyped options dict into a strictly typed CoinGlassConfig
        self.coinglass_config = CoinGlassConfig(**config.options)

        # We can override the base provider's config timeout if CoinGlass requires a different one
        if self.coinglass_config.timeout_seconds:
            config.timeout_seconds = self.coinglass_config.timeout_seconds

        super().__init__(config)

        # Configure the HTTP client with CoinGlass specific headers
        api_key = self.coinglass_config.api_key.get_secret_value()
        if not api_key:
            logger.warning(
                "CoinGlassProvider initialized without API key! Requests will likely fail."
            )

        self._client.headers.update({"accept": "application/json", "coinglassSecret": api_key})

    @property
    def provider_name(self) -> str:
        return "CoinGlass"

    @property
    def provider_type(self) -> str:
        return "Analytics"

    async def fetch_signals(self, context: PipelineContext) -> ProviderResult:
        """Fetch signals from CoinGlass endpoints."""
        signals: list[IntelligenceSignal] = []
        errors = 0
        start_time = time.monotonic()

        # We will fetch the long/short ratio for each configured symbol
        tasks = [
            self._fetch_long_short_ratio(symbol, context.timestamp)
            for symbol in self.coinglass_config.symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Error fetching from CoinGlass: %s", result)
                errors += 1
            elif result is not None:
                signals.append(result)

        latency_ms = (time.monotonic() - start_time) * 1000.0

        quality = DataQuality(
            latency_ms=latency_ms,
            error_count=errors,
            is_stale=False,
        )

        return ProviderResult(signals=tuple(signals), quality=quality)

    async def _fetch_long_short_ratio(
        self, symbol: str, ingestion_timestamp: int
    ) -> IntelligenceSignal | None:
        """Fetch the long/short ratio for a given symbol."""
        url = f"{self.coinglass_config.base_url.rstrip('/')}/api/futures/global-long-short-account-ratio/history"

        # According to API docs, we can pass symbol, exchange, interval
        params = {"exchange": "Binance", "symbol": symbol, "interval": "4h", "limit": 1}

        # Note: _request_with_retry is provided by BaseRateLimitedProvider
        response = await self._request_with_retry(
            method="GET",
            url=url,
            params=params,
            max_retries=self.coinglass_config.retry_count,
            base_backoff_ms=500,
        )

        data = response.json()

        # API might return code != "0" on logical errors (e.g., invalid symbol)
        if data.get("code") != "0":
            logger.warning(
                "CoinGlass API returned error code %s: %s", data.get("code"), data.get("msg")
            )
            return None

        return map_long_short_ratio_to_signal(data, symbol, ingestion_timestamp)
