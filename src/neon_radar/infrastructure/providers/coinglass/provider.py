"""CoinGlass provider implementation."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from neon_radar.application.intelligence.registry import provider_registry
from neon_radar.domain.market_intelligence.models import DataQuality, ProviderResult
from neon_radar.infrastructure.providers.base import BaseRateLimitedProvider
from neon_radar.infrastructure.providers.coinglass.config import CoinGlassConfig
from neon_radar.infrastructure.providers.coinglass.mapper import (
    map_funding_rate_to_signal,
    map_liquidations_to_signal,
    map_long_short_ratio_to_signal,
    map_open_interest_to_signal,
)
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

        self._client.headers.update({"accept": "application/json", "CG-API-KEY": api_key})

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

        # We will fetch all metrics for each configured symbol
        tasks = []
        for symbol in self.coinglass_config.symbols:
            tasks.append(self._fetch_long_short_ratio(symbol, context.timestamp))
            tasks.append(self._fetch_funding_rate(symbol, context.timestamp))
            tasks.append(self._fetch_open_interest(symbol, context.timestamp))
            tasks.append(self._fetch_liquidations(symbol, context.timestamp))

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

    async def _fetch_metric(
        self,
        metric_name: str,
        endpoint: str,
        params: dict,
        mapper: callable,
        symbol: str,
        ingestion_timestamp: int,
    ) -> IntelligenceSignal | None:
        """Centralized method to fetch and map a CoinGlass metric."""
        url = f"{self.coinglass_config.base_url.rstrip('/')}{endpoint}"

        response = await self._request_with_retry(
            method="GET",
            url=url,
            params=params,
            max_retries=self.coinglass_config.retry_count,
            base_backoff_ms=500,
            timeout=self.coinglass_config.timeout_seconds,
        )

        try:
            data = response.json()
        except ValueError as e:
            logger.error("CoinGlass API returned invalid JSON for %s.", metric_name)
            raise ValueError(f"Invalid JSON response from CoinGlass for {metric_name}") from e

        if data.get("code") != "0":
            msg = data.get("msg", "Unknown error")
            code = data.get("code", "unknown")
            logger.error(
                "CoinGlass API returned logical error for %s: %s - %s", metric_name, code, msg
            )
            raise RuntimeError(f"CoinGlass API error: {code} - {msg}")

        signal = mapper(data, symbol, ingestion_timestamp)
        if signal is None:
            logger.warning("Failed to map %s data for symbol %s", metric_name, symbol)

        return signal

    async def _fetch_long_short_ratio(
        self, symbol: str, ingestion_timestamp: int
    ) -> IntelligenceSignal | None:
        """Fetch the long/short ratio for a given symbol."""
        return await self._fetch_metric(
            metric_name="long/short ratio",
            endpoint="/api/futures/global-long-short-account-ratio/history",
            params={"exchange": "Binance", "symbol": symbol, "interval": "4h", "limit": 1},
            mapper=map_long_short_ratio_to_signal,
            symbol=symbol,
            ingestion_timestamp=ingestion_timestamp,
        )

    async def _fetch_funding_rate(
        self, symbol: str, ingestion_timestamp: int
    ) -> IntelligenceSignal | None:
        """Fetch the funding rate for a given symbol."""
        return await self._fetch_metric(
            metric_name="funding rate",
            endpoint="/api/futures/funding-rate/history",
            params={"exchange": "Binance", "symbol": symbol, "interval": "4h", "limit": 1},
            mapper=map_funding_rate_to_signal,
            symbol=symbol,
            ingestion_timestamp=ingestion_timestamp,
        )

    async def _fetch_open_interest(
        self, symbol: str, ingestion_timestamp: int
    ) -> IntelligenceSignal | None:
        """Fetch the open interest for a given symbol."""
        return await self._fetch_metric(
            metric_name="open interest",
            endpoint="/api/futures/open-interest/history",
            params={"exchange": "Binance", "symbol": symbol, "interval": "4h", "limit": 2},
            mapper=map_open_interest_to_signal,
            symbol=symbol,
            ingestion_timestamp=ingestion_timestamp,
        )

    async def _fetch_liquidations(
        self, symbol: str, ingestion_timestamp: int
    ) -> IntelligenceSignal | None:
        """Fetch liquidations for a given symbol."""
        return await self._fetch_metric(
            metric_name="liquidations",
            endpoint="/api/futures/liquidation/history",
            params={"exchange": "Binance", "symbol": symbol, "interval": "4h", "limit": 1},
            mapper=map_liquidations_to_signal,
            symbol=symbol,
            ingestion_timestamp=ingestion_timestamp,
        )
