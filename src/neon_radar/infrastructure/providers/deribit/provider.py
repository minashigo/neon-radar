"""Deribit Market Intelligence provider."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from neon_radar.application.intelligence.registry import provider_registry
from neon_radar.domain.market_intelligence.models import DataQuality, ProviderResult
from neon_radar.infrastructure.providers.base import BaseRateLimitedProvider
from neon_radar.infrastructure.providers.deribit.mapper import (
    map_dvol_to_signal,
    map_put_call_ratio_to_signal,
)

if TYPE_CHECKING:
    from neon_radar.config.intelligence import ProviderConfig
    from neon_radar.domain.market_intelligence.models import (
        IntelligenceSignal,
        PipelineContext,
    )

logger = logging.getLogger(__name__)


@provider_registry.register("Deribit")
class DeribitProvider(BaseRateLimitedProvider):
    """Fetches public market intelligence from Deribit (DVOL, Put/Call Ratio)."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = "https://www.deribit.com"
        self._client.headers.update({"accept": "application/json"})

    @property
    def provider_name(self) -> str:
        return "Deribit"

    @property
    def provider_type(self) -> str:
        return "Exchange"

    async def fetch_signals(self, context: PipelineContext) -> ProviderResult:
        """Fetch signals from Deribit endpoints."""
        signals: list[IntelligenceSignal] = []
        errors = 0
        start_time = time.monotonic()

        tasks = [
            self._fetch_dvol(context.timestamp),
            self._fetch_put_call_ratio(context.timestamp),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Error fetching from Deribit: %s", result)
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

    async def _fetch_dvol(self, ingestion_timestamp: int) -> IntelligenceSignal | None:
        """Fetch the DVOL index for BTC."""
        # Get data for the last 24 hours to ensure we get the latest daily candle
        now = ingestion_timestamp
        past = now - 86400 * 1000
        url = f"{self.base_url}/api/v2/public/get_volatility_index_data"
        params = {
            "currency": "BTC",
            "start_timestamp": past,
            "end_timestamp": now,
            "resolution": "1D",
        }

        response = await self._request_with_retry(
            method="GET",
            url=url,
            params=params,
            max_retries=1,  # Public endpoints can be strict, we keep retries low
            base_backoff_ms=500,
            timeout=self.config.timeout_seconds,
        )

        try:
            data = response.json()
        except ValueError as e:
            logger.error("Deribit API returned invalid JSON for DVOL.")
            raise ValueError("Invalid JSON response from Deribit") from e

        # Deribit wraps responses in a standard JSON-RPC format with "result" or "error"
        if "error" in data:
            code = data["error"].get("code", "unknown")
            msg = data["error"].get("message", "Unknown error")
            logger.error("Deribit API returned logical error for DVOL: %s - %s", code, msg)
            raise RuntimeError(f"Deribit API error: {code} - {msg}")

        signal = map_dvol_to_signal(data, "BTC", ingestion_timestamp)
        if signal is None:
            logger.warning("Failed to map DVOL data for symbol BTC")

        return signal

    async def _fetch_put_call_ratio(
        self, ingestion_timestamp: int
    ) -> IntelligenceSignal | None:
        """Fetch the Put/Call Ratio from options book summary for BTC."""
        url = f"{self.base_url}/api/v2/public/get_book_summary_by_currency"
        params = {
            "currency": "BTC",
            "kind": "option",
        }

        response = await self._request_with_retry(
            method="GET",
            url=url,
            params=params,
            max_retries=1,
            base_backoff_ms=500,
            timeout=self.config.timeout_seconds,
        )

        try:
            data = response.json()
        except ValueError as e:
            logger.error("Deribit API returned invalid JSON for Put/Call Ratio.")
            raise ValueError("Invalid JSON response from Deribit") from e

        if "error" in data:
            code = data["error"].get("code", "unknown")
            msg = data["error"].get("message", "Unknown error")
            logger.error("Deribit API returned logical error for Put/Call Ratio: %s - %s", code, msg)
            raise RuntimeError(f"Deribit API error: {code} - {msg}")

        signal = map_put_call_ratio_to_signal(data, "BTC", ingestion_timestamp)
        if signal is None:
            logger.warning("Failed to map Put/Call Ratio data for symbol BTC")

        return signal
