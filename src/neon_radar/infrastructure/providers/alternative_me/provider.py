"""Alternative.me provider implementation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from neon_radar.application.intelligence.registry import provider_registry
from neon_radar.domain.market_intelligence.models import DataQuality, ProviderResult
from neon_radar.infrastructure.providers.alternative_me.mapper import map_fng_to_signal
from neon_radar.infrastructure.providers.base import BaseRateLimitedProvider
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from neon_radar.config.intelligence import ProviderConfig
    from neon_radar.domain.market_intelligence.models import PipelineContext

logger = get_logger(__name__)


@provider_registry.register("AlternativeMe")
class AlternativeMeProvider(BaseRateLimitedProvider):
    """Fetches Fear & Greed Index from Alternative.me."""

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the Alternative.me provider."""
        super().__init__(config)
        self._client.headers.update({"accept": "application/json"})

    @property
    def provider_name(self) -> str:
        return "AlternativeMe"

    @property
    def provider_type(self) -> str:
        return "Analytics"

    async def fetch_signals(self, context: PipelineContext) -> ProviderResult:
        """Fetch the Fear & Greed index signal."""
        signals = []
        error_count = 0
        success_count = 0
        latencies = []

        endpoint = "https://api.alternative.me/fng/?limit=1"
        start_time = time.monotonic()

        try:
            response = await self._request_with_retry(method="GET", url=endpoint)

            data = response.json()
            signal = map_fng_to_signal(data, ingestion_timestamp=context.timestamp)

            if signal:
                signals.append(signal)
                success_count += 1
            else:
                # API responded but data was invalid/missing
                error_count += 1

        except ValueError:
            logger.error("Alternative.me API returned invalid JSON.")
            error_count += 1
        except Exception as e:
            logger.error(f"Failed to fetch Alternative.me data: {e}")
            error_count += 1

        latency = (time.monotonic() - start_time) * 1000
        latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        quality = DataQuality(
            latency_ms=avg_latency,
            error_count=error_count,
            is_stale=False,
        )

        return ProviderResult(signals=tuple(signals), quality=quality)
