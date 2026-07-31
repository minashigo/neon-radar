"""Mock provider for testing the Market Intelligence Pipeline."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from neon_radar.application.intelligence.registry import provider_registry
from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType
from neon_radar.domain.market_intelligence.models import (
    DataQuality,
    IntelligenceSignal,
    ProviderResult,
)

if TYPE_CHECKING:
    from neon_radar.config.intelligence import ProviderConfig
    from neon_radar.domain.market_intelligence.models import PipelineContext


@provider_registry.register("mock")
class MockProvider:
    """A provider that generates mock signals without making network requests."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def provider_name(self) -> str:
        return "MockProvider"

    @property
    def provider_type(self) -> str:
        return "Mock"

    async def fetch_signals(self, context: PipelineContext) -> ProviderResult:
        """Generate some fake signals based on the context timestamp."""
        # Simulate network latency
        await asyncio.sleep(0.1)

        event_ts = context.timestamp - 1000

        sig1 = IntelligenceSignal(
            type=IntelligenceSignalType.RSI,
            direction=1.0,
            strength=0.8,
            event_timestamp=event_ts,
            ingestion_timestamp=context.timestamp,
            source_id="mock_source_1",
        )

        sig2 = IntelligenceSignal(
            type=IntelligenceSignalType.SOCIAL_VOLUME,
            direction=0.5,
            strength=0.5,
            event_timestamp=event_ts,
            ingestion_timestamp=context.timestamp,
            source_id="mock_source_1",
        )

        quality = DataQuality(
            latency_ms=100.0,
            error_count=0,
            is_stale=False,
        )

        return ProviderResult(signals=(sig1, sig2), quality=quality)
