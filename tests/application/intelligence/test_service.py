import asyncio

import pytest

from neon_radar.application.intelligence.providers import IntelligenceProvider
from neon_radar.application.intelligence.registry import provider_registry
from neon_radar.application.intelligence.service import MarketIntelligenceService
from neon_radar.config.intelligence import IntelligenceConfig, ProviderConfig
from neon_radar.domain.market_intelligence.enums import (
    ConsensusDirection,
    IntelligenceSignalType,
    SourceReliability,
)
from neon_radar.domain.market_intelligence.models import (
    DataQuality,
    IntelligenceSignal,
    PipelineContext,
    ProviderResult,
)


def create_signal(provider_name: str) -> IntelligenceSignal:
    return IntelligenceSignal(
        type=IntelligenceSignalType.RSI,
        direction=1.0,
        strength=1.0,
        event_timestamp=1000,
        ingestion_timestamp=1010,
        source_id=f"{provider_name}_id",
        provider_name=provider_name,
        provider_type="API",
        reliability=SourceReliability.OFFICIAL,
        weight=1.0,
    )


class BaseTestMockProvider(IntelligenceProvider):
    def __init__(self, config: ProviderConfig, name: str, delay: float = 0.0):
        self._config = config
        self._name = name
        self._delay = delay

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def provider_type(self) -> str:
        return "API"

    async def fetch_signals(self, context: PipelineContext) -> ProviderResult:
        if self._delay:
            await asyncio.sleep(self._delay)

        sig = create_signal(self._name)
        quality = DataQuality(latency_ms=10.0, error_count=0, is_stale=False)
        return ProviderResult(signals=(sig,), quality=quality)

    async def close(self) -> None:
        pass


@provider_registry.register("Mock1")
class MockProvider1(BaseTestMockProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config, "Mock1")


@provider_registry.register("MockSlow")
class MockSlowProvider(BaseTestMockProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config, "MockSlow", delay=0.5)


@pytest.fixture
def config():
    cfg = IntelligenceConfig()
    cfg.providers["Mock1"] = ProviderConfig(enabled=True)
    cfg.providers["MockSlow"] = ProviderConfig(enabled=True, timeout_seconds=0.1)
    cfg.noise_filter.require_independent_confirmation = False
    return cfg


@pytest.mark.asyncio
async def test_service_generates_report(config):
    # Disable slow provider for this test
    config.providers["MockSlow"].enabled = False

    service = MarketIntelligenceService(config)
    report = await service.generate_report()

    assert report.score.direction == ConsensusDirection.BULLISH
    assert len(report.signals) == 1
    assert report.score.coverage > 0


@pytest.mark.asyncio
async def test_service_handles_timeout(config):
    # Disable Mock1, only keep MockSlow
    config.providers["Mock1"].enabled = False

    service = MarketIntelligenceService(config)

    # Should not raise exception, just return empty report
    report = await service.generate_report()
    assert report.score.direction == ConsensusDirection.NEUTRAL
    assert len(report.signals) == 0


@pytest.mark.asyncio
async def test_service_disabled_raises(config):
    config.enabled = False
    service = MarketIntelligenceService(config)

    with pytest.raises(RuntimeError, match="disabled"):
        await service.generate_report()
