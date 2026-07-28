import asyncio

import pytest

from neon_radar.application.intelligence.providers import IntelligenceProvider
from neon_radar.application.intelligence.service import MarketIntelligenceService
from neon_radar.config.intelligence import IntelligenceConfig, ProviderConfig
from neon_radar.domain.market_intelligence.enums import (
    ConsensusDirection,
    IntelligenceSignalType,
    SourceReliability,
)
from neon_radar.domain.market_intelligence.models import SignalEvidence, SignalSource


class MockProvider(IntelligenceProvider):
    def __init__(self, name: str, ptype: str, signals: list[SignalEvidence], delay: float = 0.0):
        self._name = name
        self._type = ptype
        self._signals = signals
        self._delay = delay

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def provider_type(self) -> str:
        return self._type

    async def fetch_signals(self, timestamp: int) -> tuple[SignalEvidence, ...]:
        if self._delay:
            await asyncio.sleep(self._delay)
        return tuple(self._signals)


@pytest.fixture
def config():
    cfg = IntelligenceConfig()
    cfg.providers["Mock1"] = ProviderConfig(enabled=True)
    cfg.providers["Mock2"] = ProviderConfig(enabled=True)
    cfg.providers["MockSlow"] = ProviderConfig(enabled=True, timeout_seconds=0.1)
    cfg.noise_filter.require_independent_confirmation = False
    return cfg


@pytest.fixture
def source():
    return SignalSource("1", "Mock1", "API", SourceReliability.OFFICIAL, 1.0)


@pytest.mark.asyncio
async def test_service_generates_report(config, source):
    sig1 = SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.0, 1000, source)

    provider1 = MockProvider("Mock1", "API", [sig1])

    service = MarketIntelligenceService(config, [provider1])
    report = await service.generate_report()

    assert report.score.direction == ConsensusDirection.BULLISH
    assert len(report.signals) == 1
    assert report.score.coverage > 0


@pytest.mark.asyncio
async def test_service_handles_timeout(config, source):
    # This provider will take 0.5s, but config timeout is 0.1s
    provider_slow = MockProvider("MockSlow", "API", [], delay=0.5)

    service = MarketIntelligenceService(config, [provider_slow])

    # Should not raise exception, just return empty report
    report = await service.generate_report()
    assert report.score.direction == ConsensusDirection.NEUTRAL
    assert len(report.signals) == 0


@pytest.mark.asyncio
async def test_service_disabled_raises(config):
    config.enabled = False
    service = MarketIntelligenceService(config, [])

    with pytest.raises(RuntimeError, match="disabled"):
        await service.generate_report()
