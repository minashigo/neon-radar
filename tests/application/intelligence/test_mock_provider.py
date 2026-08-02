"""Tests for MockProvider."""

import pytest

from neon_radar.application.intelligence.mock_provider import MockProvider
from neon_radar.config.intelligence import ProviderConfig
from neon_radar.domain.market_intelligence.models import PipelineContext


@pytest.mark.asyncio
async def test_mock_provider_creates_valid_signals():
    config = ProviderConfig(enabled=True)
    provider = MockProvider(config)

    context = PipelineContext(timestamp=1000, run_id="test", active_providers=("MockProvider",))

    result = await provider.fetch_signals(context)

    assert result is not None
    assert result.quality.error_count == 0
    assert len(result.signals) == 2

    sig1 = result.signals[0]
    assert sig1.provider_name == "MockProvider"
    assert sig1.provider_type == "Mock"
    assert sig1.direction == 1.0
    assert sig1.strength == 0.8
    assert sig1.weight == 0.5
