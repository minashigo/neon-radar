import httpx
import pytest

from neon_radar.config.intelligence import ProviderConfig
from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType
from neon_radar.domain.market_intelligence.models import PipelineContext
from neon_radar.infrastructure.providers.alternative_me.provider import AlternativeMeProvider


@pytest.fixture
def provider_config():
    return ProviderConfig(
        name="AlternativeMe",
        timeout_seconds=2.0,
        max_retries=1,
        options={},
    )


@pytest.mark.asyncio
async def test_alternative_me_provider_success(provider_config, monkeypatch):
    provider = AlternativeMeProvider(provider_config)

    async def mock_request(*args, **kwargs):
        json_data = {
            "name": "Fear and Greed Index",
            "data": [
                {
                    "value": "71",
                    "value_classification": "Greed",
                    "timestamp": "1699920000",
                    "time_until_update": "43200",
                }
            ],
            "metadata": {"error": None},
        }
        return httpx.Response(
            status_code=200,
            json=json_data,
            request=httpx.Request("GET", "https://api.alternative.me/fng/?limit=1"),
        )

    monkeypatch.setattr(provider._client, "request", mock_request)

    context = PipelineContext(timestamp=2000, run_id="test", active_providers=("AlternativeMe",))
    result = await provider.fetch_signals(context)

    assert result.quality.error_count == 0
    assert len(result.signals) == 1

    sig = result.signals[0]
    assert sig.type == IntelligenceSignalType.FEAR_AND_GREED
    assert sig.direction == 0.42  # (71 - 50) / 50 = 21 / 50 = 0.42
    assert sig.strength == 0.42


@pytest.mark.asyncio
async def test_alternative_me_provider_http_error(provider_config, monkeypatch):
    provider = AlternativeMeProvider(provider_config)

    async def mock_request(*args, **kwargs):
        return httpx.Response(status_code=500, request=httpx.Request("GET", "url"))

    monkeypatch.setattr(provider._client, "request", mock_request)

    context = PipelineContext(timestamp=2000, run_id="test", active_providers=("AlternativeMe",))
    result = await provider.fetch_signals(context)

    assert result.quality.error_count == 1
    assert len(result.signals) == 0


@pytest.mark.asyncio
async def test_alternative_me_provider_logical_error(provider_config, monkeypatch):
    provider = AlternativeMeProvider(provider_config)

    async def mock_request(*args, **kwargs):
        json_data = {"data": [{"value": "50"}], "metadata": {"error": "Some API error"}}
        return httpx.Response(status_code=200, json=json_data, request=httpx.Request("GET", "url"))

    monkeypatch.setattr(provider._client, "request", mock_request)

    context = PipelineContext(timestamp=2000, run_id="test", active_providers=("AlternativeMe",))
    result = await provider.fetch_signals(context)

    # API returns 200 OK, but mapper rejects it due to logical error
    assert result.quality.error_count == 1
    assert len(result.signals) == 0



