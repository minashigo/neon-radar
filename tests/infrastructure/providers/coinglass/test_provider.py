"""Tests for CoinGlass provider."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from neon_radar.config.intelligence import ProviderConfig
from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType
from neon_radar.domain.market_intelligence.models import PipelineContext
from neon_radar.infrastructure.providers.coinglass.provider import CoinGlassProvider


@pytest.fixture
def provider_config():
    return ProviderConfig(
        enabled=True,
        options={
            "api_key": "test_key",
            "symbols": ["BTCUSDT"],
            "retry_count": 1,  # fast for tests
        },
    )


@pytest.mark.asyncio
async def test_coinglass_provider_fetch_success(provider_config, monkeypatch):
    provider = CoinGlassProvider(provider_config)

    mock_response = httpx.Response(
        status_code=200,
        json={
            "code": "0",
            "msg": "success",
            "data": [
                {
                    "time": 1000,
                    "global_account_long_percent": 60.0,
                    "global_account_short_percent": 40.0,
                    "global_account_long_short_ratio": 1.5,
                }
            ],
        },
        request=httpx.Request("GET", "https://test"),
    )

    async def mock_request(*args, **kwargs):
        return mock_response

    monkeypatch.setattr(provider._client, "request", mock_request)

    context = PipelineContext(timestamp=2000, run_id="test", active_providers=("CoinGlass",))
    result = await provider.fetch_signals(context)

    assert result.quality.error_count == 0
    assert len(result.signals) == 1
    sig = result.signals[0]
    assert sig.type == IntelligenceSignalType.LONG_SHORT_RATIO
    assert sig.direction == 0.2


@pytest.mark.asyncio
async def test_coinglass_provider_fetch_error_handling(provider_config, monkeypatch):
    provider = CoinGlassProvider(provider_config)

    async def mock_request(*args, **kwargs):
        raise httpx.RequestError("Network Error", request=httpx.Request("GET", "https://test"))

    monkeypatch.setattr(provider._client, "request", mock_request)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    context = PipelineContext(timestamp=2000, run_id="test", active_providers=("CoinGlass",))
    result = await provider.fetch_signals(context)

    # Should catch the error and return empty signals, not crash
    assert result.quality.error_count == 1
    assert len(result.signals) == 0


@pytest.mark.asyncio
async def test_coinglass_provider_api_logical_error(provider_config, monkeypatch):
    provider = CoinGlassProvider(provider_config)

    mock_response = httpx.Response(
        status_code=200,
        json={"code": "50001", "msg": "Invalid symbol", "data": []},
        request=httpx.Request("GET", "https://test"),
    )

    async def mock_request(*args, **kwargs):
        return mock_response

    monkeypatch.setattr(provider._client, "request", mock_request)

    context = PipelineContext(timestamp=2000, run_id="test", active_providers=("CoinGlass",))
    result = await provider.fetch_signals(context)

    # API returned an error code, should not crash, should return empty
    assert result.quality.error_count == 0  # Code != 0 doesn't raise exception, just returns None
    assert len(result.signals) == 0
