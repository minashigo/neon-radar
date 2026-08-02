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

    async def mock_request(*args, **kwargs):
        url = str(kwargs.get("url") or args[1])
        if "global-long-short-account-ratio" in url:
            json_data = {
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
            }
        elif "funding-rate" in url:
            json_data = {
                "code": "0",
                "msg": "success",
                "data": [
                    {
                        "time": 1000,
                        "close": "0.001",
                    }
                ],
            }
        elif "open-interest" in url:
            json_data = {
                "code": "0",
                "msg": "success",
                "data": [
                    {
                        "time": 1000,
                        "close": "100000",
                    },
                    {
                        "time": 2000,
                        "close": "105000",
                    },
                ],
            }
        elif "liquidation/history" in url:
            json_data = {
                "code": "0",
                "msg": "success",
                "data": [
                    {
                        "time": 1000,
                        "long_liquidation_usd": "10000",
                        "short_liquidation_usd": "30000",
                    }
                ],
            }
        else:
            json_data = {"code": "1", "msg": "unknown"}

        return httpx.Response(
            status_code=200,
            json=json_data,
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(provider._client, "request", mock_request)

    context = PipelineContext(timestamp=2000, run_id="test", active_providers=("CoinGlass",))
    result = await provider.fetch_signals(context)

    assert result.quality.error_count == 0
    assert len(result.signals) == 4

    types = {sig.type for sig in result.signals}
    assert IntelligenceSignalType.LONG_SHORT_RATIO in types
    assert IntelligenceSignalType.FUNDING in types
    assert IntelligenceSignalType.OPEN_INTEREST in types
    assert IntelligenceSignalType.LIQUIDATIONS in types


@pytest.mark.asyncio
async def test_coinglass_provider_fetch_error_handling(provider_config, monkeypatch):
    provider = CoinGlassProvider(provider_config)

    async def mock_request(*args, **kwargs):
        raise httpx.RequestError("Network Error", request=httpx.Request("GET", "https://test"))

    monkeypatch.setattr(provider._client, "request", mock_request)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    context = PipelineContext(timestamp=2000, run_id="test", active_providers=("CoinGlass",))
    result = await provider.fetch_signals(context)

    # We make 4 requests per symbol, so 4 errors
    assert result.quality.error_count == 4
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

    # API returned an error code, which raises RuntimeError and increments error_count
    assert result.quality.error_count == 4
    assert len(result.signals) == 0


def test_coinglass_provider_auth_header(provider_config):
    provider = CoinGlassProvider(provider_config)
    assert provider._client.headers.get("CG-API-KEY") == "test_key"
    assert provider._client.headers.get("accept") == "application/json"
