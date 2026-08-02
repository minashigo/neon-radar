"""Tests for BaseRateLimitedProvider."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from neon_radar.config.intelligence import ProviderConfig
from neon_radar.infrastructure.providers.base import BaseRateLimitedProvider


@pytest.fixture
def base_provider():
    config = ProviderConfig(enabled=True, options={"retry_count": 2})
    return BaseRateLimitedProvider(config)


@pytest.mark.asyncio
async def test_base_provider_retries_on_timeout(base_provider, monkeypatch):
    call_count = 0

    async def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise httpx.TimeoutException("Timeout")

    monkeypatch.setattr(base_provider._client, "request", mock_request)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    with pytest.raises(RuntimeError, match="Request failed after 3 attempts"):
        await base_provider._request_with_retry("GET", "https://test", max_retries=3)

    assert call_count == 3


@pytest.mark.asyncio
async def test_base_provider_retries_on_5xx(base_provider, monkeypatch):
    call_count = 0

    # 503 Service Unavailable
    mock_response = httpx.Response(status_code=503, request=httpx.Request("GET", "https://test"))

    async def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_response

    monkeypatch.setattr(base_provider._client, "request", mock_request)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    with pytest.raises(RuntimeError, match="Request failed after 3 attempts"):
        await base_provider._request_with_retry("GET", "https://test", max_retries=3)

    assert call_count == 3


@pytest.mark.asyncio
async def test_base_provider_no_retry_on_4xx(base_provider, monkeypatch):
    call_count = 0

    # 400 Bad Request
    mock_response = httpx.Response(status_code=400, request=httpx.Request("GET", "https://test"))

    async def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_response

    monkeypatch.setattr(base_provider._client, "request", mock_request)

    with pytest.raises(RuntimeError, match="Non-retryable HTTP error"):
        await base_provider._request_with_retry("GET", "https://test", max_retries=3)

    # Should only call once, no retries
    assert call_count == 1
