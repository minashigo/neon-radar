"""Binance Transport Layer for Market Context.

Handles HTTP requests, rate limiting, and basic retries for Binance Futures API.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from neon_radar.domain.exceptions import NetworkError, RateLimitError, ServerError
from neon_radar.infrastructure.exchanges.binance.rate_limiter import (
    RateLimiterConfig,
    TokenBucketRateLimiter,
)


class BinanceTransport:
    """Centralized HTTP transport for Binance Futures API."""

    def __init__(self, base_url: str, rate_limit_per_minute: int = 2400) -> None:
        self._base_url = base_url
        self._rate_limiter = TokenBucketRateLimiter(
            RateLimiterConfig(max_weight_per_minute=rate_limit_per_minute)
        )
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(10.0),
                headers={"User-Agent": "NeonRadar/0.1"},
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def get(self, endpoint: str, params: dict[str, Any] | None = None, weight: int = 1) -> Any:
        """Execute GET request with rate limiting and basic retry."""
        await self._rate_limiter.acquire(weight)
        client = await self._get_http()

        retries = 3
        for attempt in range(retries):
            try:
                resp = await client.get(endpoint, params=params)

                if resp.status_code == 429:
                    raise RateLimitError("Binance HTTP 429: Rate limit exceeded")
                if resp.status_code >= 500:
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise ServerError(f"Binance Server Error: {resp.status_code}")

                resp.raise_for_status()
                return resp.json()
            except httpx.RequestError as exc:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise NetworkError(f"Network error while calling {endpoint}: {exc}") from exc
