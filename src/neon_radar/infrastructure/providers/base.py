"""Base rate-limited HTTP provider infrastructure."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from neon_radar.config.intelligence import ProviderConfig


class BaseRateLimitedProvider:
    """Base provider encapsulating HTTP session, retry, and rate limiting logic."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=10.0)

        # Simple concurrent request limiting
        # In a real app, this might be a token bucket (aiolimiter)
        # based on requests_per_minute from config.
        limit = getattr(config, "max_concurrent_requests", 5)
        self._semaphore = asyncio.Semaphore(limit)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def _request_with_retry(
        self, method: str, url: str, max_retries: int = 3, base_backoff_ms: int = 500, **kwargs: Any
    ) -> httpx.Response:
        """Execute an HTTP request with exponential backoff and concurrency limits."""
        async with self._semaphore:
            attempt = 0
            while attempt < max_retries:
                try:
                    response = await self._client.request(method, url, **kwargs)
                    response.raise_for_status()
                    return response
                except httpx.HTTPError as e:
                    # Determine if error is retryable
                    retryable = False
                    if isinstance(e, httpx.RequestError) or (
                        isinstance(e, httpx.HTTPStatusError)
                        and (e.response.status_code == 429 or e.response.status_code >= 500)
                    ):
                        retryable = True

                    if not retryable:
                        raise RuntimeError(f"Non-retryable HTTP error: {e}") from e

                    attempt += 1
                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"Request failed after {max_retries} attempts: {e}"
                        ) from e

                    # Exponential backoff
                    sleep_time = (base_backoff_ms * (2 ** (attempt - 1))) / 1000.0
                    await asyncio.sleep(sleep_time)

            raise RuntimeError("Unexpected end of retry loop")
