"""Async Binance Futures REST client.

Concretely implements :class:`ExchangeClient` against Binance Futures
public endpoints. No API key required for any of the endpoints we call.

Endpoints used
--------------
* ``GET /fapi/v1/exchangeInfo`` — for :meth:`info`
* ``GET /fapi/v1/klines`` — for :meth:`get_klines`
* ``GET /fapi/v1/ticker/24hr`` — for :meth:`get_ticker`
* ``GET /fapi/v1/premiumIndex`` — for :meth:`get_funding_rate`
* ``GET /fapi/v1/openInterest`` — for :meth:`get_open_interest`

Design notes
------------
* All requests go through ``self._request()`` which centralises:
    - retry on transient failures (network, 429, 5xx)
    - rate-limit token acquisition
    - HTTP status → domain exception mapping
* The underlying :class:`httpx.AsyncClient` is created lazily so
  instantiation is cheap. Callers must invoke :meth:`close` to release
  the connection pool.
* All "weights" used for rate limiting are conservative upper bounds.
  Binance documents them per endpoint; if we get a 429 we will see
  it via the retry mechanism.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from neon_radar.domain.exceptions import (
    NeonRadarError,
    NetworkError,
    ParseError,
    RateLimitError,
    ServerError,
)
from neon_radar.infrastructure.exchanges.base import ExchangeClient, ExchangeInfo
from neon_radar.infrastructure.exchanges.binance.mapper import (
    map_funding_rate_from_premium_index,
    map_klines,
    map_open_interest,
    map_ticker,
)
from neon_radar.infrastructure.exchanges.binance.rate_limiter import (
    RateLimiterConfig,
    TokenBucketRateLimiter,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from neon_radar.config.models import ApiConfig, TimeFrame
    from neon_radar.domain.funding import FundingRate, OpenInterest
    from neon_radar.domain.models import KlineSeries, Symbol, TickerStats

# Conservative weights for our endpoints (Binance documents exact values).
_WEIGHT_KLINES = 2
_WEIGHT_TICKER = 1
_WEIGHT_PREMIUM_INDEX = 1
_WEIGHT_OPEN_INTEREST = 1


class BinanceClient(ExchangeClient):
    """Async client for Binance Futures public REST API."""

    name = "binance"

    def __init__(
        self,
        config: ApiConfig,
        *,
        rate_limiter: TokenBucketRateLimiter | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._rate_limiter = rate_limiter or TokenBucketRateLimiter(
            RateLimiterConfig(max_weight_per_minute=config.rate_limit_per_minute)
        )
        self._sleep = sleep
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=httpx.Timeout(self._config.timeout_seconds),
                headers={"User-Agent": "NeonRadar/0.1"},
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # ExchangeClient interface
    # ------------------------------------------------------------------

    async def info(self) -> ExchangeInfo:
        # We don't actually call the exchange for static info — the
        # fields below are well-known and do not change without a
        # major API version bump. If we ever need live values, fetch
        # /fapi/v1/exchangeInfo here.
        return ExchangeInfo(
            name="binance",
            display_name="Binance Futures",
            website="https://www.binance.com",
            supports_funding=True,
            supports_open_interest=True,
        )

    async def get_klines(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        *,
        limit: int = 500,
        end_time: int | None = None,
    ) -> KlineSeries:
        if limit < 1 or limit > 1500:
            raise ValueError(f"Binance kline limit must be in [1, 1500], got {limit}")
        params: dict[str, Any] = {
            "symbol": str(symbol),
            "interval": timeframe.value,
            "limit": limit,
        }
        if end_time is not None:
            params["endTime"] = end_time

        raw = await self._get_json(
            "/fapi/v1/klines", params=params, weight=_WEIGHT_KLINES
        )
        try:
            return map_klines(raw, symbol=symbol, timeframe=timeframe)
        except (ParseError, ValueError):
            # map_klines already wraps these. Re-raise for upstream handling.
            raise

    async def get_ticker(self, symbol: Symbol) -> TickerStats:
        raw = await self._get_json(
            "/fapi/v1/ticker/24hr",
            params={"symbol": str(symbol)},
            weight=_WEIGHT_TICKER,
        )
        try:
            return map_ticker(raw)
        except (ParseError, ValueError):
            raise

    async def get_funding_rate(self, symbol: Symbol) -> FundingRate:
        raw = await self._get_json(
            "/fapi/v1/premiumIndex",
            params={"symbol": str(symbol)},
            weight=_WEIGHT_PREMIUM_INDEX,
        )
        try:
            return map_funding_rate_from_premium_index(raw, symbol=symbol)
        except (ParseError, ValueError):
            raise

    async def get_open_interest(self, symbol: Symbol) -> OpenInterest:
        raw = await self._get_json(
            "/fapi/v1/openInterest",
            params={"symbol": str(symbol)},
            weight=_WEIGHT_OPEN_INTEREST,
        )
        try:
            return map_open_interest(raw, symbol=symbol)
        except (ParseError, ValueError):
            raise

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    async def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any],
        weight: int,
    ) -> Any:
        """GET ``path`` with retries, rate limiting and error mapping."""
        last_error: NeonRadarError | None = None
        backoff = self._config.retry_backoff_seconds
        for attempt in range(self._config.max_retries + 1):
            await self._rate_limiter.acquire(weight)
            try:
                http = await self._get_http()
                response = await http.get(path, params=params)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = NetworkError(f"Binance network error: {exc}")
                last_error.__cause__ = exc
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise ParseError(
                            f"Binance returned non-JSON for {path}: {exc}"
                        ) from exc
                if response.status_code == 429:
                    last_error = RateLimitError(
                        f"Binance rate limit hit on {path}"
                    )
                    # 429 is special: respect Retry-After if present.
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            await self._sleep(float(retry_after))
                            continue
                        except ValueError:
                            pass
                elif 500 <= response.status_code < 600:
                    last_error = ServerError(
                        f"Binance server error {response.status_code} on {path}"
                    )
                else:
                    # 4xx (other than 429) — do not retry.
                    body = response.text[:200]
                    raise ParseError(
                        f"Binance HTTP {response.status_code} on {path}: {body}"
                    )

            # Backoff before next retry, unless we exhausted retries.
            if attempt < self._config.max_retries:
                await self._sleep(backoff)
                backoff *= 2

        # All retries exhausted.
        assert last_error is not None  # one of the branches above always sets it
        raise last_error
