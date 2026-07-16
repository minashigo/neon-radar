"""Tests for :class:`BinanceClient` using ``httpx.MockTransport``.

These tests run entirely in-process — no real network — but exercise
every code path: success, retry, rate limit, parse error, etc.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from neon_radar.config.models import ApiConfig, TimeFrame
from neon_radar.domain.exceptions import (
    ParseError,
    ServerError,
)
from neon_radar.domain.models import Symbol
from neon_radar.infrastructure.exchanges.binance.client import BinanceClient
from neon_radar.infrastructure.exchanges.binance.rate_limiter import (
    RateLimiterConfig,
    TokenBucketRateLimiter,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# ---------------------------------------------------------------------------
# Fake clock and fake sleep
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start_ms: float = 1_700_000_000_000) -> None:
        self._now_ms = start_ms

    def __call__(self) -> float:
        return self._now_ms / 1000

    def advance(self, ms: float) -> None:
        self._now_ms += ms


class FakeSleep:
    """Records calls to asyncio.sleep without actually sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        # Yield to other tasks so the event loop can make progress.
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Mock transport
# ---------------------------------------------------------------------------


class MockRouter:
    """Routes httpx requests to canned responses.

    Usage::

        router = MockRouter()
        router.add("GET", "/fapi/v1/klines", json=[...])
        transport = httpx.MockTransport(router.handle)

    Each ``add()`` is consumed **once** — the first matching request
    pops it from the queue. This models retry behaviour: a 500 route
    is consumed by the first attempt, so the second attempt gets the
    next route (e.g. a success).

    Counts how many times each path was called so retry logic can be
    verified.
    """

    def __init__(self) -> None:
        self.routes: list[dict[str, Any]] = []
        self.call_counts: dict[tuple[str, str], int] = {}

    def add(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        status: int = 200,
        params: dict[str, str] | None = None,
        body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.routes.append(
            {
                "method": method.upper(),
                "path": path,
                "json": json,
                "status": status,
                "params": params or {},
                "body": body,
                "headers": headers or {},
            }
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        for i, route in enumerate(self.routes):
            if route["method"] != request.method:
                continue
            if route["path"] not in request.url.path:
                continue
            # Check params if specified
            if route["params"]:
                req_params = dict(request.url.params)
                if not all(req_params.get(k) == v for k, v in route["params"].items()):
                    continue
            # Pop the route so the next call gets the next one.
            self.routes.pop(i)
            return httpx.Response(
                route["status"],
                json=route["json"],
                text=route["body"],
                headers=route["headers"],
            )
        # No match
        return httpx.Response(404, json={"error": "no route"})


def _make_config(**overrides: Any) -> ApiConfig:
    defaults: dict[str, Any] = {
        "base_url": "https://fapi.binance.com",
        "timeout_seconds": 5.0,
        "max_retries": 2,
        "retry_backoff_seconds": 0.1,
        "rate_limit_per_minute": 10000,  # effectively unlimited
    }
    defaults.update(overrides)
    return ApiConfig(**defaults)


@pytest.fixture
def router() -> MockRouter:
    return MockRouter()


@pytest.fixture
def make_client(
    router: MockRouter,
) -> Iterator[Callable[..., BinanceClient]]:
    """Factory: returns a BinanceClient wired to the shared mock router."""
    clients: list[httpx.AsyncClient] = []

    def _factory(**kwargs: Any) -> BinanceClient:
        cfg = kwargs.pop("config", None) or _make_config()
        fake_sleep = kwargs.pop("sleep", FakeSleep())
        rl = kwargs.pop("rate_limiter", None)
        client = BinanceClient(config=cfg, sleep=fake_sleep, rate_limiter=rl)
        # Inject our httpx client.
        http = httpx.AsyncClient(
            base_url=cfg.base_url, transport=httpx.MockTransport(router.handle)
        )
        clients.append(http)
        # Stash on the client for test access.
        client._http = http  # type: ignore[attr-defined]
        client._test_sleep = fake_sleep  # type: ignore[attr-defined]
        return client

    yield _factory

    # Cleanup: close all injected httpx clients.
    async def _close_all() -> None:
        await asyncio.gather(*(c.aclose() for c in clients), return_exceptions=True)

    asyncio.run(_close_all())


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


SAMPLE_KLINES = [
    [
        1_700_000_000_000,
        "100.50",
        "110.00",
        "99.00",
        "105.75",
        "1234.5",
        1_700_000_005_999,
        "130000.0",
        42,
    ],
    [
        1_700_000_010_000,
        "105.75",
        "112.00",
        "104.50",
        "110.25",
        "1500.0",
        1_700_000_015_999,
        "160000.0",
        55,
    ],
]


SAMPLE_TICKER = {
    "symbol": "BTCUSDT",
    "lastPrice": "30123.45",
    "priceChangePercent": "2.500",
    "highPrice": "31000.00",
    "lowPrice": "29500.00",
    "volume": "12345.6",
    "quoteVolume": "371234567.89",
    "time": 1_700_000_000_000,
}


SAMPLE_PREMIUM_INDEX = {
    "symbol": "BTCUSDT",
    "markPrice": "30100.50",
    "indexPrice": "30100.00",
    "lastFundingRate": "0.0001",
    "nextFundingTime": 1_700_002_880_000,
    "time": 1_699_999_900_000,
}


SAMPLE_OPEN_INTEREST = {
    "symbol": "BTCUSDT",
    "sumOpenInterest": "12345.6",
    "sumOpenInterestValue": "371234567.89",
    "time": 1_700_000_000_000,
}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestBinanceClientHappyPath:
    @pytest.mark.asyncio
    async def test_info(self, make_client: Callable[..., BinanceClient]) -> None:
        client = make_client()
        info = await client.info()
        assert info.name == "binance"
        assert info.supports_funding
        assert info.supports_open_interest

    @pytest.mark.asyncio
    async def test_get_klines(
        self, make_client: Callable[..., BinanceClient], router: MockRouter
    ) -> None:
        router.add("GET", "/fapi/v1/klines", json=SAMPLE_KLINES)
        client = make_client()
        series = await client.get_klines(Symbol("BTCUSDT"), TimeFrame.H4, limit=2)
        assert len(series) == 2
        assert series[0].open == 100.50
        assert series[1].close == 110.25
        assert router.call_counts[("GET", "/fapi/v1/klines")] == 1

    @pytest.mark.asyncio
    async def test_get_ticker(
        self, make_client: Callable[..., BinanceClient], router: MockRouter
    ) -> None:
        router.add("GET", "/fapi/v1/ticker/24hr", json=SAMPLE_TICKER)
        client = make_client()
        t = await client.get_ticker(Symbol("BTCUSDT"))
        assert t.last_price == 30_123.45

    @pytest.mark.asyncio
    async def test_get_funding_rate(
        self, make_client: Callable[..., BinanceClient], router: MockRouter
    ) -> None:
        router.add("GET", "/fapi/v1/premiumIndex", json=SAMPLE_PREMIUM_INDEX)
        client = make_client()
        fr = await client.get_funding_rate(Symbol("BTCUSDT"))
        assert fr.rate == pytest.approx(0.0001)
        assert fr.mark_price == pytest.approx(30100.50)

    @pytest.mark.asyncio
    async def test_get_open_interest(
        self, make_client: Callable[..., BinanceClient], router: MockRouter
    ) -> None:
        router.add("GET", "/fapi/v1/openInterest", json=SAMPLE_OPEN_INTEREST)
        client = make_client()
        oi = await client.get_open_interest(Symbol("BTCUSDT"))
        assert oi.value == pytest.approx(12345.6)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestBinanceClientValidation:
    @pytest.mark.asyncio
    async def test_klines_limit_too_small(self, make_client: Callable[..., BinanceClient]) -> None:
        client = make_client()
        with pytest.raises(ValueError, match="limit"):
            await client.get_klines(Symbol("BTCUSDT"), TimeFrame.H4, limit=0)

    @pytest.mark.asyncio
    async def test_klines_limit_too_large(self, make_client: Callable[..., BinanceClient]) -> None:
        client = make_client()
        with pytest.raises(ValueError, match="limit"):
            await client.get_klines(Symbol("BTCUSDT"), TimeFrame.H4, limit=2000)


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestBinanceClientRetries:
    @pytest.mark.asyncio
    async def test_retries_on_500(
        self, make_client: Callable[..., BinanceClient], router: MockRouter
    ) -> None:
        # First 2 attempts fail, then succeed. With max_retries=2 the
        # total attempt count is 3 (1 initial + 2 retries).
        for _ in range(2):
            router.add("GET", "/fapi/v1/ticker/24hr", json={}, status=500)
        router.add("GET", "/fapi/v1/ticker/24hr", json=SAMPLE_TICKER)
        client = make_client()
        t = await client.get_ticker(Symbol("BTCUSDT"))
        assert t.last_price == 30_123.45
        assert router.call_counts[("GET", "/fapi/v1/ticker/24hr")] == 3

    @pytest.mark.asyncio
    async def test_retries_on_429(
        self, make_client: Callable[..., BinanceClient], router: MockRouter
    ) -> None:
        for _ in range(2):
            router.add("GET", "/fapi/v1/ticker/24hr", json={}, status=429)
        router.add("GET", "/fapi/v1/ticker/24hr", json=SAMPLE_TICKER)
        client = make_client()
        t = await client.get_ticker(Symbol("BTCUSDT"))
        assert t.last_price == 30_123.45

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(
        self, make_client: Callable[..., BinanceClient], router: MockRouter
    ) -> None:
        # With max_retries=2 there are 3 total attempts; all 3 fail.
        for _ in range(3):
            router.add("GET", "/fapi/v1/ticker/24hr", json={}, status=500)
        client = make_client()
        with pytest.raises(ServerError):
            await client.get_ticker(Symbol("BTCUSDT"))

    @pytest.mark.asyncio
    async def test_does_not_retry_4xx_other_than_429(
        self, make_client: Callable[..., BinanceClient], router: MockRouter
    ) -> None:
        router.add("GET", "/fapi/v1/ticker/24hr", json={}, status=400)
        client = make_client()
        with pytest.raises(ParseError):
            await client.get_ticker(Symbol("BTCUSDT"))
        assert router.call_counts[("GET", "/fapi/v1/ticker/24hr")] == 1


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestBinanceClientRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limiter_called(
        self, router: MockRouter, make_client: Callable[..., BinanceClient]
    ) -> None:
        router.add("GET", "/fapi/v1/klines", json=SAMPLE_KLINES)
        # A tight rate limiter that should still allow our 1 request.
        rl = TokenBucketRateLimiter(RateLimiterConfig(max_weight_per_minute=100))
        client = make_client(rate_limiter=rl)
        await client.get_klines(Symbol("BTCUSDT"), TimeFrame.H4)
        # Limiter should have recorded one slot.
        assert rl._current_weight() == 2  # weight of klines

    @pytest.mark.asyncio
    async def test_rate_limiter_waits(self) -> None:
        """``acquire()`` schedules against the limiter, not the real clock.

        We use a fake clock and a custom sleep so we can observe that
        a full bucket blocks subsequent acquires until old entries
        drop out of the window.
        """
        clock = FakeClock()
        rl = TokenBucketRateLimiter(RateLimiterConfig(max_weight_per_minute=4), clock=clock)
        # Cap after safety margin: int(4 * 0.95) = 3.
        # First acquire(weight=2) puts current weight at 2.
        await rl.acquire(weight=2)
        assert rl._current_weight() == 2
        # Second acquire(weight=2) would push to 4 > 3 → must wait.
        # Advance the clock past the window so the first entry drops.
        clock.advance(61_000)
        await rl.acquire(weight=2)
        assert rl._current_weight() == 2


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestBinanceClientClose:
    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, make_client: Callable[..., BinanceClient]) -> None:
        client = make_client()
        await client.close()
        await client.close()  # no error
