"""Tests for the Binance rate limiter."""

from __future__ import annotations

import asyncio

import pytest

from neon_radar.infrastructure.exchanges.binance.rate_limiter import (
    RateLimiterConfig,
    TokenBucketRateLimiter,
)


class FakeClock:
    def __init__(self, ms: float = 1_700_000_000_000.0) -> None:
        self._ms = ms

    def __call__(self) -> float:
        return self._ms / 1000

    def advance(self, ms: float) -> None:
        self._ms += ms


class TestTokenBucketRateLimiter:
    @pytest.mark.asyncio
    async def test_single_request_consumes_weight(self) -> None:
        clock = FakeClock()
        rl = TokenBucketRateLimiter(RateLimiterConfig(max_weight_per_minute=100), clock=clock)
        await rl.acquire(weight=5)
        assert rl._current_weight() == 5

    @pytest.mark.asyncio
    async def test_old_entries_drop_out(self) -> None:
        clock = FakeClock()
        rl = TokenBucketRateLimiter(RateLimiterConfig(max_weight_per_minute=100), clock=clock)
        await rl.acquire(weight=50)
        assert rl._current_weight() == 50
        clock.advance(61_000)  # past 60s window
        await rl.acquire(weight=1)
        # Only the new entry remains.
        assert rl._current_weight() == 1

    @pytest.mark.asyncio
    async def test_rejects_non_positive_weight(self) -> None:
        rl = TokenBucketRateLimiter()
        with pytest.raises(ValueError):
            await rl.acquire(weight=0)
        with pytest.raises(ValueError):
            await rl.acquire(weight=-1)

    @pytest.mark.asyncio
    async def test_concurrent_acquires_serialise(self) -> None:
        """Multiple concurrent callers must not exceed the cap."""
        clock = FakeClock()
        rl = TokenBucketRateLimiter(RateLimiterConfig(max_weight_per_minute=100), clock=clock)
        # Schedule 10 concurrent acquires of weight 20. Cap is 95 (with
        # 5% safety margin), so only 4 can succeed without the clock
        # advancing. With our fake sleep we never advance, so all 10
        # will hang. We use a small timeout to verify they DON'T all
        # succeed at once.

        async def acquire_later() -> bool:
            try:
                # Wrap with timeout so the test doesn't hang.
                await asyncio.wait_for(rl.acquire(weight=20), timeout=0.05)
                return True
            except TimeoutError:
                return False

        results = await asyncio.gather(*(acquire_later() for _ in range(10)))
        # 4 acquire immediately (4 * 20 = 80 ≤ 95 cap). The rest time out.
        succeeded = sum(1 for r in results if r)
        assert succeeded == 4
        assert rl._current_weight() == 80

    @pytest.mark.asyncio
    async def test_default_config(self) -> None:
        rl = TokenBucketRateLimiter()
        assert rl.config.max_weight_per_minute == 1200
        assert rl.config.safety_margin == 0.05
