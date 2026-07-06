"""Token-bucket rate limiter for the Binance API.

Binance enforces weight-based limits: each endpoint has a "weight"
(from 1 to ~20), and the total weight per minute is capped. This
limiter keeps a sliding 60-second window of consumed weight and
``await``s if a request would exceed the cap.

Design notes
------------
* Pure in-memory state — no I/O. Tests don't need a clock mock if we
  keep durations in milliseconds and expose ``now_ms()`` as a hook.
* The window is a simple list of ``(timestamp_ms, weight)`` tuples,
  pruned on each call. Memory cost is one tuple per request in the
  last 60s; trivial for our usage (we make <100 req/min).
* We expose ``acquire(weight)`` as an async method so callers
  ``await`` it. No callback hell.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(slots=True)
class RateLimiterConfig:
    """Configuration for :class:`TokenBucketRateLimiter`."""

    max_weight_per_minute: int = 1200
    # Safety margin: we stay 5% below the hard cap to avoid 429s.
    safety_margin: float = 0.05


class TokenBucketRateLimiter:
    """Async token-bucket-style rate limiter.

    Parameters
    ----------
    config
        Limits (weight per minute, safety margin).
    clock
        Callable returning current Unix time in milliseconds. Injectable
        for tests; defaults to ``time.time``.
    """

    def __init__(
        self,
        config: RateLimiterConfig | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config or RateLimiterConfig()
        self._clock = clock
        self._window: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    @property
    def config(self) -> RateLimiterConfig:
        return self._config

    def _prune(self, now_ms: float) -> None:
        """Drop entries older than 60s from the window."""
        cutoff = now_ms - 60_000
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _current_weight(self) -> int:
        return sum(weight for _, weight in self._window)

    async def acquire(self, weight: int = 1) -> None:
        """Wait until ``weight`` tokens are available, then consume them.

        Sleeps cooperatively (yields control) until the window has
        enough headroom. Multiple concurrent callers serialise through
        ``self._lock`` so we never oversubscribe.
        """
        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight}")
        async with self._lock:
            while True:
                now_ms = self._clock() * 1000
                self._prune(now_ms)
                used = self._current_weight()
                cap = int(self._config.max_weight_per_minute * (1 - self._config.safety_margin))
                if used + weight <= cap:
                    self._window.append((now_ms, weight))
                    return
                # How long until the oldest entry drops out of the window?
                oldest_ms = self._window[0][0]
                wait_ms = max(0, int(oldest_ms + 60_000 - now_ms)) + 5
                # Release lock and sleep — another caller may consume first.
                self._lock.release()
                try:
                    await asyncio.sleep(wait_ms / 1000)
                finally:
                    await self._lock.acquire()
