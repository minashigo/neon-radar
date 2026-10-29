"""Historical funding snapshots and complete settlement events without look-ahead."""

from __future__ import annotations

import bisect
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from neon_radar.domain.funding import FundingRate, FundingSettlement

if TYPE_CHECKING:
    from neon_radar.application.services.market_context_provider import FundingProvider
    from neon_radar.domain.models import Symbol
    from neon_radar.infrastructure.exchanges.binance.client import BinanceClient


class BinanceHistoricalFundingProvider:
    """Fetch settlement pages and retain last-known lookup for signal consumers.

    Event queries require a successfully prefetched covering range. Fetch failures
    propagate rather than silently changing missing funding into zero cost.
    """

    def __init__(
        self,
        client: BinanceClient | None = None,
        funding_provider: FundingProvider | None = None,
    ) -> None:
        self._client = client
        self._funding_provider = funding_provider
        self._rates: dict[str, list[FundingRate]] = {}
        self._coverage: dict[str, tuple[int, int]] = {}

    async def prefetch(self, symbols: tuple[Symbol, ...], start_date: date, end_date: date) -> None:
        """Fetch inclusive UTC dates atomically; page by last returned event time."""
        if end_date < start_date:
            raise ValueError("Funding end date precedes start date")
        if self._client is None and self._funding_provider is None:
            raise ValueError("Funding prefetch requires a data source")
        start_ms = int(
            datetime.combine(start_date, datetime.min.time(), tzinfo=UTC).timestamp() * 1000
        )
        end_ms = int(
            datetime.combine(
                end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC
            ).timestamp()
            * 1000
        )
        fetched: dict[str, list[FundingRate]] = {}
        for sym in symbols:
            cursor = start_ms
            rates: list[FundingRate] = []
            while cursor < end_ms:
                # Bound context-adapter requests as well as REST pages. Even
                # hourly funding fits in a normal 30-day page; a full page still
                # advances by its last event so additional records are not lost.
                page_end = min(cursor + 30 * 24 * 3600 * 1000, end_ms) - 1
                page = await self._fetch_page(sym, cursor, page_end)
                if not page:
                    cursor = page_end + 1
                    continue
                times = [r.timestamp for r in page]
                if any(t is None or not cursor <= t <= page_end for t in times):
                    raise ValueError("Funding page has missing or out-of-range timestamps")
                if times != sorted(set(times)):
                    raise ValueError("Funding page must have unique ascending timestamps")
                rates.extend(page)
                cursor = times[-1] + 1 if len(page) >= 1000 else page_end + 1
            fetched[str(sym)] = rates
        # A failed symbol must not leave the other symbols partially refreshed.
        self._rates.update(fetched)
        self._coverage.update(dict.fromkeys(fetched, (start_ms, end_ms)))

    async def _fetch_page(self, symbol: Symbol, start_ms: int, end_ms: int) -> list[FundingRate]:
        if self._funding_provider is not None:
            series = await self._funding_provider.get_funding_history(
                symbol, start_ms, end_ms, limit=1000
            )
            if series is None:
                raise ValueError(f"Funding history unavailable for {symbol}")
            return [
                FundingRate(
                    symbol=symbol,
                    rate=item.raw_funding,
                    timestamp=item.time_context.event_time,
                    mark_price=item.mark_price,
                )
                for item in series.items
            ]
        assert self._client is not None
        raw_data: Any = await self._client._get_json(  # type: ignore[reportPrivateUsage]
            "/fapi/v1/fundingRate",
            params={"symbol": str(symbol), "startTime": start_ms, "endTime": end_ms, "limit": 1000},
            weight=1,
        )
        if not isinstance(raw_data, list):
            raise ValueError(f"Invalid funding history response for {symbol}")
        return [
            FundingRate(
                symbol=symbol,
                rate=float(raw["fundingRate"]),
                timestamp=int(raw["fundingTime"]),
                mark_price=float(raw["markPrice"]) if raw.get("markPrice") is not None else None,
            )
            for raw in raw_data
        ]

    def get_funding_rate_at(self, symbol: Symbol, timestamp: int) -> FundingRate | None:
        """Return the last published rate, never a future record; not a cash-flow query."""
        rates = self._rates.get(str(symbol), [])
        idx = bisect.bisect_right(rates, timestamp, key=lambda r: r.timestamp)
        return rates[idx - 1] if idx else None

    def get_funding_events(
        self, symbol: Symbol, entry_time: int, exit_time: int
    ) -> tuple[FundingSettlement, ...]:
        """Return actual events in (entry, exit]; never carry a rate forward."""
        if exit_time < entry_time:
            raise ValueError("Funding exit precedes entry")
        if exit_time == entry_time:
            return ()
        coverage = self._coverage.get(str(symbol))
        if coverage is None or entry_time + 1 < coverage[0] or exit_time >= coverage[1]:
            raise ValueError(f"Funding history does not cover requested interval for {symbol}")
        rates = self._rates[str(symbol)]
        left = bisect.bisect_right(rates, entry_time, key=lambda r: r.timestamp)
        right = bisect.bisect_right(rates, exit_time, key=lambda r: r.timestamp)
        events = []
        for rate in rates[left:right]:
            if rate.mark_price is None:
                raise ValueError(f"Missing settlement mark price for {symbol} at {rate.timestamp}")
            events.append(FundingSettlement(symbol, rate.timestamp, rate.rate, rate.mark_price))
        return tuple(events)
