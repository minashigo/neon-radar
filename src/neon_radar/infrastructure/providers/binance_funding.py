"""Historical funding rate provider for backtesting without look-ahead bias."""

from __future__ import annotations

import bisect
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from neon_radar.domain.funding import FundingRate
from neon_radar.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from neon_radar.application.services.market_context_provider import FundingProvider
    from neon_radar.domain.models import Symbol
    from neon_radar.infrastructure.exchanges.binance.client import BinanceClient


class BinanceHistoricalFundingProvider:
    """Historical funding provider implementing HistoricalFundingProvider protocol.

    Can operate via an active BinanceClient (REST API) or a FundingProvider instance.
    """

    def __init__(
        self,
        client: BinanceClient | None = None,
        funding_provider: FundingProvider | None = None,
    ) -> None:
        self._client = client
        self._funding_provider = funding_provider
        self._rates: dict[str, list[FundingRate]] = {}

    async def prefetch(
        self, symbols: tuple[Symbol, ...], start_date: date, end_date: date
    ) -> None:
        """Prefetch historical funding rates for symbols across the date range."""
        start_ms = int(
            datetime.combine(start_date, datetime.min.time(), tzinfo=UTC).timestamp()
            * 1000
        )
        end_ms = int(
            datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()
            * 1000
        )

        for sym in symbols:
            sym_str = str(sym)
            rates: list[FundingRate] = []

            if self._funding_provider is not None:
                try:
                    series = await self._funding_provider.get_funding_history(
                        sym, start_ms, end_ms, limit=1000
                    )
                    if series and series.items:
                        rates = [
                            FundingRate(symbol=sym, rate=item.rate, timestamp=item.timestamp)
                            for item in series.items
                        ]
                except Exception as exc:
                    logger.warning(f"Failed to fetch funding history for {sym_str}: {exc}")

            elif self._client is not None:
                chunk_size_ms = 30 * 24 * 60 * 60 * 1000  # 30 days
                current_start = start_ms
                while current_start < end_ms:
                    current_end = min(current_start + chunk_size_ms, end_ms)
                    try:
                        raw_data: Any = await self._client._get_json(  # type: ignore[reportPrivateUsage]
                            "/fapi/v1/fundingRate",
                            params={
                                "symbol": sym_str,
                                "startTime": current_start,
                                "endTime": current_end,
                                "limit": 1000,
                            },
                            weight=1,
                        )
                        if raw_data:
                            for raw in raw_data:
                                rates.append(
                                    FundingRate(
                                        symbol=sym,
                                        rate=float(raw["fundingRate"]),
                                        timestamp=int(raw["fundingTime"]),
                                    )
                                )
                    except Exception as exc:
                        logger.warning(f"Failed to fetch funding rate for {sym_str}: {exc}")
                        break
                    current_start = current_end + 1

            rates.sort(key=lambda r: r.timestamp)
            self._rates[sym_str] = rates

    def get_funding_rate_at(self, symbol: Symbol, timestamp: int) -> FundingRate | None:
        """Returns the funding rate active at or immediately before timestamp."""
        rates = self._rates.get(str(symbol))
        if not rates:
            return None

        ts_list = [r.timestamp for r in rates]
        idx = bisect.bisect_right(ts_list, timestamp)
        if idx > 0:
            return rates[idx - 1]

        # If timestamp is within 8h before the first recorded rate, return that first rate
        if rates and abs(rates[0].timestamp - timestamp) <= 8 * 3600 * 1000:
            return rates[0]

        return None
