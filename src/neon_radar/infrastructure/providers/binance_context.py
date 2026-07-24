"""Concrete Binance Market Context Providers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from neon_radar.application.services.market_context.cache import ContextCache
from neon_radar.application.services.market_context.normalizers import (
    normalize_binance_funding,
    normalize_binance_funding_history,
    normalize_binance_long_short_ratio,
    normalize_binance_open_interest,
    normalize_binance_open_interest_history,
    normalize_binance_taker_volume,
)
from neon_radar.application.services.market_context_provider import (
    FundingProvider,
    LongShortProvider,
    OpenInterestProvider,
    TakerFlowProvider,
)
from neon_radar.infrastructure.providers.binance_dto import (
    BinanceFundingRateHistoryDTO,
    BinanceLongShortRatioDTO,
    BinanceOpenInterestDTO,
    BinanceOpenInterestHistoryDTO,
    BinancePremiumIndexDTO,
    BinanceTakerVolumeDTO,
)

if TYPE_CHECKING:
    from neon_radar.domain.market_context import (
        FundingContext,
        FundingSeries,
        LongShortRatioContext,
        LongShortSeries,
        OpenInterestContext,
        OpenInterestSeries,
        TakerFlowContext,
        TakerFlowSeries,
    )
    from neon_radar.domain.models import Symbol
    from neon_radar.infrastructure.exchanges.binance_transport import BinanceTransport


class BinanceContextProviders(FundingProvider, OpenInterestProvider, LongShortProvider, TakerFlowProvider):
    """Unified provider class that implements all Binance microstructure context."""

    def __init__(self, transport: BinanceTransport, cache: ContextCache) -> None:
        self._transport = transport
        self._cache = cache

    @property
    def name(self) -> str:
        return "binance_context_provider"

    async def get_funding(self, symbol: Symbol, timestamp: int) -> FundingContext | None:
        cache_key = f"funding_{symbol}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            data = await self._transport.get("/fapi/v1/premiumIndex", {"symbol": str(symbol)})
            dto = BinancePremiumIndexDTO.from_dict(data)
            context = normalize_binance_funding(dto, int(time.time() * 1000))

            # Funding updates roughly every 8h, but premium index updates every minute.
            # We can cache it for 1 minute safely.
            self._cache.set(cache_key, context, ttl_seconds=60.0)
            return context
        except Exception:
            return None

    async def get_open_interest(self, symbol: Symbol, timestamp: int) -> OpenInterestContext | None:
        cache_key = f"oi_{symbol}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            # We need mark price to normalize OI to USD notional
            premium_data = await self._transport.get("/fapi/v1/premiumIndex", {"symbol": str(symbol)})
            mark_price = float(premium_data["markPrice"])

            oi_data = await self._transport.get("/fapi/v1/openInterest", {"symbol": str(symbol)})
            dto = BinanceOpenInterestDTO.from_dict(oi_data)

            context = normalize_binance_open_interest(dto, mark_price, int(time.time() * 1000))
            self._cache.set(cache_key, context, ttl_seconds=60.0)
            return context
        except Exception:
            return None

    async def get_long_short_ratio(self, symbol: Symbol, timestamp: int) -> LongShortRatioContext | None:
        cache_key = f"ls_ratio_{symbol}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            # Use 5m period as default for global account ratio
            data = await self._transport.get("/futures/data/globalLongShortAccountRatio", {
                "symbol": str(symbol),
                "period": "5m",
                "limit": 1
            })
            if not data:
                return None

            dto = BinanceLongShortRatioDTO.from_dict(data[0])
            context = normalize_binance_long_short_ratio(dto, int(time.time() * 1000))

            # This data updates every 5m
            self._cache.set(cache_key, context, ttl_seconds=300.0)
            return context
        except Exception:
            return None

    async def get_taker_flow(self, symbol: Symbol, timestamp: int) -> TakerFlowContext | None:
        cache_key = f"taker_flow_{symbol}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            data = await self._transport.get("/futures/data/takerlongshortRatio", {
                "symbol": str(symbol),
                "period": "5m",
                "limit": 1
            })
            if not data:
                return None

            dto = BinanceTakerVolumeDTO.from_dict(data[0])
            context = normalize_binance_taker_volume(dto, int(time.time() * 1000))

            self._cache.set(cache_key, context, ttl_seconds=300.0)
            return context
        except Exception as e:
            print(f"DEBUG: get_taker_flow failed: {e}")
            return None

    async def get_funding_history(
        self, symbol: Symbol, start_time: int, end_time: int, limit: int = 500
    ) -> FundingSeries | None:
        from neon_radar.domain.market_context import FundingSeries
        cache_key = f"funding_history_{symbol}_{start_time}_{end_time}_{limit}"
        cached = self._cache.get_json(cache_key, FundingSeries.from_dict)
        if cached:
            return cached

        try:
            data = await self._transport.get("/fapi/v1/fundingRate", {
                "symbol": str(symbol),
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit
            })
            if not data:
                return None

            items = []
            ingest_time = int(time.time() * 1000)
            for raw in data:
                dto = BinanceFundingRateHistoryDTO.from_dict(raw)
                items.append(normalize_binance_funding_history(dto, ingest_time))

            series = FundingSeries(symbol=symbol, items=tuple(items))
            self._cache.set_json(cache_key, series, ttl_seconds=3600.0)
            return series
        except Exception:
            return None

    async def get_open_interest_history(
        self, symbol: Symbol, start_time: int, end_time: int, limit: int = 500
    ) -> OpenInterestSeries | None:
        from neon_radar.domain.market_context import OpenInterestSeries
        cache_key = f"oi_history_{symbol}_{start_time}_{end_time}_{limit}"
        cached = self._cache.get_json(cache_key, OpenInterestSeries.from_dict)
        if cached:
            return cached

        try:
            data = await self._transport.get("/futures/data/openInterestHist", {
                "symbol": str(symbol),
                "period": "5m",
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit
            })
            if not data:
                return None

            items = []
            ingest_time = int(time.time() * 1000)
            for raw in data:
                dto = BinanceOpenInterestHistoryDTO.from_dict(raw)
                items.append(normalize_binance_open_interest_history(dto, ingest_time))

            series = OpenInterestSeries(symbol=symbol, items=tuple(items))
            self._cache.set_json(cache_key, series, ttl_seconds=3600.0)
            return series
        except Exception:
            return None

    async def get_long_short_ratio_history(
        self, symbol: Symbol, start_time: int, end_time: int, limit: int = 500
    ) -> LongShortSeries | None:
        from neon_radar.domain.market_context import LongShortSeries
        cache_key = f"ls_history_{symbol}_{start_time}_{end_time}_{limit}"
        cached = self._cache.get_json(cache_key, LongShortSeries.from_dict)
        if cached:
            return cached

        try:
            data = await self._transport.get("/futures/data/globalLongShortAccountRatio", {
                "symbol": str(symbol),
                "period": "5m",
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit
            })
            if not data:
                return None

            items = []
            ingest_time = int(time.time() * 1000)
            for raw in data:
                dto = BinanceLongShortRatioDTO.from_dict(raw)
                items.append(normalize_binance_long_short_ratio(dto, ingest_time))

            series = LongShortSeries(symbol=symbol, items=tuple(items))
            self._cache.set_json(cache_key, series, ttl_seconds=3600.0)
            return series
        except Exception:
            return None

    async def get_taker_flow_history(
        self, symbol: Symbol, start_time: int, end_time: int, limit: int = 500
    ) -> TakerFlowSeries | None:
        from neon_radar.domain.market_context import TakerFlowSeries
        cache_key = f"taker_history_{symbol}_{start_time}_{end_time}_{limit}"
        cached = self._cache.get_json(cache_key, TakerFlowSeries.from_dict)
        if cached:
            return cached

        try:
            data = await self._transport.get("/futures/data/takerlongshortRatio", {
                "symbol": str(symbol),
                "period": "5m",
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit
            })
            if not data:
                return None

            items = []
            ingest_time = int(time.time() * 1000)
            for raw in data:
                dto = BinanceTakerVolumeDTO.from_dict(raw)
                items.append(normalize_binance_taker_volume(dto, ingest_time))

            series = TakerFlowSeries(symbol=symbol, items=tuple(items))
            self._cache.set_json(cache_key, series, ttl_seconds=3600.0)
            return series
        except Exception:
            return None
