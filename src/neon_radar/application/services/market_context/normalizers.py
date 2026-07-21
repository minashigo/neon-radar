"""Pure functions for normalizing raw DTOs into Market Context domain models."""

from neon_radar.domain.market_context import (
    FundingContext,
    LongShortRatioContext,
    OpenInterestContext,
    TakerFlowContext,
    TimeContext,
)
from neon_radar.infrastructure.providers.binance_dto import (
    BinanceLongShortRatioDTO,
    BinanceOpenInterestDTO,
    BinancePremiumIndexDTO,
    BinanceTakerVolumeDTO,
)


def normalize_binance_funding(dto: BinancePremiumIndexDTO, ingest_time_ms: int) -> FundingContext:
    """Normalize Binance /premiumIndex into FundingContext.
    
    Binance funding is calculated and applied every 8 hours.
    So funding_8h_equiv is equal to raw_funding.
    """
    raw_rate = float(dto.lastFundingRate)
    mark_price = float(dto.markPrice)

    return FundingContext(
        raw_funding=raw_rate,
        funding_8h_equiv=raw_rate,
        annualized_apr=raw_rate * 3 * 365,
        mark_price=mark_price,
        next_funding_time_utc=dto.nextFundingTime,
        time_context=TimeContext(
            event_time=dto.time,
            publish_time=dto.time,
            ingest_time=ingest_time_ms,
        ),
    )


def normalize_binance_open_interest(dto: BinanceOpenInterestDTO, mark_price: float, ingest_time_ms: int) -> OpenInterestContext:
    """Normalize Binance /openInterest into OpenInterestContext.
    
    Binance provides OI in base asset (coin). We derive USD notional using mark_price.
    """
    oi_coin = float(dto.openInterest)
    return OpenInterestContext(
        oi_coin=oi_coin,
        oi_usd_notional=oi_coin * mark_price,
        time_context=TimeContext(
            event_time=dto.time,
            publish_time=dto.time,
            ingest_time=ingest_time_ms,
        ),
    )


def normalize_binance_long_short_ratio(dto: BinanceLongShortRatioDTO, ingest_time_ms: int) -> LongShortRatioContext:
    return LongShortRatioContext(
        long_pct=float(dto.longAccount),
        short_pct=float(dto.shortAccount),
        ls_ratio=float(dto.longShortRatio),
        time_context=TimeContext(
            event_time=dto.timestamp,
            publish_time=dto.timestamp,
            ingest_time=ingest_time_ms,
        ),
    )


def normalize_binance_taker_volume(dto: BinanceTakerVolumeDTO, ingest_time_ms: int) -> TakerFlowContext:
    buy_vol = float(dto.buyVol)
    sell_vol = float(dto.sellVol)
    return TakerFlowContext(
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        net_buy_volume=buy_vol - sell_vol,
        time_context=TimeContext(
            event_time=dto.timestamp,
            publish_time=dto.timestamp,
            ingest_time=ingest_time_ms,
        ),
    )
