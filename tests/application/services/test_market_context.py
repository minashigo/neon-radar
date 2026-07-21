
from neon_radar.application.services.market_context.normalizers import (
    normalize_binance_funding,
    normalize_binance_long_short_ratio,
    normalize_binance_open_interest,
    normalize_binance_taker_volume,
)
from neon_radar.infrastructure.providers.binance_dto import (
    BinanceLongShortRatioDTO,
    BinanceOpenInterestDTO,
    BinancePremiumIndexDTO,
    BinanceTakerVolumeDTO,
)


def test_normalize_binance_funding():
    dto = BinancePremiumIndexDTO(
        symbol="BTCUSDT",
        markPrice="50000.0",
        indexPrice="50000.0",
        estimatedSettlePrice="50000.0",
        lastFundingRate="0.0001",
        nextFundingTime=123456789,
        interestRate="0.0001",
        time=123456000
    )

    ingest_time = 123456100
    context = normalize_binance_funding(dto, ingest_time)

    assert context.raw_funding == 0.0001
    assert context.funding_8h_equiv == 0.0001
    assert context.annualized_apr == 0.0001 * 3 * 365
    assert context.mark_price == 50000.0
    assert context.next_funding_time_utc == 123456789

    assert context.time_context.event_time == 123456000
    assert context.time_context.publish_time == 123456000
    assert context.time_context.ingest_time == ingest_time


def test_normalize_binance_open_interest():
    dto = BinanceOpenInterestDTO(
        symbol="BTCUSDT",
        openInterest="100.5",
        time=123456000
    )

    mark_price = 50000.0
    context = normalize_binance_open_interest(dto, mark_price, 123456100)

    assert context.oi_coin == 100.5
    assert context.oi_usd_notional == 100.5 * 50000.0
    assert context.time_context.event_time == 123456000


def test_normalize_long_short_ratio():
    dto = BinanceLongShortRatioDTO(
        longShortRatio="1.5",
        longAccount="0.60",
        shortAccount="0.40",
        timestamp=123456000
    )

    context = normalize_binance_long_short_ratio(dto, 123456100)
    assert context.long_pct == 0.60
    assert context.short_pct == 0.40
    assert context.ls_ratio == 1.5


def test_normalize_taker_volume():
    dto = BinanceTakerVolumeDTO(
        buySellRatio="1.2",
        buyVol="120.0",
        sellVol="100.0",
        timestamp=123456000
    )

    context = normalize_binance_taker_volume(dto, 123456100)
    assert context.buy_volume == 120.0
    assert context.sell_volume == 100.0
    assert context.net_buy_volume == 20.0
