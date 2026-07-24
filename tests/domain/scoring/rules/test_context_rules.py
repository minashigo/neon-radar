import pytest

from neon_radar.config.models import TimeFrame
from neon_radar.domain.market_context import (
    FundingContext,
    FundingSeries,
    HistoricalMarketContext,
    LiquidationContext,
    LiquidationSeries,
    LongShortRatioContext,
    LongShortSeries,
    OpenInterestContext,
    OpenInterestSeries,
    TimeContext,
)
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.scoring.rules.context_flow import (
    LiquidationCascadeRule,
    LongShortCrowdedRule,
)
from neon_radar.domain.scoring.rules.context_funding import FundingTrendRule
from neon_radar.domain.scoring.rules.context_oi import (
    OpenInterestExpansionRule,
)


@pytest.fixture
def base_state():
    symbol = Symbol("BTCUSDT")
    now = 1000000

    # We need a primary series for the OI rules to check price
    candles = [
        OHLCV(open=50000.0, high=50500.0, low=49500.0, close=50100.0, volume=10.0, open_time=now - 5000),
        OHLCV(open=50100.0, high=51000.0, low=50000.0, close=51000.0, volume=20.0, open_time=now),
    ]
    primary_series = KlineSeries(symbol=symbol, timeframe=TimeFrame.M5, candles=tuple(candles))

    return MarketState(
        symbol=symbol,
        timestamp=now,
        primary_series=primary_series
    )

def _tc(i):
    return TimeContext(event_time=i, publish_time=i, ingest_time=i)


def test_funding_trend_rule(base_state):
    symbol = Symbol("BTCUSDT")
    items = tuple(
        FundingContext(
            raw_funding=0.0001 + i * 0.00005,
            funding_8h_equiv=0.0001 + i * 0.00005,
            annualized_apr=0.1,
            mark_price=50000.0,
            next_funding_time_utc=0,
            time_context=_tc(i)
        ) for i in range(5)
    )
    hmc = HistoricalMarketContext(
        symbol=symbol,
        timestamp=1000,
        funding_history=FundingSeries(symbol=symbol, items=items)
    )
    state = base_state
    object.__setattr__(state, "historical_context", hmc)

    rule = FundingTrendRule(window_size=5, trend_threshold=0.0001)
    signal = rule.evaluate(state)

    assert signal is not None
    assert signal.name == "funding_trend"
    # Trend is positive (0.0001 to 0.0003, delta = 0.0002)
    # delta > trend_threshold -> bearish -> value should be negative
    assert signal.value < 0.0


def test_oi_expansion_rule(base_state):
    symbol = Symbol("BTCUSDT")

    # OI expands from 100 to 110 (10%)
    items = tuple(
        OpenInterestContext(
            oi_coin=100.0 + i * 2.5,
            oi_usd_notional=5000000.0,
            time_context=_tc(i)
        ) for i in range(5)
    )
    hmc = HistoricalMarketContext(
        symbol=symbol,
        timestamp=1000,
        open_interest_history=OpenInterestSeries(symbol=symbol, items=items)
    )
    state = base_state
    object.__setattr__(state, "historical_context", hmc)

    rule = OpenInterestExpansionRule(window_size=5, oi_expansion_threshold=0.05, price_move_threshold=0.005)
    signal = rule.evaluate(state)

    assert signal is not None
    assert signal.name == "oi_expansion"
    # Base state has price moving from 50000 to 51000 (2%)
    # OI expanded by 10%
    # Positive price move -> positive signal
    assert signal.value > 0.0


def test_ls_crowded_rule(base_state):
    symbol = Symbol("BTCUSDT")
    items = (
        LongShortRatioContext(
            long_pct=0.8, short_pct=0.2, ls_ratio=4.0, time_context=_tc(1)
        ),
    )
    hmc = HistoricalMarketContext(
        symbol=symbol,
        timestamp=1000,
        ls_ratio_history=LongShortSeries(symbol=symbol, items=items)
    )
    state = base_state
    object.__setattr__(state, "historical_context", hmc)

    rule = LongShortCrowdedRule(extreme_long_ratio=2.5, extreme_short_ratio=0.6)
    signal = rule.evaluate(state)

    assert signal is not None
    assert signal.name == "ls_crowded"
    # Ratio is 4.0 > 2.5 -> crowd is long -> bearish signal
    assert signal.value < 0.0


def test_liquidation_cascade_rule(base_state):
    symbol = Symbol("BTCUSDT")
    items = (
        LiquidationContext(
            long_liquidations=100.0,
            short_liquidations=0.0,
            long_liquidations_usd=6_000_000.0,
            short_liquidations_usd=100_000.0,
            time_context=_tc(1)
        ),
    )
    hmc = HistoricalMarketContext(
        symbol=symbol,
        timestamp=1000,
        liquidations_history=LiquidationSeries(symbol=symbol, items=items)
    )
    state = base_state
    object.__setattr__(state, "historical_context", hmc)

    rule = LiquidationCascadeRule(window_size=3, cascade_threshold_usd=5_000_000.0)
    signal = rule.evaluate(state)

    assert signal is not None
    assert signal.name == "liquidation_cascade"
    # Massive long liquidations ($6M) -> flush out -> bullish signal
    assert signal.value > 0.0
