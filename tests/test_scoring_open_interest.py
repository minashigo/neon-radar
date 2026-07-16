from neon_radar.config.models import TimeFrame
from neon_radar.domain.funding import OpenInterest
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.scoring.rules.open_interest_confirmation import (
    OpenInterestConfirmationRule,
)
from neon_radar.domain.scoring.value_objects import Signal


def make_series(prices, volumes, timeframe=TimeFrame.D1):
    assert len(prices) == len(volumes)
    candles = []
    ts = 1_700_000_000_000
    for p, v in zip(prices, volumes):
        candles.append(OHLCV(open_time=ts, open=p, high=p, low=p, close=p, volume=v))
        ts += 86_400_000
    return KlineSeries(symbol=Symbol("BTCUSDT"), timeframe=timeframe, candles=tuple(candles))


def test_open_interest_confirms_price_move():
    # Price rises; avg quote vol small -> high ratio -> confirmation
    prices = [100.0, 101.0, 102.0]
    volumes = [1.0, 1.0, 1.0]  # base volumes
    series = make_series(prices, volumes)

    latest_close = series.latest().close
    # avg_quote_vol = mean(volume * close) ~ 101 -> set oi_quote high
    avg_quote_vol = sum([c.volume * c.close for c in series.candles]) / len(series.candles)
    oi_quote = avg_quote_vol * 5.0  # ratio = 5.0 (>= high_ratio default 3.0)

    oi = OpenInterest(symbol=Symbol("BTCUSDT"), value=0.0, value_quote=oi_quote)
    state = MarketState(
        symbol=Symbol("BTCUSDT"), timestamp=0, primary_series=series, open_interest=oi
    )

    rule = OpenInterestConfirmationRule()
    sig = rule.evaluate(state)
    assert isinstance(sig, Signal)
    assert sig.value == 0.0
    # high ratio should give high confidence (0.9)
    assert sig.confidence == 0.9


def test_open_interest_diverges_from_price():
    # Price rises but OI small -> low ratio -> reduced confidence
    prices = [100.0, 101.0]
    volumes = [10.0, 10.0]  # larger volumes so avg_quote_vol is larger
    series = make_series(prices, volumes)

    avg_quote_vol = sum([c.volume * c.close for c in series.candles]) / len(series.candles)
    oi_quote = avg_quote_vol * 0.1  # ratio = 0.1 (< low_ratio 0.5)

    oi = OpenInterest(symbol=Symbol("BTCUSDT"), value=0.0, value_quote=oi_quote)
    state = MarketState(
        symbol=Symbol("BTCUSDT"), timestamp=0, primary_series=series, open_interest=oi
    )

    rule = OpenInterestConfirmationRule()
    sig = rule.evaluate(state)
    assert isinstance(sig, Signal)
    assert sig.value == 0.0
    # low ratio should give low confidence (0.25)
    assert sig.confidence == 0.25


def test_open_interest_no_data_returns_none():
    prices = [100.0, 101.0]
    volumes = [1.0, 1.0]
    series = make_series(prices, volumes)
    state = MarketState(
        symbol=Symbol("BTCUSDT"), timestamp=0, primary_series=series, open_interest=None
    )

    rule = OpenInterestConfirmationRule()
    sig = rule.evaluate(state)
    assert sig is None
