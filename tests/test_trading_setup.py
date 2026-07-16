import pytest

from neon_radar.config.models import TimeFrame
from neon_radar.domain.enums import Bias
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.scoring.value_objects import Score
from neon_radar.domain.trading.setup import TradeSetup, TradeSetupEngine


def _make_series(closes: list[float]) -> KlineSeries:
    candles = [
        OHLCV(
            open_time=1000 * i,
            open=c,
            high=c + 5,
            low=c - 5,
            close=c,
            volume=100.0,
        )
        for i, c in enumerate(closes)
    ]
    return KlineSeries(Symbol("BTCUSDT"), TimeFrame.H1, tuple(candles))


def test_trade_setup_validation() -> None:
    # Valid Bullish
    TradeSetup(Bias.BULLISH, 100.0, 90.0, 110.0, 120.0, (1.5, 3.0))

    # Invalid Bullish (SL > Entry)
    with pytest.raises(ValueError):
        TradeSetup(Bias.BULLISH, 100.0, 105.0, 110.0, 120.0, (1.5, 3.0))

    # Valid Bearish
    TradeSetup(Bias.BEARISH, 100.0, 110.0, 90.0, 80.0, (1.5, 3.0))

    # Invalid Bearish (SL < Entry)
    with pytest.raises(ValueError):
        TradeSetup(Bias.BEARISH, 100.0, 90.0, 90.0, 80.0, (1.5, 3.0))

    with pytest.raises(ValueError, match="NEUTRAL"):
        TradeSetup(Bias.NEUTRAL, 100.0, 90.0, 110.0, 120.0, (1.5, 3.0))


def test_trade_setup_engine_bullish() -> None:
    series = _make_series([100.0, 105.0, 110.0])
    from neon_radar.domain.indicators.base import IndicatorKind, IndicatorSeries

    atr_series = IndicatorSeries(
        name="atr_14", kind=IndicatorKind.META, snapshots=tuple([{"atr": 10.0}] * 3)
    )

    state = MarketState(
        symbol=Symbol("BTCUSDT"),
        timestamp=0,
        primary_series=series,
        indicator_series=(atr_series,),
    )

    score = Score(0.5, 0.8, 0.5, 0.0, 2)  # Bullish > 0.2

    engine = TradeSetupEngine(atr_period=14, sl_multiplier=1.5, tp1_rr=1.5, tp2_rr=3.0)
    setup = engine.build_setup(state, score)

    assert setup is not None
    assert setup.direction == Bias.BULLISH
    assert setup.entry_price == 110.0
    # Risk = 10 * 1.5 = 15
    assert setup.stop_loss == 110.0 - 15.0  # 95.0
    assert setup.take_profit_1 == 110.0 + (15.0 * 1.5)  # 132.5
    assert setup.take_profit_2 == 110.0 + (15.0 * 3.0)  # 155.0


def test_trade_setup_engine_bearish() -> None:
    series = _make_series([100.0, 105.0, 110.0])
    from neon_radar.domain.indicators.base import IndicatorKind, IndicatorSeries

    atr_series = IndicatorSeries(
        name="atr_14", kind=IndicatorKind.META, snapshots=tuple([{"atr": 5.0}] * 3)
    )
    state = MarketState(
        symbol=Symbol("BTCUSDT"),
        timestamp=0,
        primary_series=series,
        indicator_series=(atr_series,),
    )

    score = Score(-0.5, 0.8, 0.0, 0.5, 2)  # Bearish < -0.2

    engine = TradeSetupEngine(atr_period=14, sl_multiplier=2.0, tp1_rr=1.0, tp2_rr=2.0)
    setup = engine.build_setup(state, score)

    assert setup is not None
    assert setup.direction == Bias.BEARISH
    assert setup.entry_price == 110.0
    # Risk = 5 * 2.0 = 10
    assert setup.stop_loss == 110.0 + 10.0  # 120.0
    assert setup.take_profit_1 == 110.0 - (10.0 * 1.0)  # 100.0
    assert setup.take_profit_2 == 110.0 - (10.0 * 2.0)  # 90.0


def test_trade_setup_engine_neutral_returns_none() -> None:
    engine = TradeSetupEngine()
    series = _make_series([100.0])
    state = MarketState(
        symbol=Symbol("BTCUSDT"), timestamp=0, primary_series=series, indicator_series=()
    )
    score = Score(0.1, 0.5, 0.1, 0.0, 1)  # Neutral

    assert engine.build_setup(state, score) is None


def test_trade_setup_engine_missing_atr_returns_none() -> None:
    engine = TradeSetupEngine(atr_period=14)
    series = _make_series([100.0])
    state = MarketState(
        symbol=Symbol("BTCUSDT"), timestamp=0, primary_series=series, indicator_series=()
    )
    score = Score(0.5, 0.5, 0.5, 0.0, 1)  # Bullish

    assert engine.build_setup(state, score) is None
