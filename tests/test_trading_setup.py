import pytest

from neon_radar.config.models import TimeFrame
from neon_radar.domain.enums import Bias
from neon_radar.domain.indicators.base import IndicatorKind, IndicatorSeries
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.scoring.value_objects import AnalysisResult, Score, Signal, SignalCategory
from neon_radar.domain.trading.regime import MarketRegime, RegimeFilterConfig
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
    result = AnalysisResult(score=score, signals=(), summary="", computed_at=0)
    engine = TradeSetupEngine(atr_period=14, sl_multiplier=1.5, tp1_rr=1.5, tp2_rr=3.0)
    setup = engine.build_setup(state, result)

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
    result = AnalysisResult(score=score, signals=(), summary="", computed_at=0)

    engine = TradeSetupEngine(atr_period=14, sl_multiplier=2.0, tp1_rr=1.0, tp2_rr=2.0)
    setup = engine.build_setup(state, result)

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
    result = AnalysisResult(score=score, signals=(), summary="", computed_at=0)

    assert engine.build_setup(state, result) is None


def test_trade_setup_engine_missing_atr_returns_none() -> None:
    engine = TradeSetupEngine(atr_period=14)
    series = _make_series([100.0])
    state = MarketState(
        symbol=Symbol("BTCUSDT"), timestamp=0, primary_series=series, indicator_series=()
    )
    score = Score(0.5, 0.5, 0.5, 0.0, 1)  # Bullish
    result = AnalysisResult(score=score, signals=(), summary="", computed_at=0)

    assert engine.build_setup(state, result) is None


def _create_setup_context(
    direction: Bias,
    local_regime: MarketRegime,
    htf_value: float | None = None,
    regime_config: RegimeFilterConfig | None = None,
) -> tuple[TradeSetupEngine, MarketState, AnalysisResult]:
    series = _make_series([100.0, 105.0, 110.0])
    atr_series = IndicatorSeries(
        name="atr_14", kind=IndicatorKind.META, snapshots=tuple([{"atr": 5.0}] * 3)
    )
    state = MarketState(
        symbol=Symbol("BTCUSDT"),
        timestamp=0,
        primary_series=series,
        indicator_series=(atr_series,),
        regime=local_regime,
    )

    signals: list[Signal] = []
    if htf_value is not None:
        signals.append(
            Signal(
                name="higher_tf_trend",
                weight=0.2,
                value=htf_value,
                confidence=1.0,
                description="HTF trend signal",
                category=SignalCategory.TECHNICAL,
            )
        )

    score_val = 0.5 if direction == Bias.BULLISH else -0.5
    score = Score(
        value=score_val,
        confidence=0.8,
        long_score=0.5 if direction == Bias.BULLISH else 0.0,
        short_score=0.5 if direction == Bias.BEARISH else 0.0,
        contributing_signals=1,
    )
    result = AnalysisResult(
        score=score,
        signals=tuple(signals),
        summary="",
        computed_at=0,
        market_state=state,
    )

    config = regime_config or RegimeFilterConfig(enabled=True)
    engine = TradeSetupEngine(
        atr_period=14,
        min_confidence=0.5,
        regime_config=config,
    )
    return engine, state, result


def test_htf_long_filter_bull_local_bull_htf_allowed() -> None:
    # 1. BULL local + BULL HTF + Long -> allowed
    engine, state, result = _create_setup_context(
        direction=Bias.BULLISH,
        local_regime=MarketRegime.BULL_TREND,
        htf_value=1.0,
    )
    setup = engine.build_setup(state, result)
    assert setup is not None
    assert setup.direction == Bias.BULLISH


def test_htf_long_filter_bull_local_bear_htf_rejected() -> None:
    # 2. BULL local + BEAR HTF + Long -> rejected
    engine, state, result = _create_setup_context(
        direction=Bias.BULLISH,
        local_regime=MarketRegime.BULL_TREND,
        htf_value=-1.0,
    )
    setup = engine.build_setup(state, result)
    assert setup is None


def test_htf_long_filter_bear_local_bear_htf_long_rejected_by_regime() -> None:
    # 3. BEAR local + BEAR HTF + Long -> rejected by regime filter
    engine, state, result = _create_setup_context(
        direction=Bias.BULLISH,
        local_regime=MarketRegime.BEAR_TREND,
        htf_value=-1.0,
    )
    setup = engine.build_setup(state, result)
    assert setup is None


def test_htf_long_filter_bear_local_bull_htf_long_rejected_by_regime() -> None:
    # 4. BEAR local + BULL HTF + Long -> rejected by regime filter
    engine, state, result = _create_setup_context(
        direction=Bias.BULLISH,
        local_regime=MarketRegime.BEAR_TREND,
        htf_value=1.0,
    )
    setup = engine.build_setup(state, result)
    assert setup is None


def test_htf_long_filter_short_setups_behavior_unchanged() -> None:
    # 5. Short setups -> behavior not changed
    # BEAR local + BEAR HTF + Short -> allowed
    engine, state, result = _create_setup_context(
        direction=Bias.BEARISH,
        local_regime=MarketRegime.BEAR_TREND,
        htf_value=-1.0,
    )
    setup = engine.build_setup(state, result)
    assert setup is not None
    assert setup.direction == Bias.BEARISH

    # BEAR local + BULL HTF + Short -> allowed (Short logic untouched)
    engine, state, result = _create_setup_context(
        direction=Bias.BEARISH,
        local_regime=MarketRegime.BEAR_TREND,
        htf_value=1.0,
    )
    setup = engine.build_setup(state, result)
    assert setup is not None
    assert setup.direction == Bias.BEARISH

    # BULL local + BEAR HTF + Short -> rejected by regime filter (unchanged)
    engine, state, result = _create_setup_context(
        direction=Bias.BEARISH,
        local_regime=MarketRegime.BULL_TREND,
        htf_value=-1.0,
    )
    setup = engine.build_setup(state, result)
    assert setup is None


def test_htf_long_filter_missing_or_unknown_htf_data() -> None:
    # 6. Absence of HTF data / UNKNOWN -> does not invent direction
    # BULL local + missing HTF + Long -> allowed
    engine, state, result = _create_setup_context(
        direction=Bias.BULLISH,
        local_regime=MarketRegime.BULL_TREND,
        htf_value=None,
    )
    setup = engine.build_setup(state, result)
    assert setup is not None

    # BULL local + neutral HTF (value == 0) + Long -> allowed
    engine, state, result = _create_setup_context(
        direction=Bias.BULLISH,
        local_regime=MarketRegime.BULL_TREND,
        htf_value=0.0,
    )
    setup = engine.build_setup(state, result)
    assert setup is not None

    # UNKNOWN local + BEAR HTF + Long -> allowed (UNKNOWN regime permitted by default)
    engine, state, result = _create_setup_context(
        direction=Bias.BULLISH,
        local_regime=MarketRegime.UNKNOWN,
        htf_value=-1.0,
    )
    setup = engine.build_setup(state, result)
    assert setup is not None


def test_htf_long_filter_toggle_off() -> None:
    # With filter_htf_bear_on_bull_long=False, setup is allowed even if HTF is BEAR
    config = RegimeFilterConfig(enabled=True, filter_htf_bear_on_bull_long=False)
    engine, state, result = _create_setup_context(
        direction=Bias.BULLISH,
        local_regime=MarketRegime.BULL_TREND,
        htf_value=-1.0,
        regime_config=config,
    )
    setup = engine.build_setup(state, result)
    assert setup is not None
