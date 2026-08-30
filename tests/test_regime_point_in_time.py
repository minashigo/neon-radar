"""Regression tests for Regime classification Point-in-Time integrity."""


from neon_radar.application.services.analysis import analyze_series
from neon_radar.application.services.regime_classifier import RuleBasedRegimeClassifier
from neon_radar.config.models import TimeFrame
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.trading.regime import RegimeFilterConfig


def build_series(n_candles: int, start_time: int = 1600000000000) -> KlineSeries:
    """Build a synthetic candle series of length n."""
    candles = []
    for i in range(n_candles):
        # Create a strong downtrend so it classifies as BEAR_TREND
        # High volatility
        candles.append(
            OHLCV(
                open_time=start_time + i * 86400000,
                open=50000.0 - i * 50,
                high=51000.0 - i * 50,
                low=49000.0 - i * 50,
                close=49500.0 - i * 50,
                volume=1000.0,
            )
        )
    return KlineSeries(Symbol("BTCUSDT"), TimeFrame("1d"), tuple(candles))


def test_regime_point_in_time_integrity():
    """Verify that future candles do not alter the regime of a historical candle."""

    # Setup the regime classifier
    config = RegimeFilterConfig(
        enabled=True,
        adx_period=14,
        ema_fast_period=9,
        ema_slow_period=21,
        atr_period=14,
    )
    classifier = RuleBasedRegimeClassifier(config)

    # 1. Evaluate regime at T = 100 candles
    series_t = build_series(100)
    result_t = analyze_series(
        series_t,
        rules=[],
        regime_classifier=classifier,
        regime_config=config,
    )

    regime_at_t = result_t.market_state.regime

    # The regime should be BEAR_TREND because of our synthetic data
    # (or at least something that is not UNKNOWN if 30 candles are enough)

    # 2. Add 20 future candles (T = 120 total)
    series_t_plus_20 = build_series(120)

    # Now, evaluate the pipeline AT candle 100 again, but by slicing the 120-candle series
    # In TradeBacktester, it literally does:
    # `context = self.cache.slice(T)`
    # Which yields a KlineSeries of length 30!
    sliced_series = KlineSeries(
        series_t_plus_20.symbol,
        series_t_plus_20.timeframe,
        series_t_plus_20.candles[:100]
    )

    result_sliced = analyze_series(
        sliced_series,
        rules=[],
        regime_classifier=classifier,
        regime_config=config,
    )

    regime_sliced = result_sliced.market_state.regime

    # The regime computed from the sliced series MUST exactly match
    # the regime computed when we only had 30 candles in existence.
    assert regime_at_t == regime_sliced

    # Let's also verify it is BEAR_TREND
    from neon_radar.domain.trading.regime import MarketRegime
    assert regime_at_t == MarketRegime.BEAR_TREND
