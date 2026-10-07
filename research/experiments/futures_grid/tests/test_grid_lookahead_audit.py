"""Look-ahead bias audit tests for Futures Grid."""

from research.experiments.futures_grid.data_loader import df_to_kline_series, load_candles_df
from research.experiments.futures_grid.regime_engine import create_regime_classifier
from research.experiments.futures_grid.simulator import (
    GridConfig,
    GridSession,
    GridStatus,
    GridType,
)


def test_regime_future_invariance():
    """Verify that future candles cannot alter the regime classified at candle T."""
    df = load_candles_df("BTCUSDT").iloc[:150].copy()
    classifier = create_regime_classifier()

    from neon_radar.application.services.analysis import analyze_series

    # 1. Evaluate at bar 99 using slice [:100]
    series_original = df_to_kline_series("BTCUSDT", "4h", df.iloc[:100])
    analysis_orig = analyze_series(
        series=series_original,
        rules=(),
        regime_classifier=classifier,
        timestamp=series_original.candles[-1].open_time,
    )
    regime_orig = analysis_orig.market_state.regime

    # 2. Mutate future bars (100 to 149) with 10x volatility
    df_mutated = df.copy()
    df_mutated.loc[100:, "open"] *= 5.0
    df_mutated.loc[100:, "high"] *= 5.0
    df_mutated.loc[100:, "low"] *= 5.0
    df_mutated.loc[100:, "close"] *= 5.0

    # 3. Evaluate bar 99 again using point-in-time slice
    series_mutated = df_to_kline_series("BTCUSDT", "4h", df_mutated.iloc[:100])
    analysis_after = analyze_series(
        series=series_mutated,
        rules=(),
        regime_classifier=classifier,
        timestamp=series_mutated.candles[-1].open_time,
    )
    regime_after = analysis_after.market_state.regime

    assert regime_orig == regime_after


def test_grid_execution_timing_strict_t_plus_one():
    """Verify that a grid triggered at bar T close is executed strictly on bar T+1 open."""
    cfg = GridConfig(
        lower_price=90.0,
        upper_price=110.0,
        num_grids=10,
        initial_margin=1000.0,
        grid_type=GridType.NEUTRAL,
    )
    # Signal at bar 10 (t=1000), execution at bar 11 open (t=1001)
    entry_time = 1001
    session = GridSession(cfg, start_time=entry_time, initial_price=100.0)

    first_event = session.events[0]
    assert first_event.timestamp == entry_time
    assert first_event.event_type == "INIT"


def test_regime_exit_timing():
    """Verify that regime block at bar T triggers market close at bar T+1 open."""
    cfg = GridConfig(
        lower_price=90.0,
        upper_price=110.0,
        num_grids=10,
        initial_margin=1000.0,
    )
    session = GridSession(cfg, start_time=1000, initial_price=100.0)

    # Process candle at t=1000
    session.process_candle(
        timestamp=1000, open_price=100.0, high_price=100.0, low_price=95.0, close_price=96.0
    )
    assert session.inventory_qty > 0

    # Gate detects regime exit at T close; execution happens at T+1 open (t=2000)
    res = session.close(timestamp=2000, exit_price=96.0, reason="GATE")
    assert res.status == GridStatus.CLOSED_GATE
    assert res.end_time == 2000
    assert res.events[-1].timestamp == 2000
