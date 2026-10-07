"""Neon Radar Regime Classification Engine for Futures Grid gating.

Evaluates market regime at candle close strictly point-in-time (ZERO look-ahead bias):
- CHOP: Range-bound / sideways conditions (ADX < 20.0) -> GRID_ALLOWED = True
- BULL_TREND: Upward trending (ADX >= 20.0 and EMA9 > EMA21) -> GRID_ALLOWED = False
- BEAR_TREND: Downward trending (ADX >= 20.0 and EMA9 < EMA21) -> GRID_ALLOWED = False
- VOLATILE_CRASH: Extreme ATR surge (ATR/Price > 8%) -> GRID_ALLOWED = False
- UNKNOWN: Insufficient history or neutral fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from research.experiments.futures_grid.data_loader import (
    df_to_kline_series,
    load_candles_df,
    resample_to_1d,
)

from neon_radar.application.services.analysis import analyze_series
from neon_radar.application.services.regime_classifier import RuleBasedRegimeClassifier
from neon_radar.domain.models import KlineSeries
from neon_radar.domain.trading.regime import MarketRegime, RegimeFilterConfig

CACHE_DIR = Path("research/experiments/futures_grid/cache")


def create_regime_classifier(
    adx_chop_threshold: float = 20.0,
    atr_crash_threshold_pct: float = 0.08,
) -> RuleBasedRegimeClassifier:
    """Creates standard Neon Radar rule-based regime classifier."""
    config = RegimeFilterConfig(
        enabled=True,
        adx_period=14,
        adx_chop_threshold=adx_chop_threshold,
        ema_fast_period=9,
        ema_slow_period=21,
        atr_period=14,
        atr_crash_threshold_pct=atr_crash_threshold_pct,
    )
    return RuleBasedRegimeClassifier(config)


def precompute_regimes(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    warmup_bars: int = 60,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Computes point-in-time regime classification for every candle in df.

    Strict zero look-ahead bias: bar i is evaluated using candles[0:i+1].
    The classification is recorded at bar i close, with execution on bar i+1 open.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{timeframe}_regimes.csv"

    if cache_file.exists() and not force_recompute:
        cached = pd.read_csv(cache_file)
        if len(cached) == len(df):
            return cached

    classifier = create_regime_classifier()
    full_series = df_to_kline_series(symbol, timeframe, df)
    candles = full_series.candles

    results: list[dict[str, Any]] = []

    for i in range(len(candles)):
        t_open = candles[i].open_time
        if i < warmup_bars:
            results.append({
                "open_time_ms": t_open,
                "regime": MarketRegime.UNKNOWN.value,
                "grid_allowed": False,
                "reason": "Warmup",
                "atr": 0.0,
                "atr_pct": 0.0,
                "adx": 0.0,
            })
            continue

        # Point-in-time slice
        sub_candles = candles[max(0, i - 120): i + 1]
        sub_series = KlineSeries(
            symbol=full_series.symbol,
            timeframe=full_series.timeframe,
            candles=tuple(sub_candles),
        )

        try:
            analysis = analyze_series(
                series=sub_series,
                rules=(),  # Only need indicators for regime classification
                regime_classifier=classifier,
                timestamp=t_open,
            )
            regime = analysis.market_state.regime
            reg_val = regime.value if regime else MarketRegime.UNKNOWN.value
            state = analysis.market_state

            atr_val = state.get_indicator_value("atr_14") or 0.0
            latest = state.primary_series.latest()
            close_p = latest.close if latest else 1.0
            atr_pct = (atr_val / close_p) if close_p > 0 else 0.0
            adx_val = state.get_indicator_value("adx_14", "adx") or 0.0

        except Exception:
            reg_val = MarketRegime.UNKNOWN.value
            atr_val = 0.0
            atr_pct = 0.0
            adx_val = 0.0

        # Deterministic Gate: Only CHOP permits grid trading
        grid_allowed = (reg_val == MarketRegime.CHOP.value)

        results.append({
            "open_time_ms": t_open,
            "regime": reg_val,
            "grid_allowed": grid_allowed,
            "reason": f"Regime={reg_val}",
            "atr": atr_val,
            "atr_pct": atr_pct,
            "adx": adx_val,
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(cache_file, index=False)
    return res_df


def get_candles_and_regimes(symbol: str, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads candle data and precomputed regimes for an asset."""
    df_raw = load_candles_df(symbol)
    df = resample_to_1d(df_raw) if timeframe == "1d" else df_raw
    regimes_df = precompute_regimes(symbol, timeframe, df)
    return df, regimes_df
