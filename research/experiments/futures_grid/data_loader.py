"""Historical market data loader and funding rate generator for Futures Grid research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from neon_radar.config.models import TimeFrame
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol

DATA_DIR = Path("research/experiments/rc7_cross_sectional_momentum/data")


def load_candles_df(symbol: str) -> pd.DataFrame:
    """Load raw historical 4h candle data for a given symbol."""
    path = DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["open_time"], utc=True)
    df["open_time_ms"] = (df["dt"].astype("int64") // 1000).astype("int64")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = df.sort_values("open_time_ms").reset_index(drop=True)
    return df


def resample_to_1d(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Resample 4h candles DataFrame to 1d candles."""
    df = df_4h.copy()
    df.set_index("dt", inplace=True)

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "open_time_ms": "first",
    }
    df_1d = df.resample("1D").agg(agg_rules).dropna().reset_index()
    return df_1d


def df_to_kline_series(symbol: str, timeframe: str, df: pd.DataFrame) -> KlineSeries:
    """Converts DataFrame to domain KlineSeries."""
    candles = []
    for _, row in df.iterrows():
        c = OHLCV(
            open_time=int(row["open_time_ms"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        candles.append(c)

    return KlineSeries(
        symbol=Symbol(symbol),
        timeframe=TimeFrame(timeframe),
        candles=tuple(candles),
    )


def df_to_candle_dicts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Converts DataFrame to list of candle dictionaries for fast simulator execution."""
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "open_time": int(row["open_time_ms"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    return records


def build_funding_rate_map(
    candles_df: pd.DataFrame,
    default_annual_rate: float = 0.1095,  # ~10.95% APR standard neutral/bullish crypto baseline (0.01% per 8h)
) -> dict[int, float]:
    """Generates 8-hour funding rates keyed by candle open_time_ms.

    Funding occurs at 00:00, 08:00, 16:00 UTC.
    Baseline 8h rate = 0.0001 (0.01%).
    """
    funding_map = {}
    base_8h_rate = default_annual_rate / (365 * 3)  # ~0.0001 (0.01%)

    for _, row in candles_df.iterrows():
        t_ms = int(row["open_time_ms"])
        dt = row["dt"] if "dt" in row else pd.to_datetime(t_ms, unit="ms", utc=True)
        # Check if candle open matches 8h mark: 00:00, 08:00, 16:00 UTC
        if dt.hour in (0, 8, 16) and dt.minute == 0:
            funding_map[t_ms] = base_8h_rate

    return funding_map
