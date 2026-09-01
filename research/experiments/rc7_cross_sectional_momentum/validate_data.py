import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "MATICUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "SOLUSDT",
    "UNIUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "ATOMUSDT",
]


def validate_data():
    data_dir = Path("research/experiments/rc7_cross_sectional_momentum/data")

    missing_symbols = []
    report = []

    for sym in UNIVERSE:
        p = data_dir / f"{sym}.csv"
        if not p.exists():
            missing_symbols.append(sym)
            continue

        df = pd.read_csv(p, parse_dates=["open_time", "close_time"])

        # Check sorted
        if not df["open_time"].is_monotonic_increasing:
            raise ValueError(f"{sym}: Timestamps are not sorted!")

        # Check duplicates
        if df["open_time"].duplicated().any():
            raise ValueError(f"{sym}: Contains duplicate timestamps!")

        # Basic stats
        st = df["open_time"].min()
        et = df["open_time"].max()
        count = len(df)

        # Check missing periods
        # For 4h candles, diff should be exactly 14400000 ms
        diffs = df["open_time"].diff().dt.total_seconds() / 3600
        gaps = (diffs > 4.0).sum()

        status = "Active"
        if et < pd.to_datetime("2026-08-01", utc=True) - pd.Timedelta(days=7):
            status = "Delisted/Inactive"

        report.append(
            {
                "symbol": sym,
                "earliest_candle": st,
                "latest_candle": et,
                "count": count,
                "gaps": gaps,
                "status": status,
            }
        )

    if missing_symbols:
        logger.error(f"Missing symbols entirely: {missing_symbols}")
        raise ValueError("Not all 15 symbols are present!")

    rep_df = pd.DataFrame(report)
    print("\nData Availability Report:")
    print(rep_df.to_string(index=False))


if __name__ == "__main__":
    validate_data()
