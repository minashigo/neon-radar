"""CSV exporter for simulated trades."""

import csv
from collections.abc import Iterable
from pathlib import Path

from neon_radar.domain.trading.backtest import Trade


def export_trades_to_csv(trades: Iterable[Trade], filepath: Path) -> None:
    """Export a list of trades to a CSV file.

    Columns: Symbol, Direction, Entry Time, Exit Time, Entry Price,
    Exit Price, Exit Reason, PnL (%), Holding Time.
    """
    with filepath.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Symbol",
                "Direction",
                "Entry Time",
                "Exit Time",
                "Entry Price",
                "Exit Price",
                "Exit Reason",
                "PnL (%)",
                "Holding Time (ms)",
            ]
        )
        for t in trades:
            holding_time = ""
            if t.exit_time is not None:
                holding_time = str(t.exit_time - t.entry_time)

            ep = f"{t.entry_price:.4f}"
            xp = f"{t.exit_price:.4f}" if t.exit_price is not None else ""
            pnl = f"{t.pnl_pct * 100:.2f}"

            writer.writerow(
                [
                    str(t.symbol),
                    t.direction.name,
                    t.entry_time,
                    t.exit_time if t.exit_time else "",
                    ep,
                    xp,
                    t.exit_reason.value,
                    pnl,
                    holding_time,
                ]
            )
