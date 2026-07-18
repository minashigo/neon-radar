"""CSV exporter for simulated trades."""

import csv
from collections.abc import Iterable
from pathlib import Path

from neon_radar.domain.trading.backtest import Trade


def export_trades_to_csv(trades: Iterable[Trade], filepath: Path) -> None:
    """Export a list of trades to a CSV file.

    Columns: Symbol, Direction, Entry Time, Exit Time, Entry Price,
    Exit Price, Exit Reason, Gross PnL (%), Fees (%), Slippage (%), 
    Funding (%), Execution Costs (%), Net PnL (%), Holding Time.
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
                "Gross PnL (%)",
                "Fees (%)",
                "Slippage (%)",
                "Funding (%)",
                "Execution Costs (%)",
                "Net PnL (%)",
                "Holding Time (ms)",
            ]
        )
        for t in trades:
            holding_time = ""
            if t.exit_time is not None:
                holding_time = str(t.exit_time - t.entry_time)

            ep = f"{t.entry_price:.4f}"
            xp = f"{t.exit_price:.4f}" if t.exit_price is not None else ""
            gross_pnl = f"{t.gross_pnl_pct * 100:.2f}"

            if t.costs:
                fees = f"{t.costs.fees_pct * 100:.3f}"
                slippage = f"{t.costs.slippage_pct * 100:.3f}"
                funding = f"{t.costs.funding_pct * 100:.3f}"
                total_costs = f"{t.costs.total_costs_pct * 100:.3f}"
            else:
                fees = slippage = funding = total_costs = "0.000"

            net_pnl = f"{t.net_pnl_pct * 100:.2f}"

            writer.writerow(
                [
                    str(t.symbol),
                    t.direction.name,
                    t.entry_time,
                    t.exit_time if t.exit_time else "",
                    ep,
                    xp,
                    t.exit_reason.value,
                    gross_pnl,
                    fees,
                    slippage,
                    funding,
                    total_costs,
                    net_pnl,
                    holding_time,
                ]
            )
