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
                "Entry Reason",
                "Exit Reason",
                "Gross PnL (%)",
                "Fees (%)",
                "Slippage (%)",
                "Funding (%)",
                "Execution Costs (%)",
                "Net PnL (%)",
                "Holding Time (ms)",
                "ADX",
                "ATR",
                "RSI",
                "EMA Spread (%)",
                "HTF Trend",
                "Confidence",
                "Final Score",
                "Triggered Rules",
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

            entry_reason = t.diagnostics.entry_reason.value if t.diagnostics else "unknown"
            adx_val = f"{t.diagnostics.adx:.2f}" if t.diagnostics and t.diagnostics.adx is not None else ""
            atr_val = f"{t.diagnostics.atr:.4f}" if t.diagnostics and t.diagnostics.atr is not None else ""
            rsi_val = f"{t.diagnostics.rsi:.2f}" if t.diagnostics and t.diagnostics.rsi is not None else ""
            ema_spread = f"{t.diagnostics.ema_spread_pct:.2f}" if t.diagnostics and t.diagnostics.ema_spread_pct is not None else ""
            htf_trend = f"{t.diagnostics.htf_trend:.2f}" if t.diagnostics and t.diagnostics.htf_trend is not None else ""
            confidence = f"{t.diagnostics.confidence:.2f}" if t.diagnostics else ""
            final_score = f"{t.diagnostics.final_score:.2f}" if t.diagnostics else ""
            rules_str = t.diagnostics.triggered_rules if t.diagnostics else ""

            writer.writerow(
                [
                    str(t.symbol),
                    t.direction.name,
                    t.entry_time,
                    t.exit_time if t.exit_time else "",
                    ep,
                    xp,
                    entry_reason,
                    t.exit_reason.value,
                    gross_pnl,
                    fees,
                    slippage,
                    funding,
                    total_costs,
                    net_pnl,
                    holding_time,
                    adx_val,
                    atr_val,
                    rsi_val,
                    ema_spread,
                    htf_trend,
                    confidence,
                    final_score,
                    rules_str,
                ]
            )
