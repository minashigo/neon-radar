"""Generate Market Regime Diagnostics Report from trades_export.csv."""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def calculate_metrics(trades: list[dict]) -> dict:
    total = len(trades)
    if total == 0:
        return {
            "total": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "avg_r": 0.0
        }

    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]

    win_rate = len(wins) / total

    gross_wins = sum(t["gross_pnl"] for t in wins)
    gross_losses = sum(abs(t["gross_pnl"]) for t in losses)

    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    net_pnl_sum = sum(t["net_pnl"] for t in trades)
    avg_r = net_pnl_sum / total
    expectancy = avg_r  # simplified Net Expectancy as Avg Net PnL per trade

    return {
        "total": total,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_r": avg_r,
    }


def format_table(groups: dict[str, list[dict]], title: str) -> str:
    lines = [f"### {title}"]
    lines.append("| Group | Trades | Win Rate | Profit Factor | Net Expectancy | Avg R (%) |")
    lines.append("|-------|--------|----------|---------------|----------------|-----------|")

    for key, trades in sorted(groups.items()):
        m = calculate_metrics(trades)
        pf = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float("inf") else "inf"
        lines.append(
            f"| {key} | {m['total']} | {m['win_rate']*100:.1f}% | {pf} | {m['expectancy']:.3f}% | {m['avg_r']:.3f}% |"
        )
    return "\n".join(lines) + "\n\n"


def run_diagnostics(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    trades = []
    with csv_path.open(mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("Net PnL (%)"):
                continue

            t = {
                "symbol": row["Symbol"],
                "direction": row["Direction"],
                "net_pnl": float(row["Net PnL (%)"]),
                "gross_pnl": float(row["Gross PnL (%)"]),
                "adx": float(row["ADX"]) if row.get("ADX") else None,
                "rsi": float(row["RSI"]) if row.get("RSI") else None,
                "confidence": float(row["Confidence"]) if row.get("Confidence") else None,
                "rules": row.get("Triggered Rules", ""),
            }
            trades.append(t)

    if not trades:
        print("No valid trades found.")
        return

    # Groups
    overall = {"All Trades": trades}

    by_direction = defaultdict(list)
    by_symbol = defaultdict(list)
    by_adx = defaultdict(list)
    by_rsi = defaultdict(list)
    by_confidence = defaultdict(list)
    by_rule = defaultdict(list)

    for t in trades:
        by_direction[t["direction"]].append(t)
        by_symbol[t["symbol"]].append(t)

        # ADX
        if t["adx"] is not None:
            if t["adx"] < 20: by_adx["< 20 (Chop)"].append(t)
            elif t["adx"] < 30: by_adx["20 - 30 (Trending)"].append(t)
            else: by_adx["> 30 (Strong Trend)"].append(t)

        # RSI
        if t["rsi"] is not None:
            if t["rsi"] < 30: by_rsi["< 30 (Oversold)"].append(t)
            elif t["rsi"] < 50: by_rsi["30 - 50 (Bearish)"].append(t)
            elif t["rsi"] < 70: by_rsi["50 - 70 (Bullish)"].append(t)
            else: by_rsi["> 70 (Overbought)"].append(t)

        # Confidence
        if t["confidence"] is not None:
            if t["confidence"] < 0.6: by_confidence["< 0.6"].append(t)
            elif t["confidence"] < 0.8: by_confidence["0.6 - 0.8"].append(t)
            else: by_confidence["> 0.8"].append(t)

        # Rules
        rules = [r.split(":")[0] for r in t["rules"].split(", ") if r]
        for r in rules:
            by_rule[r].append(t)

    # Generate Report
    report = ["# Trade Diagnostics Report (Sprint 13)\n\n"]

    report.append(format_table(overall, "Overall Performance"))
    report.append(format_table(by_direction, "By Direction"))
    report.append(format_table(by_symbol, "By Asset"))
    report.append(format_table(by_adx, "By ADX Regime"))
    report.append(format_table(by_rsi, "By RSI Level"))
    report.append(format_table(by_confidence, "By Confidence Level"))
    report.append(format_table(by_rule, "By Triggered Rule"))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("".join(report))

    print(f"Diagnostics report saved to {out_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    csv_file = base_dir / "results" / "trades_export.csv"
    out_file = base_dir / "results" / "diagnostics_report.md"
    run_diagnostics(csv_file, out_file)
