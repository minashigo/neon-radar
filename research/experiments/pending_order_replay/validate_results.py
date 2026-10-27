"""Validate paired inputs, ledger arithmetic, and quarterly comparison outputs."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {"baseline": "Initial", "rsi_only": "RSI only", "both_fixes": "RSI + reservations"}


def validate():
    all_results = {}
    summaries = []
    for tf in ("1h", "4h"):
        previous = None
        for name in LABELS:
            result = json.loads((ROOT / "results" / f"{name}_{tf}.json").read_text())
            if previous is not None:
                for field in (
                    "settings",
                    "start",
                    "end_exclusive",
                    "scoring_sha256",
                    "archive_manifest_sha256",
                ):
                    assert result[field] == previous[field], (name, tf, field)
            previous = result
            assert result["open_positions"] == 0
            assert result["closed_trades"] == len(result["trades"])
            assert math.isclose(
                result["net_closed"], sum(p["net"] for p in result["trades"]), abs_tol=1e-7
            )
            assert math.isclose(result["equity"], 10000 + result["net_closed"], abs_tol=1e-7)
            for p in result["trades"]:
                assert p["entry"] <= p["exit"]
                assert math.isclose(
                    p["net"], p["gross"] - p["fees"] - p["slippage"] - p["funding"], abs_tol=1e-7
                )
            if name == "both_fixes":
                assert result["max_open_positions"] <= 3
                assert result["open_limit_events"] == 0
            equities = {row["time"]: row["equity"] for row in result["curve"]}
            start_equity = 10000
            monthly = {}
            for month, end in (
                ("January", "2025-02-01"),
                ("February", "2025-03-01"),
                ("March", "2025-04-01"),
            ):
                end_ms = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000)
                end_equity = equities[end_ms]
                monthly[month] = end_equity - start_equity
                start_equity = end_equity
            assert math.isclose(sum(monthly.values()), result["net_closed"], abs_tol=1e-7)
            row = {k: v for k, v in result.items() if k not in ("curve", "trades")}
            row["variant"] = name
            row["monthly_equity_pnl"] = monthly
            summaries.append(row)
            all_results[name, tf] = result
    (ROOT / "summary.json").write_text(json.dumps(summaries, indent=2))
    for row in summaries:
        print(
            row["timeframe"],
            row["variant"],
            "trades",
            row["closed_trades"],
            "net",
            round(row["net_closed"], 2),
            "PF",
            round(row["net_profit_factor"], 3),
            "DD",
            round(row["max_drawdown_pct"], 2),
            "maxpositions",
            row["max_open_positions"],
            "monthly",
            {k: round(v, 2) for k, v in row["monthly_equity_pnl"].items()},
        )
    print("All paired-input and ledger validation checks passed.")


if __name__ == "__main__":
    validate()
