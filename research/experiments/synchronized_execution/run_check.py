"""Replay the identical quarter with canonical and reversed input order."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
REPLAY = REPO / "research/experiments/pending_order_replay/replay.py"


def main():
    def run(order):
        path = ROOT / f"{order}_4h.json"
        subprocess.run(
            [
                sys.executable,
                str(REPLAY),
                "--source-root",
                str(REPO),
                "--timeframe",
                "4h",
                "--symbols-order",
                order,
                "--output",
                str(path),
            ],
            check=True,
        )
        return json.loads(path.read_text())

    with ThreadPoolExecutor(max_workers=2) as pool:
        canonical, reverse = list(pool.map(run, ("canonical", "reverse")))
    assert canonical["requested_symbol_order"] == list(reversed(reverse["requested_symbol_order"]))
    # Compare full trades, equity curves, source hashes and scalar metrics.
    left = {k: v for k, v in canonical.items() if k != "requested_symbol_order"}
    right = {k: v for k, v in reverse.items() if k != "requested_symbol_order"}
    assert left == right, "Portfolio result depends on input symbol order"
    assert canonical["max_open_positions"] <= 3 and canonical["open_limit_events"] == 0
    for result in (canonical, reverse):
        assert abs(result["equity"] - 10000 - result["net_closed"]) < 1e-7
        assert abs(sum(t["net"] for t in result["trades"]) - result["net_closed"]) < 1e-7
        for trade in result["trades"]:
            assert (
                abs(
                    trade["net"]
                    - trade["gross"]
                    + trade["fees"]
                    + trade["slippage"]
                    + trade["funding"]
                )
                < 1e-7
            )
    previous = json.loads(
        (REPO / "research/experiments/execution_audit/audited_4h.json").read_text()
    )
    for key in ("settings", "scoring_sha256", "archive_manifest_sha256", "start", "end_exclusive"):
        assert previous[key] == canonical[key], key
    keys = (
        "closed_trades",
        "net_closed",
        "net_profit_factor",
        "max_drawdown_pct",
        "max_open_positions",
        "open_limit_events",
    )
    summary = {
        "all_trade_and_curve_fields_identical": True,
        "previous_sequential": {k: previous[k] for k in keys},
        "synchronized": {k: canonical[k] for k in keys},
    }
    (ROOT / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
