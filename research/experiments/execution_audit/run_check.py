"""Reprice the existing Q1 4h experiment after execution chronology repairs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
REPLAY = REPO / "research/experiments/pending_order_replay"
FILES = (
    "src/neon_radar/application/services/execution.py",
    "src/neon_radar/domain/execution_costs.py",
    "src/neon_radar/infrastructure/providers/binance_funding.py",
)


def main():
    metadata = {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_source_sha256": {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in FILES
        },
        "replay_sha256": hashlib.sha256((REPLAY / "replay.py").read_bytes()).hexdigest(),
    }
    output = ROOT / "audited_4h.json"
    subprocess.run(
        [
            sys.executable,
            str(REPLAY / "replay.py"),
            "--source-root",
            str(REPO),
            "--timeframe",
            "4h",
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text())
    result["audited_source"] = metadata
    output.write_text(json.dumps(result, indent=2))
    with ZipFile(REPLAY / "results_snapshot.zip") as archive:
        previous = json.loads(archive.read("both_fixes_4h.json"))
    for field in (
        "start",
        "end_exclusive",
        "settings",
        "scoring_sha256",
        "archive_manifest_sha256",
    ):
        assert result[field] == previous[field], field
    for r in (previous, result):
        assert r["open_positions"] == 0
        assert abs(r["equity"] - 10000 - r["net_closed"]) < 1e-7
        for trade in r["trades"]:
            assert (
                abs(
                    trade["net"]
                    - (trade["gross"] - trade["fees"] - trade["slippage"] - trade["funding"])
                )
                < 1e-7
            )
    fields = (
        "closed_trades",
        "net_closed",
        "net_profit_factor",
        "max_drawdown_pct",
        "max_open_positions",
        "open_limit_events",
        "fees",
        "slippage",
        "funding",
    )
    comparison = {
        "previous": {k: previous[k] for k in fields},
        "audited": {k: result[k] for k in fields},
    }
    (ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
