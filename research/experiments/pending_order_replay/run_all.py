"""Reproduce the paired experiment without changing the active checkout or paper state."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
REVISIONS = {"baseline": "358e285", "rsi_only": "bf10761", "both_fixes": "7ed6d3a"}


def main():
    subprocess.run([sys.executable, str(ROOT / "download.py")], check=True)
    sources = {}
    for name, revision in REVISIONS.items():
        source = REPO / ".cache" / "pending_order_replay" / name
        expected = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", revision], text=True
        ).strip()
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "-C", str(REPO), "worktree", "add", "--detach", str(source), revision],
                check=True,
            )
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if expected != actual:
            raise ValueError(f"Unexpected cached checkout: {source}")
        sources[name] = source

    def run(job):
        name, tf = job
        output = ROOT / "results" / f"{name}_{tf}.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "replay.py"),
                "--source-root",
                str(sources[name]),
                "--timeframe",
                tf,
                "--output",
                str(output),
            ],
            check=True,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, [(name, tf) for name in REVISIONS for tf in ("1h", "4h")]))
    subprocess.run([sys.executable, str(ROOT / "validate_results.py")], check=True)


if __name__ == "__main__":
    main()
