import json
import os
import subprocess

# Neon Radar Phase 1 Validation Execution Script

PERIODS = [
    ("Bull 1 (2020-2021)", "2020-10-01", "2021-05-01"),
    ("Bear (2021-2022)", "2021-11-01", "2022-12-31"),
    ("Chop (2023)", "2023-04-01", "2023-09-30"),
    ("Bull 2 (2023-2024)", "2023-10-01", "2024-03-31"),
    ("COVID Crash (2020)", "2020-02-15", "2020-04-15"),
    ("FTX Crash (2022)", "2022-10-15", "2022-12-15"),
]

TIMEFRAMES = ["1d", "4h"]
SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT"
MIN_HISTORY = 100

def run_backtest(name, start, end, tf):
    print(f"Running {name} on {tf}...", flush=True)
    cmd = [
        ".venv/Scripts/python.exe", "-m", "neon_radar.presentation.cli", "backtest",
        "--start", start,
        "--end", end,
        "--timeframe", tf,
        "--symbols", SYMBOLS,
        "--min-history", str(MIN_HISTORY),
        "--feature-analysis",
        "--output", "json"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {name} on {tf}:\n{result.stderr}")
        return None

    try:
        # Standard output might contain logging lines before the JSON output
        # So we search for the first '{'
        out = result.stdout
        start_idx = out.find('{')
        if start_idx == -1:
            print(f"No JSON found in output for {name} on {tf}:\n{out}")
            return None
        json_str = out[start_idx:]
        return json.loads(json_str)
    except Exception as e:
        print(f"Failed to parse JSON for {name} on {tf}: {e}")
        return None

def main():
    results = {}

    for name, start, end in PERIODS:
        results[name] = {}
        for tf in TIMEFRAMES:
            res = run_backtest(name, start, end, tf)
            if res:
                results[name][tf] = res

    # Dump results
    os.makedirs("results", exist_ok=True)
    with open("results/phase1_raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("All backtests completed. Results saved to results/phase1_raw.json")

if __name__ == "__main__":
    main()
