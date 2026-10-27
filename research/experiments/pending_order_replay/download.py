"""Download checksum-verified, immutable Binance USD-M monthly archives."""

import asyncio
import hashlib
import json
from pathlib import Path

import httpx

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
ROOT = Path(__file__).resolve().parent / "data"


async def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    previous_path = ROOT / "manifest.json"
    previous = (
        {r["file"]: r["sha256"] for r in json.loads(previous_path.read_text())}
        if previous_path.exists()
        else {}
    )
    semaphore = asyncio.Semaphore(6)
    manifest = []
    months = [f"2024-{m:02}" for m in range(5, 13)] + [f"2025-{m:02}" for m in range(1, 4)]
    jobs = [(s, "klines", m) for s in SYMBOLS for m in months]
    jobs += [(s, "fundingRate", m) for s in SYMBOLS for m in months[-4:]]
    async with httpx.AsyncClient(timeout=40, follow_redirects=True) as client:

        async def fetch(symbol, kind, month):
            name = (
                f"{symbol}-1h-{month}.zip"
                if kind == "klines"
                else f"{symbol}-fundingRate-{month}.zip"
            )
            suffix = (
                f"klines/{symbol}/1h/{name}" if kind == "klines" else f"fundingRate/{symbol}/{name}"
            )
            url = f"https://data.binance.vision/data/futures/um/monthly/{suffix}"
            path = ROOT / name
            async with semaphore:
                if (
                    path.exists()
                    and name in previous
                    and hashlib.sha256(path.read_bytes()).hexdigest() != previous[name]
                ):
                    raise ValueError(f"Cached archive changed: {name}")
                if not path.exists() or name not in previous:
                    for attempt in range(3):
                        try:
                            response = await client.get(url)
                            response.raise_for_status()
                            checksum = await client.get(url + ".CHECKSUM")
                            checksum.raise_for_status()
                            digest = hashlib.sha256(response.content).hexdigest()
                            if digest != checksum.text.split()[0]:
                                raise ValueError(f"Checksum mismatch: {name}")
                            path.write_bytes(response.content)
                            break
                        except (httpx.HTTPError, ValueError):
                            if attempt == 2:
                                raise
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest.append({"file": name, "sha256": digest, "url": url})
                if len(manifest) % 15 == 0:
                    print(f"{len(manifest)}/{len(jobs)} verified archives", flush=True)

        await asyncio.gather(*(fetch(*job) for job in jobs))
    (ROOT / "manifest.json").write_text(
        json.dumps(sorted(manifest, key=lambda x: x["file"]), indent=2)
    )
    print("Download complete", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
