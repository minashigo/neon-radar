import argparse
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from neon_radar.domain.models import KlineSeries, OHLCV, Symbol
from neon_radar.config.models import TimeFrame
from neon_radar.presentation.cli import _run_scan


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    import json
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "symbols": [{"symbol": "BTCUSDT", "enabled": True}],
        "timeframes": ["1d"],
        "api": {
            "base_url": "https://fapi.binance.com",
            "timeout_seconds": 5.0,
            "max_retries": 1,
            "retry_backoff_seconds": 0.1,
            "rate_limit_per_minute": 1200,
        },
    }))
    return path


@pytest.fixture
def minimal_scoring(tmp_path: Path) -> Path:
    import json
    path = tmp_path / "scoring.json"
    path.write_text(json.dumps({
        "rules": [
            {
                "name": "ema_trend",
                "weight": 0.30,
                "params": {"fast_period": 20, "slow_period": 50},
            }
        ]
    }))
    return path


@pytest.mark.asyncio
async def test_run_scan_success(minimal_config: Path, minimal_scoring: Path, capsys: pytest.CaptureFixture):
    args = argparse.Namespace(
        config=minimal_config,
        scoring=minimal_scoring,
        timeframe="1d",
        limit=100,
        explain=False,
        no_color=True,
    )

    mock_client = AsyncMock()
    
    def mock_get_klines(symbol, tf, limit=100, **kwargs):
        return KlineSeries(
            symbol=Symbol("BTCUSDT"),
            timeframe=tf,
            candles=tuple([
                OHLCV(open_time=1000 + i * 86400000, open=100, high=110, low=90, close=105, volume=1000, close_time=1000 + (i + 1) * 86400000 - 1)
                for i in range(100)
            ])
        )
    mock_client.get_klines.side_effect = mock_get_klines
    mock_client.get_funding_rate.return_value = None

    with patch("neon_radar.presentation.cli.BinanceClient", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        rc = await _run_scan(args)

    assert rc == 0
    captured = capsys.readouterr()
    assert "BTCUSDT" in captured.out
    assert "Failed to score" not in captured.err
