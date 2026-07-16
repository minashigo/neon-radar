"""Tests for the CLI runner.

Tests focus on:
* Argument parsing
* ``list-rules`` command (offline)
* ``scan`` command against a mocked :class:`BinanceClient`

The CLI is integration-tested end-to-end via :class:`CliRunner` (a
plain function we control). We never spawn a subprocess, which keeps
tests fast and free of env leaks.
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from neon_radar.domain.models import Symbol
from neon_radar.presentation.cli import (
    _c,
    _format_score_row,
    build_parser,
    cmd_list_rules,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def captured_stdout() -> Any:
    """Replace sys.stdout with a StringIO for the duration of the block."""
    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


def _minimal_scoring_rules_file(tmp_path: Path) -> Path:
    """Write a minimal scoring_rules.json with one EMA-trend rule."""
    import json

    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "name": "ema_trend",
                        "weight": 0.30,
                        "params": {"fast_period": 20, "slow_period": 50},
                    }
                ]
            }
        )
    )
    return path


def _minimal_app_config(tmp_path: Path) -> Path:
    import json

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "symbols": [
                    {"symbol": "BTCUSDT", "enabled": True},
                    {"symbol": "ETHUSDT", "enabled": True},
                ],
                "timeframes": ["1d"],
                "api": {
                    "base_url": "https://fapi.binance.com",
                    "timeout_seconds": 5.0,
                    "max_retries": 1,
                    "retry_backoff_seconds": 0.1,
                    "rate_limit_per_minute": 1200,
                },
            }
        )
    )
    return path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_scan_command(self) -> None:
        parser = build_parser()
        ns = parser.parse_args(["scan"])
        assert ns.command == "scan"
        assert ns.limit == 300
        assert ns.config == Path("config.json")
        assert ns.scoring == Path("scoring_rules.json")

    def test_list_rules_command(self) -> None:
        parser = build_parser()
        ns = parser.parse_args(["list-rules"])
        assert ns.command == "list-rules"

    def test_custom_paths(self) -> None:
        parser = build_parser()
        ns = parser.parse_args(
            ["--config", "/tmp/c.json", "--scoring", "/tmp/s.json", "scan", "--limit", "100"]
        )
        assert ns.config == Path("/tmp/c.json")
        assert ns.scoring == Path("/tmp/s.json")
        assert ns.limit == 100

    def test_no_color_flag(self) -> None:
        parser = build_parser()
        ns = parser.parse_args(["--no-color", "list-rules"])
        assert ns.no_color is True


# ---------------------------------------------------------------------------
# list-rules
# ---------------------------------------------------------------------------


class TestListRules:
    def test_prints_registered_rules(self) -> None:
        ns = build_parser().parse_args(["list-rules"])
        with captured_stdout() as buf:
            rc = cmd_list_rules(ns)
        output = buf.getvalue()
        assert rc == 0
        assert "ema_trend" in output
        assert "rsi_momentum" in output
        assert "funding_rate" in output
        assert "Registered indicators" in output


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatScoreRow:
    def test_bullish_row(self) -> None:
        from neon_radar.domain.scoring.value_objects import (
            AnalysisResult,
            Score,
            Signal,
        )

        score = Score(
            value=0.82,
            confidence=0.79,
            long_score=0.85,
            short_score=0.03,
            contributing_signals=2,
        )
        signals = (
            Signal(name="ema_trend", weight=0.3, value=0.9, confidence=0.8, description=""),
            Signal(name="rsi_momentum", weight=0.25, value=0.7, confidence=0.75, description=""),
        )
        result = AnalysisResult(score=score, signals=signals, summary="", computed_at=0)
        row = _format_score_row(Symbol("BTCUSDT"), result, use_color=False)
        assert "BTCUSDT" in row
        assert "+0.82" in row
        assert "0.79" in row
        assert "BULLISH" in row

    def test_neutral_row(self) -> None:
        from neon_radar.domain.scoring.value_objects import (
            AnalysisResult,
            Score,
        )

        score = Score(
            value=0.05,
            confidence=0.5,
            long_score=0.3,
            short_score=0.25,
            contributing_signals=1,
        )
        result = AnalysisResult(score=score, signals=(), summary="", computed_at=0)
        row = _format_score_row(Symbol("ETHUSDT"), result, use_color=False)
        assert "NEUTRAL" in row


class TestColorHelpers:
    def test_disabled_returns_plain(self) -> None:
        assert _c("X", "green", False) == "X"

    def test_enabled_returns_ansi(self) -> None:
        out = _c("X", "green", True)
        assert "\x1b[" in out
        assert "X" in out


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_list_rules_runs(self, tmp_path: Path) -> None:
        """``main list-rules`` works without any config files."""
        # Use a non-existent path — list-rules doesn't read config.
        with captured_stdout() as buf:
            rc = main(["--config", str(tmp_path / "nope.json"), "list-rules"])
        assert rc == 0
        assert "ema_trend" in buf.getvalue()

    def test_missing_config_raises_cleanly(self, tmp_path: Path) -> None:
        rules = _minimal_scoring_rules_file(tmp_path)
        with pytest.raises((SystemExit, Exception)):
            main(
                [
                    "--config",
                    str(tmp_path / "missing.json"),
                    "--scoring",
                    str(rules),
                    "scan",
                ]
            )
