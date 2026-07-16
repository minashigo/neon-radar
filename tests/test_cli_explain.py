"""Tests for the CLI ``--explain`` mode and ``backtest`` subcommand."""

from __future__ import annotations

import io
import json
import sys
from contextlib import contextmanager
from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

from neon_radar.domain.scoring.value_objects import (
    AnalysisResult,
    Score,
    Signal,
)
from neon_radar.presentation.cli import (
    _print_breakdown,
    build_parser,
    main,
    print_backtest_report,
    print_result_json,
)

if TYPE_CHECKING:
    from pathlib import Path


@contextmanager
def captured_stdout():
    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


def _score_with_signals():
    score = Score(
        value=0.62,
        confidence=0.74,
        long_score=0.7,
        short_score=0.08,
        contributing_signals=3,
    )
    signals = (
        Signal(name="ema_trend", weight=0.3, value=0.9, confidence=0.8, description="EMA20>EMA50"),
        Signal(name="rsi_momentum", weight=0.25, value=0.4, confidence=0.7, description="RSI=62"),
        Signal(
            name="volume_confirmation",
            weight=0.25,
            value=0.5,
            confidence=0.6,
            description="Vol 1.6x",
        ),
    )
    return AnalysisResult(score=score, signals=signals, summary="bullish", computed_at=0)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestScanExplain:
    def test_explain_flag(self) -> None:
        ns = build_parser().parse_args(["scan", "--explain"])
        assert ns.explain is True

    def test_no_explain_by_default(self) -> None:
        ns = build_parser().parse_args(["scan"])
        assert ns.explain is False


class TestBacktestParser:
    def test_required_dates(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["backtest", "--end", "2024-12-31"])

    def test_minimal_trade_backtest(self) -> None:
        ns = build_parser().parse_args(["backtest", "--start", "2024-01-01", "--end", "2024-12-31"])
        assert ns.start == date(2024, 1, 1)
        assert ns.end == date(2024, 12, 31)
        assert ns.timeframe == "1d"

    def test_minimal_signals_backtest(self) -> None:
        ns = build_parser().parse_args(
            ["signals-backtest", "--start", "2024-01-01", "--end", "2024-12-31"]
        )
        assert ns.start == date(2024, 1, 1)
        assert ns.end == date(2024, 12, 31)
        assert ns.timeframe == "1d"
        assert ns.horizons == "1,3,7"

    def test_invalid_date(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["backtest", "--start", "not-a-date", "--end", "2024-12-31"])


# ---------------------------------------------------------------------------
# Breakdown rendering
# ---------------------------------------------------------------------------


class TestBreakdown:
    def test_renders_bullish_green(self) -> None:
        result = _score_with_signals()
        with captured_stdout() as buf:
            _print_breakdown(result, use_color=False)
        output = buf.getvalue()
        assert "ema_trend" in output
        assert "+0.27" in output  # 0.9 * 0.3
        assert "rsi_momentum" in output
        assert "+0.10" in output  # 0.4 * 0.25

    def test_handles_no_signals(self) -> None:
        empty = AnalysisResult(
            score=Score(
                value=0.0,
                confidence=0.0,
                long_score=0.0,
                short_score=0.0,
                contributing_signals=0,
            ),
            signals=(),
            summary="",
            computed_at=0,
        )
        with captured_stdout() as buf:
            _print_breakdown(empty, use_color=False)
        output = buf.getvalue()
        assert "no signals" in output

    def test_color_codes_when_enabled(self) -> None:
        result = _score_with_signals()
        with captured_stdout() as buf:
            _print_breakdown(result, use_color=True)
        output = buf.getvalue()
        assert "\x1b[" in output


# ---------------------------------------------------------------------------
# Backtest report rendering
# ---------------------------------------------------------------------------


def _empty_backtest_result() -> Any:
    from neon_radar.domain.scoring.backtest import BacktestResult

    return BacktestResult(
        config=None,  # type: ignore[arg-type]
        n_evaluations=0,
    )


class TestBacktestReport:
    def test_empty_result_warns(self) -> None:
        with captured_stdout() as buf:
            print_backtest_report(_empty_backtest_result(), use_color=False)
        assert "No evaluations" in buf.getvalue()


class TestJsonOutput:
    def test_serializes_to_json(self) -> None:
        from dataclasses import dataclass, field

        @dataclass
        class _MiniResult:
            config: Any = field(default=None)
            n_evaluations: int = 0
            symbol_results: dict = field(default_factory=dict)

        _MiniResult(n_evaluations=42)
        with captured_stdout() as buf:
            # Call the underlying print function on a real result with
            # minimal data. We construct it inline.
            from neon_radar.domain.scoring.backtest import BacktestResult

            real = BacktestResult(
                config=None,  # type: ignore[arg-type]
                n_evaluations=42,
            )
            print_result_json(real)
        out = buf.getvalue()
        parsed = json.loads(out)
        assert parsed["n_evaluations"] == 42


# ---------------------------------------------------------------------------
# main() entry point — list-rules path (offline)
# ---------------------------------------------------------------------------


class TestMainListRules:
    def test_runs_without_config(self, tmp_path: Path) -> None:
        with captured_stdout() as buf:
            rc = main(["--config", str(tmp_path / "nope.json"), "list-rules"])
        assert rc == 0
        assert "Registered scoring rules" in buf.getvalue()
