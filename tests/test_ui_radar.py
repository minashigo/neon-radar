"""Tests for the Stage 6A UI components — pytest-qt based.

These tests use a ``QApplication`` provided by pytest-qt. They exercise
the widgets directly without opening a real network connection.
"""

from __future__ import annotations

from dataclasses import dataclass

from neon_radar.config.models import ApiConfig, AppConfig, RefreshConfig, TimeFrame
from neon_radar.config.scoring_models import (
    RuleSpec,
    ScoringRulesConfig,
)
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import KlineSeries
from neon_radar.domain.scoring.value_objects import (
    AnalysisResult,
    Score,
    Signal,
)
from neon_radar.presentation.theme.neon_palette import NeonPalette
from neon_radar.presentation.widgets.detail_panel import DetailPanel
from neon_radar.presentation.widgets.ranking_table import RankingRow, RankingTable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    symbol: str,
    *,
    score: float,
    confidence: float,
    bias: Bias,
    factor_signals: tuple[tuple[str, float, float, str], ...] = (),
) -> AnalysisResult:
    """Build an AnalysisResult with the given score + breakdown."""
    signals = tuple(
        Signal(
            name=name,
            weight=weight,
            value=value,
            confidence=0.8,
            description=description,
        )
        for name, value, weight, description in factor_signals
    )
    long_score = sum(s.value * s.weight for s in signals if s.value > 0)
    short_score = -sum(s.value * s.weight for s in signals if s.value < 0)
    score_obj = Score(
        value=score,
        confidence=confidence,
        long_score=long_score,
        short_score=short_score,
        contributing_signals=len(signals),
    )
    return AnalysisResult(
        score=score_obj,
        signals=signals,
        summary="",
        computed_at=0,
    )


# ---------------------------------------------------------------------------
# RankingTable tests
# ---------------------------------------------------------------------------


class TestRankingTable:
    def test_initial_state_is_empty(self, qtbot) -> None:
        table = RankingTable()
        assert table.rowCount() == 0

    def test_update_rows_sorts_by_score_desc(self, qtbot) -> None:
        table = RankingTable()
        rows = [
            RankingRow("AAA", 0.3, 0.7, Bias.BULLISH, "trend↑"),
            RankingRow("BBB", 0.8, 0.6, Bias.BULLISH, "trend↑ mom↑"),
            RankingRow("CCC", -0.4, 0.5, Bias.BEARISH, "trend↓"),
        ]
        table.update_rows(rows)
        symbols_in_order = [table.item(r, 1).text() for r in range(table.rowCount())]
        assert symbols_in_order == ["BBB", "AAA", "CCC"]

    def test_ranks_are_one_indexed(self, qtbot) -> None:
        table = RankingTable()
        rows = [RankingRow("X", 0.5, 0.5, Bias.NEUTRAL, "x→")]
        table.update_rows(rows)
        assert table.item(0, 0).text() == "1"

    def test_bias_colors(self, qtbot) -> None:
        table = RankingTable()
        table.update_rows(
            [
                RankingRow("BU", 0.6, 0.7, Bias.BULLISH, "↑"),
                RankingRow("BE", -0.6, 0.7, Bias.BEARISH, "↓"),
                RankingRow("NE", 0.0, 0.7, Bias.NEUTRAL, "→"),
            ]
        )
        # Rows are sorted by score descending: BU (0.6), NE (0.0), BE (-0.6).
        bullish = table.item(0, 2).foreground().color().name().lower()
        bearish = table.item(2, 2).foreground().color().name().lower()
        neutral = table.item(1, 2).foreground().color().name().lower()
        assert bullish == NeonPalette.ACCENT_BULLISH.lower()
        assert bearish == NeonPalette.ACCENT_BEARISH.lower()
        assert neutral == NeonPalette.ACCENT_NEUTRAL.lower()

    def test_emits_symbol_selected_on_click(self, qtbot) -> None:
        table = RankingTable()
        captured = []
        table.symbol_selected.connect(captured.append)
        table.update_rows(
            [
                RankingRow("AAA", 0.5, 0.7, Bias.BULLISH, "↑"),
                RankingRow("BBB", 0.3, 0.7, Bias.BULLISH, "↑"),
            ]
        )
        # Select second row.
        table.selectRow(1)
        qtbot.waitUntil(lambda: captured != [])
        assert captured[-1] == "BBB"

    def test_select_symbol(self, qtbot) -> None:
        table = RankingTable()
        table.update_rows(
            [
                RankingRow("AAA", 0.5, 0.7, Bias.BULLISH, "↑"),
                RankingRow("BBB", 0.3, 0.7, Bias.BULLISH, "↑"),
            ]
        )
        table.select_symbol("AAA")
        assert table.selected_symbol() == "AAA"


# ---------------------------------------------------------------------------
# DetailPanel tests
# ---------------------------------------------------------------------------


class TestDetailPanel:
    def test_initial_state_shows_dashes(self, qtbot) -> None:
        panel = DetailPanel()
        assert "—" in panel._title.text() or "—" in panel._score_value.text()

    def test_show_result_populates_values(self, qtbot) -> None:
        panel = DetailPanel()
        result = _make_result(
            "BTCUSDT",
            score=0.72,
            confidence=0.81,
            bias=Bias.BULLISH,
            factor_signals=(
                ("ema_trend", 0.9, 0.30, "EMA20 above EMA50"),
                ("rsi_momentum", 0.5, 0.25, "RSI in bull zone"),
            ),
        )
        panel.show_result("BTCUSDT", result)
        assert "BTCUSDT" in panel._title.text()
        assert "+0.72" in panel._score_value.text()
        assert "0.81" in panel._conf_value.text()
        assert "BULLISH" in panel._bias_value.text()
        assert panel._breakdown.rowCount() == 2

    def test_breakdown_contributions_colored(self, qtbot) -> None:
        panel = DetailPanel()
        result = _make_result(
            "X",
            score=0.5,
            confidence=0.5,
            bias=Bias.BULLISH,
            factor_signals=(
                ("bull", 0.8, 0.5, "bullish"),
                ("bear", -0.6, 0.5, "bearish"),
            ),
        )
        panel.show_result("X", result)
        # First row (bull) should be green-ish.
        bull_color = panel._breakdown.item(0, 1).foreground().color().name().lower()
        assert bull_color == NeonPalette.ACCENT_BULLISH.lower()

    def test_clear_resets_to_dashes(self, qtbot) -> None:
        panel = DetailPanel()
        panel.show_result(
            "X",
            _make_result("X", score=0.5, confidence=0.5, bias=Bias.BULLISH),
        )
        panel.clear()
        assert panel._breakdown.rowCount() == 0


# ---------------------------------------------------------------------------
# MainWindow integration tests (with fake exchange)
# ---------------------------------------------------------------------------


@dataclass
class _FakeExchange:
    """Returns deterministic synthetic klines."""

    def __init__(self, api_config=None) -> None:  # accept and ignore
        pass

    async def get_klines(self, symbol, timeframe, *, limit=500, end_time=None):
        from neon_radar.domain.models import OHLCV

        base_ts = 1_700_000_000_000
        n = min(limit, 100)
        candles = tuple(
            OHLCV(
                open_time=base_ts + i * 86_400_000,
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000.0,
            )
            for i in range(n)
        )
        return KlineSeries(symbol=symbol, timeframe=timeframe, candles=candles)

    async def get_funding_rate(self, symbol):
        from neon_radar.domain.funding import FundingRate

        return FundingRate(symbol=symbol, rate=0.0001)

    async def close(self) -> None:
        return None


def _make_app_config() -> AppConfig:
    return AppConfig(
        symbols=[
            {"symbol": "BTCUSDT", "enabled": True},
            {"symbol": "ETHUSDT", "enabled": True},
        ],
        timeframes=[TimeFrame.D1],
        refresh=RefreshConfig(interval_seconds=60, auto_refresh=True),
        api=ApiConfig(),
    )


def _make_scoring_config() -> ScoringRulesConfig:
    return ScoringRulesConfig(
        rules=[
            RuleSpec(name="ema_trend", enabled=True, weight=0.30, params={}),
            RuleSpec(name="rsi_momentum", enabled=True, weight=0.25, params={}),
        ],
        min_confidence=0.0,
    )


class TestMainWindow:
    def test_constructs_without_network(self, qtbot, monkeypatch) -> None:
        from neon_radar.domain.scoring import (
            EMATrendRule,
            RSIMomentumRule,
        )
        from neon_radar.infrastructure import cache as cache_mod
        from neon_radar.infrastructure.cache import KlineCache
        from neon_radar.infrastructure.exchanges import binance as binance_mod
        from neon_radar.presentation.main_window import MainWindow

        # Patch KlineCache so it does not touch the real filesystem.
        def _noop_cache(*args, **kwargs):
            return KlineCache.__new__(KlineCache)

        monkeypatch.setattr(cache_mod, "KlineCache", _noop_cache)
        # Patch BinanceClient to use our FakeExchange.
        monkeypatch.setattr(binance_mod, "BinanceClient", _FakeExchange)

        # Pre-built rule instances — what the loader would produce.
        rules = (EMATrendRule(weight=0.30), RSIMomentumRule(weight=0.25))

        window = MainWindow(
            config=_make_app_config(),
            scoring_config=_make_scoring_config(),
            exchange_factory=_FakeExchange,
            refresh_seconds=60,
            rules=rules,
        )
        qtbot.addWidget(window)
        assert window.windowTitle() == "Neon Radar"
        window.close()
