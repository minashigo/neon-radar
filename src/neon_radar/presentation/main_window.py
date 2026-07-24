"""Main application window.

Layout::

    ┌─────────────────────────────────────────────┐
    │ StatusBar: last refresh, auto-refresh status │
    ├─────────────────────────────────────────────┤
    │   RankingTable (top, ~50%)                  │
    │   ─────────────────────────────────          │
    │   DetailPanel (bottom, ~25%)                │
    ├─────────────────────────────────────────────┤
    │   Chart dock (right or bottom, ~25%)         │ ← Stage 6B
    │   (chart appears when "View Chart" clicked) │
    └─────────────────────────────────────────────┘

The window owns:

* one :class:`MarketDataService` whose AsyncWorker thread runs the
  scoring computation off the UI thread;
* a :class:`QTimer` that fires every ``refresh_seconds`` to kick off a
  fresh batch of ``request_klines`` for every enabled symbol;
* per-symbol caches ``_last_results[symbol]`` and
  ``_last_klines[symbol]`` so the chart can render instantly when
  the user clicks "View Chart".
"""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QSplitter,
    QStatusBar,
)

from neon_radar.application.services.analysis import analyze_series
from neon_radar.application.services.market_data import MarketDataService
from neon_radar.domain.models import KlineSeries, Symbol
from neon_radar.infrastructure.cache import KlineCache
from neon_radar.presentation.widgets.chart_widget import ChartWidget
from neon_radar.presentation.widgets.detail_panel import DetailPanel
from neon_radar.presentation.widgets.ranking_table import (
    RankingRow,
    RankingTable,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from neon_radar.config.models import AppConfig, TimeFrame
    from neon_radar.config.scoring_models import ScoringRulesConfig
    from neon_radar.domain.funding import FundingRate
    from neon_radar.domain.indicators.base import IndicatorSeries
    from neon_radar.domain.scoring import AnalysisResult
    from neon_radar.infrastructure.exchanges.base import ExchangeClient


class MainWindow(QMainWindow):
    """The Neon Radar main window."""

    def __init__(
        self,
        *,
        config: AppConfig,
        scoring_config: ScoringRulesConfig,
        rules: tuple | None = None,
        exchange_factory: Callable[..., ExchangeClient] | None = None,
        refresh_seconds: int = 60,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._scoring_config = scoring_config
        self._refresh_seconds = refresh_seconds

        # ``scoring_config.enabled_rules()`` returns ``RuleSpec`` Pydantic
        # models, not actual ``FactorRule`` instances. The CLI loader
        # (``load_rules``) instantiates them; we accept pre-built rules
        # for testability and so the app entry point can wire them in.
        if rules is None:
            raise ValueError(
                "Pass pre-built rule instances via `rules=`. "
                "Use `neon_radar.config.scoring_loader.load_rules` to "
                "instantiate them from scoring_rules.json."
            )
        self._rules = rules
        self._timeframe = config.timeframes[0]  # type: ignore[attr-defined]

        # Cached analysis output per symbol — instant updates for chart.
        self._last_results: dict[Symbol, AnalysisResult] = {}
        self._last_klines: dict[Symbol, KlineSeries] = {}
        self._last_indicators: dict[Symbol, tuple[IndicatorSeries, ...]] = {}
        self._last_funding: dict[Symbol, FundingRate] = {}

        self._setup_service(exchange_factory)
        self._setup_ui()
        self._setup_status_bar()
        self._setup_timer()

        # Connect service signals.
        self._service.klines_ready.connect(self._on_klines_ready)
        self._service.funding_ready.connect(self._on_funding_ready)
        self._service.error_occurred.connect(self._on_error)
        self._detail.view_chart_requested.connect(self._on_view_chart)

        # Initial population: empty ranking table.
        self._ranking.update_rows([])
        self._detail.clear()

        # Kick off the first refresh immediately.
        self._refresh_all()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_service(self, exchange_factory: Callable | None) -> None:
        """Build and start the async market data service."""
        from neon_radar.infrastructure.exchanges.binance import BinanceClient

        factory = exchange_factory or BinanceClient
        exchange = factory(self._config.api)
        cache = (
            KlineCache(
                self._config.cache.directory,
                ttl_seconds=self._config.cache.ttl_seconds,
            )
            if self._config.cache.enabled
            else None
        )
        self._service = MarketDataService(exchange=exchange, cache=cache)
        self._service.start()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Neon Radar")
        self.resize(1400, 900)

        # Splitter: ranking table on top, detail panel below.
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        self._ranking = RankingTable()
        self._ranking.symbol_selected.connect(self._on_symbol_selected)

        self._detail = DetailPanel()

        splitter.addWidget(self._ranking)
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 3)  # ranking takes 3/4
        splitter.setStretchFactor(1, 1)  # detail takes 1/4

        self.setCentralWidget(splitter)

        # Chart dock (initially hidden).
        self._chart = ChartWidget()
        self._chart_dock = QDockWidget("Chart", self)
        self._chart_dock.setWidget(self._chart)
        self._chart_dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._chart_dock.hide()  # user opens via "View Chart"
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._chart_dock)

    def _setup_status_bar(self) -> None:
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._update_status()

    def _setup_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.setInterval(self._refresh_seconds * 1000)
        self._timer.timeout.connect(self._refresh_all)
        self._timer.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt override
        self._timer.stop()
        with contextlib.suppress(Exception):
            self._service.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        for sym_cfg in self._config.enabled_symbols():
            symbol = Symbol(sym_cfg.symbol)
            self._service.request_klines(
                symbol,
                self._timeframe,
                limit=300,
            )
            self._service.request_funding_rate(symbol)

    def _on_klines_ready(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        series: KlineSeries,
    ) -> None:
        self._last_klines[symbol] = series
        self._apply_scoring(symbol, series)

    def _on_funding_ready(self, symbol: Symbol, funding: FundingRate) -> None:
        self._last_funding[symbol] = funding
        series = self._last_klines.get(symbol)
        if series is not None:
            self._apply_scoring(symbol, series)

    def _apply_scoring(self, symbol: Symbol, series: KlineSeries) -> None:
        funding = self._last_funding.get(symbol)
        result, indicators = self._compute_result(symbol, series, funding_rate=funding)
        if result is None:
            return
        self._last_results[symbol] = result
        self._last_indicators[symbol] = indicators
        self._update_ranking()
        if self._ranking.selected_symbol() == str(symbol):
            self._detail.show_result(str(symbol), result)
            self._render_chart_for(symbol)
        self._update_status()

    def _on_error(self, kind: str, message: str) -> None:
        # Surface non-fatal errors in the status bar.
        self._status_bar.showMessage(f"[{kind}] {message}", 5_000)

    def _on_symbol_selected(self, symbol: str) -> None:
        result = self._last_results.get(Symbol(symbol))
        if result is not None:
            self._detail.show_result(symbol, result)
            self._render_chart_for(Symbol(symbol))

    def _on_view_chart(self) -> None:
        """Show the chart dock and render current selection."""
        self._chart_dock.show()
        self._chart_dock.raise_()
        symbol_str = self._ranking.selected_symbol()
        if symbol_str:
            self._render_chart_for(Symbol(symbol_str))

    # ------------------------------------------------------------------
    # Scoring computation
    # ------------------------------------------------------------------

    def _compute_result(
        self,
        symbol: Symbol,
        series: KlineSeries,
        *,
        funding_rate: FundingRate | None = None,
    ) -> tuple[AnalysisResult | None, tuple[IndicatorSeries, ...]]:
        """Run the engine on the supplied series."""
        from neon_radar.application.services.indicator_pipeline import IndicatorSpec

        ui_indicators = (
            IndicatorSpec(name="ema", params={"period": 20}, tag="20"),
            IndicatorSpec(name="ema", params={"period": 50}, tag="50"),
        )

        try:
            result = analyze_series(
                series,
                self._rules,
                min_confidence=self._scoring_config.min_confidence,
                confluence_bonus=self._scoring_config.confluence_bonus,
                confluence_penalty=self._scoring_config.confluence_penalty,
                max_confidence_boost=self._scoring_config.max_confidence_boost,
                timestamp=int(series.candles[-1].open_time),
                funding_rate=funding_rate,
                extra_indicators=ui_indicators,
            )
        except Exception:
            return None, ()
        assert result.market_state is not None
        return result, tuple(result.market_state.indicator_series)

    # ------------------------------------------------------------------
    # Chart rendering
    # ------------------------------------------------------------------

    def _render_chart_for(self, symbol: Symbol) -> None:
        series = self._last_klines.get(symbol)
        if series is None:
            return
        result = self._last_results.get(symbol)
        indicators = self._last_indicators.get(symbol, ())
        trade_setup = result.trade_setup if result else None
        self._chart.render(series, indicators, trade_setup=trade_setup)

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------

    def _update_ranking(self) -> None:
        rows = [
            RankingRow(
                symbol=str(symbol),
                score=r.score.value,
                confidence=r.score.confidence,
                bias=r.score.bias,
                factor_arrows=self._factor_arrows(r),
            )
            for symbol, r in self._last_results.items()
        ]
        self._ranking.update_rows(rows)
        # Re-select if there was a selection.
        current = self._ranking.selected_symbol()
        if current is not None:
            self._ranking.select_symbol(current)

    @staticmethod
    def _factor_arrows(result: AnalysisResult) -> str:
        """Compact per-factor arrow summary for the ranking table."""
        arrows = {
            "ema_trend": "trend",
            "rsi_momentum": "mom",
            "volume_confirmation": "vol",
            "volatility_filter": "volat",
            "funding_rate": "fund",
        }
        parts: list[str] = []
        for sig in result.signals:
            short = arrows.get(sig.name, sig.name)
            if sig.value > 0.05:
                parts.append(f"{short}↑")
            elif sig.value < -0.05:
                parts.append(f"{short}↓")
            else:
                parts.append(f"{short}→")
        return "  ".join(parts)

    def _update_status(self) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        n = len(self._last_results)
        msg = (
            f"Neon Radar v0.1  |  Last refresh: {ts}  |  "
            f"Auto: {self._refresh_seconds}s  |  "
            f"Symbols: {n}/{len(self._config.enabled_symbols())}"
        )
        self._status_bar.showMessage(msg)
