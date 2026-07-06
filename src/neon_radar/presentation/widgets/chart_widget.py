"""Chart widget — candlesticks + indicators + volume for one symbol.

Uses pyqtgraph's :class:`GraphicsLayoutWidget` with two stacked
plots (price on top, volume on bottom). Candlesticks are rendered via
a custom :class:`CandlestickItem` — a :class:`QGraphicsObject` that
draws itself as a :class:`QPicture` for fast repaints.

Indicators (EMA, Bollinger, …) are drawn as :class:`PlotDataItem`
lines on the price plot. Volume bars are drawn as :class:`BarGraphItem`
on the volume plot (color-coded by candle direction).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pyqtgraph as pg
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPicture
from PySide6.QtWidgets import QVBoxLayout, QWidget

from neon_radar.presentation.theme.neon_palette import NeonPalette

if TYPE_CHECKING:
    from neon_radar.domain.indicators.base import IndicatorSeries
    from neon_radar.domain.models import KlineSeries


class CandlestickItem(pg.GraphicsObject):
    """One candlestick — drawn as a pre-rendered :class:`QPicture`.

    Drawing each candle as a QPicture (instead of issuing per-pixel
    QPainter calls in :meth:`paint`) lets Qt batch the render and
    keeps scrolling / zooming smooth with hundreds of candles.
    """

    def __init__(self, x: float, o: float, h: float, lo: float, c: float) -> None:
        super().__init__()
        self.x = x
        self.o = o
        self.h = h
        self.lo = lo
        self.c = c
        self._picture = QPicture()
        self._generate_picture()

    def _generate_picture(self) -> None:
        painter = QPainter(self._picture)
        try:
            is_bull = self.c >= self.o
            color_hex = (
                NeonPalette.ACCENT_BULLISH if is_bull else NeonPalette.ACCENT_BEARISH
            )
            color = QColor(color_hex)
            pen = QPen(color)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(QBrush(color))

            # Wick (high-low line).
            painter.drawLine(
                pg.QtCore.QPointF(self.x, self.lo),
                pg.QtCore.QPointF(self.x, self.h),
            )
            # Body (open-close rectangle).
            body = pg.QtCore.QRectF(
                self.x - 0.3,
                min(self.o, self.c),
                0.6,
                abs(self.c - self.o) if self.c != self.o else 0.001,
            )
            painter.drawRect(body)
        finally:
            painter.end()

    def paint(self, painter, *_args) -> None:
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> pg.QtCore.QRectF:  # noqa: N802 — Qt override
        # Pad slightly so anti-aliased wicks don't get clipped.
        return pg.QtCore.QRectF(self.x - 0.5, self.lo, 1.0, max(self.h - self.lo, 0.001))


class ChartWidget(QWidget):
    """Price chart + indicators + volume for one symbol."""

    #: Default number of recent candles to display.
    DEFAULT_VISIBLE = 200

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        series: KlineSeries,
        indicators: tuple[IndicatorSeries, ...] = (),
        *,
        visible_candles: int = DEFAULT_VISIBLE,
    ) -> None:
        """Render the supplied series + indicators.

        ``indicators`` is a tuple of :class:`IndicatorSeries` — each is
        drawn as a line on the price plot. Names like ``"ema_20"``,
        ``"bollinger"`` get specific colours; unknown names use a
        neutral palette colour.
        """
        self._price_plot.clear()
        self._volume_plot.clear()

        if not series.candles:
            return

        # Keep the last ``visible_candles`` for clarity.
        recent = series.candles[-visible_candles:]
        n = len(recent)
        # X values are Unix timestamps in seconds (pyqtgraph DateAxisItem).
        xs = [c.open_time / 1000.0 for c in recent]

        # Candles.
        for i, candle in enumerate(recent):
            self._price_plot.addItem(
                CandlestickItem(xs[i], candle.open, candle.high, candle.low, candle.close)
            )

        # Indicators as lines.
        for ind in indicators:
            color_hex = self._indicator_color(ind.name)
            ys = []
            xs_ind = []
            for ts, snap in zip(series.candles[-visible_candles:], ind.snapshots[-visible_candles:], strict=True):
                v = snap.get(self._primary_value_name(ind.name))
                if v is None or math.isnan(v):
                    continue
                ys.append(float(v))
                xs_ind.append(ts.open_time / 1000.0)
            if not ys:
                continue
            pen = pg.mkPen(color_hex, width=2)
            self._price_plot.plot(xs_ind, ys, pen=pen, name=ind.name)

        # Volume bars.
        volumes = [c.volume for c in recent]
        is_bull = [c.close >= c.open for c in recent]
        brushes = []
        for bull in is_bull:
            color = QColor(
                NeonPalette.ACCENT_BULLISH if bull else NeonPalette.ACCENT_BEARISH
            )
            color.setAlpha(120)  # slight transparency for layered look
            brushes.append(pg.mkBrush(color))
        bar = pg.BarGraphItem(
            x=xs,
            height=volumes,
            width=(xs[1] - xs[0]) * 0.7 if n > 1 else 86_400 * 0.7,
            brushes=brushes,
        )
        self._volume_plot.addItem(bar)

        # Auto-range the price plot.
        self._price_plot.enableAutoRange(axis="y", enable=True)

    def clear(self) -> None:
        """Reset to empty state."""
        self._price_plot.clear()
        self._volume_plot.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plot_widget.setBackground(NeonPalette.BG_DARK)
        layout.addWidget(self._plot_widget)

        # Price plot — takes ~75% of vertical space.
        date_axis = pg.DateAxisItem(orientation="bottom")
        self._price_plot = self._plot_widget.addPlot(
            row=0, col=0, axisItems={"bottom": date_axis}
        )
        self._price_plot.showGrid(x=True, y=True, alpha=0.3)
        self._price_plot.setLabel("left", "Price")
        self._price_plot.getAxis("left").setTextPen(pg.mkPen(NeonPalette.TEXT_PRIMARY))
        self._price_plot.getAxis("bottom").setTextPen(pg.mkPen(NeonPalette.TEXT_PRIMARY))

        # Volume plot — fixed height, share X axis with price.
        vol_axis = pg.DateAxisItem(orientation="bottom")
        self._volume_plot = self._plot_widget.addPlot(
            row=1, col=0, axisItems={"bottom": vol_axis}
        )
        self._volume_plot.setMaximumHeight(120)
        self._volume_plot.setLabel("left", "Vol")
        self._volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self._volume_plot.setXLink(self._price_plot)

        # Default row stretch: price gets more space.
        self._plot_widget.ci.layout.setRowStretchFactor(0, 3)
        self._plot_widget.ci.layout.setRowStretchFactor(1, 1)

    @staticmethod
    def _indicator_color(name: str) -> str:
        """Pick a colour for an indicator line."""
        palette = {
            "ema_20": "#4fc3f7",     # cyan
            "ema_50": "#ab47bc",     # purple
            "ema_": "#4fc3f7",      # any EMA
            "rsi": "#ff9800",        # orange
            "macd": "#9c27b0",       # deep purple
            "bollinger": "#8a8f99",  # dim grey
            "atr": "#607d8b",        # blue-grey
            "volume_ma": "#607d8b",
        }
        for prefix, color in palette.items():
            if name.startswith(prefix):
                return color
        return NeonPalette.TEXT_DIM

    @staticmethod
    def _primary_value_name(indicator_name: str) -> str:
        """Map IndicatorSeries.name → primary field to plot.

        Most indicators have a single output. Multi-value ones
        (Bollinger, MACD) use their middle / main line.
        """
        if indicator_name.startswith("bollinger"):
            return "middle"
        if indicator_name.startswith("macd"):
            return "macd"
        return indicator_name.split("_")[0] if "_" in indicator_name else indicator_name
