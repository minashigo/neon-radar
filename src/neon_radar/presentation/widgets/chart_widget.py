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
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPicture
from PySide6.QtWidgets import QVBoxLayout, QWidget

from neon_radar.presentation.theme.neon_palette import NeonPalette

if TYPE_CHECKING:
    from neon_radar.domain.indicators.base import IndicatorSeries
    from neon_radar.domain.models import KlineSeries
    from neon_radar.domain.trading.setup import TradeSetup


class CandlestickItem(pg.GraphicsObject):
    """Batched candlestick chart item.

    Draws all candles into a single :class:`QPicture` for maximum
    performance during panning and zooming.
    """

    def __init__(self, data: list[tuple[float, float, float, float, float]]) -> None:
        """Initialize with data array.

        Args:
            data: List of (time, open, high, low, close) tuples.
        """
        super().__init__()
        self.data = data
        self._picture = QPicture()
        self._bounding_rect = pg.QtCore.QRectF()
        self._generate_picture()

    def _generate_picture(self) -> None:
        if not self.data:
            return

        painter = QPainter(self._picture)

        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        try:
            pen = QPen()
            pen.setWidth(1)

            # Width of the candle body based on timeframe spacing.
            # Using 0.6 standard width if t is unix timestamp in seconds.
            # If t is seconds, 1 hour = 3600. So width 0.6 is tiny!
            # Wait, previously we had x values in seconds?
            # Let's check: previous code used width 0.6 for x in seconds?
            # Actually, previous code:
            # `body = pg.QtCore.QRectF(self.x - 0.3, ..., 0.6, ...)`
            # But the volume bar used width `(xs[1] - xs[0]) * 0.7`.
            # Let's use `(xs[1] - xs[0]) * 0.7` for candle width too!

            if len(self.data) > 1:
                w = (self.data[1][0] - self.data[0][0]) * 0.7
            else:
                w = 86400 * 0.7

            for t, o, h, lo, c in self.data:
                is_bull = c >= o
                color_hex = NeonPalette.ACCENT_BULLISH if is_bull else NeonPalette.ACCENT_BEARISH
                color = QColor(color_hex)
                pen.setColor(color)
                painter.setPen(pen)
                painter.setBrush(QBrush(color))

                # Wick (high-low line).
                painter.drawLine(
                    pg.QtCore.QPointF(t, lo),
                    pg.QtCore.QPointF(t, h),
                )
                # Body (open-close rectangle).
                body_h = abs(c - o) if c != o else 0.001
                body_y = min(o, c)
                body = pg.QtCore.QRectF(
                    t - w / 2,
                    body_y,
                    w,
                    body_h,
                )
                painter.drawRect(body)

                min_x = min(min_x, t)
                max_x = max(max_x, t)
                min_y = min(min_y, lo)
                max_y = max(max_y, h)
        finally:
            painter.end()

        # Precompute bounding rect
        pad_x = w if "w" in locals() else 0.5
        self._bounding_rect = pg.QtCore.QRectF(
            min_x - pad_x, min_y, max_x - min_x + 2 * pad_x, max_y - min_y
        )

    def paint(self, painter, *_args) -> None:
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> pg.QtCore.QRectF:  # noqa: N802
        return self._bounding_rect

    def dataBounds(
        self, ax: int, frac: float = 1.0, orthoRange: tuple[float, float] | None = None
    ) -> tuple[float, float] | None:  # noqa: N802, N803
        """Provide accurate data bounds for auto-scaling.

        Args:
            ax: 0 for X-axis, 1 for Y-axis.
            frac: Ignored.
            orthoRange: The visible range of the orthogonal axis.
        """
        if not self.data:
            return None
        if ax == 0:
            return (self.data[0][0], self.data[-1][0])
        elif ax == 1:
            if orthoRange is not None:
                min_x, max_x = orthoRange
                visible = [d for d in self.data if min_x <= d[0] <= max_x]
                if visible:
                    return (min(d[3] for d in visible), max(d[2] for d in visible))
            return (min(d[3] for d in self.data), max(d[2] for d in self.data))
        return None


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
        trade_setup: TradeSetup | None = None,
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
        candle_data = [(xs[i], c.open, c.high, c.low, c.close) for i, c in enumerate(recent)]
        self._price_plot.addItem(CandlestickItem(candle_data))

        # Indicators as lines.
        from neon_radar.domain.indicators.base import IndicatorKind

        for ind in indicators:
            if ind.kind != IndicatorKind.OVERLAY:
                continue

            color_hex = self._indicator_color(ind.name)
            ys = []
            xs_ind = []
            for ts, snap in zip(
                series.candles[-visible_candles:], ind.snapshots[-visible_candles:], strict=True
            ):
                v = snap.get(self._primary_value_name(ind.name))
                if v is None or math.isnan(v):
                    continue
                ys.append(float(v))
                xs_ind.append(ts.open_time / 1000.0)
            if not ys:
                continue
            pen = pg.mkPen(color_hex, width=2)
            self._price_plot.plot(xs_ind, ys, pen=pen, name=ind.name)

        # Trade Setup Overlay.
        if trade_setup:
            color_hex = (
                NeonPalette.ACCENT_BULLISH
                if trade_setup.direction.name == "BULLISH"
                else NeonPalette.ACCENT_BEARISH
            )
            color = QColor(color_hex)

            # Entry
            entry_line = pg.InfiniteLine(
                pos=trade_setup.entry_price,
                angle=0,
                pen=pg.mkPen(color, width=2, style=Qt.PenStyle.DashLine),
                label="Entry",
                labelOpts={"position": 0.05, "color": color, "movable": True},
            )
            self._price_plot.addItem(entry_line)

            # Stop Loss
            sl_color = QColor(
                NeonPalette.ACCENT_BEARISH
                if trade_setup.direction.name == "BULLISH"
                else NeonPalette.ACCENT_BULLISH
            )
            sl_line = pg.InfiniteLine(
                pos=trade_setup.stop_loss,
                angle=0,
                pen=pg.mkPen(sl_color, width=2, style=Qt.PenStyle.DashLine),
                label="SL",
                labelOpts={"position": 0.05, "color": sl_color, "movable": True},
            )
            self._price_plot.addItem(sl_line)

            # TP1 & TP2
            tp_pen = pg.mkPen(color, width=2, style=Qt.PenStyle.DotLine)
            tp1_line = pg.InfiniteLine(
                pos=trade_setup.take_profit_1,
                angle=0,
                pen=tp_pen,
                label="TP1",
                labelOpts={"position": 0.05, "color": color, "movable": True},
            )
            self._price_plot.addItem(tp1_line)

            tp2_line = pg.InfiniteLine(
                pos=trade_setup.take_profit_2,
                angle=0,
                pen=tp_pen,
                label="TP2",
                labelOpts={"position": 0.05, "color": color, "movable": True},
            )
            self._price_plot.addItem(tp2_line)

        # Volume bars.
        volumes = [c.volume for c in recent]
        is_bull = [c.close >= c.open for c in recent]
        brushes = []
        for bull in is_bull:
            color = QColor(NeonPalette.ACCENT_BULLISH if bull else NeonPalette.ACCENT_BEARISH)
            color.setAlpha(120)  # slight transparency for layered look
            brushes.append(pg.mkBrush(color))
        bar = pg.BarGraphItem(
            x=xs,
            height=volumes,
            width=(xs[1] - xs[0]) * 0.7 if n > 1 else 86_400 * 0.7,
            brushes=brushes,
        )
        self._volume_plot.addItem(bar)

        # Auto-range the price plot dynamically as we pan
        self._price_plot.getViewBox().setAutoVisible(y=True)
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
        self._price_plot = self._plot_widget.addPlot(row=0, col=0, axisItems={"bottom": date_axis})
        self._price_plot.showGrid(x=True, y=True, alpha=0.3)
        self._price_plot.setLabel("left", "Price")
        self._price_plot.getAxis("left").setTextPen(pg.mkPen(NeonPalette.TEXT_PRIMARY))
        self._price_plot.getAxis("bottom").setTextPen(pg.mkPen(NeonPalette.TEXT_PRIMARY))

        # Volume plot — fixed height, share X axis with price.
        vol_axis = pg.DateAxisItem(orientation="bottom")
        self._volume_plot = self._plot_widget.addPlot(row=1, col=0, axisItems={"bottom": vol_axis})
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
            "ema_20": "#4fc3f7",  # cyan
            "ema_50": "#ab47bc",  # purple
            "ema_": "#4fc3f7",  # any EMA
            "rsi": "#ff9800",  # orange
            "macd": "#9c27b0",  # deep purple
            "bollinger": "#8a8f99",  # dim grey
            "atr": "#607d8b",  # blue-grey
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
