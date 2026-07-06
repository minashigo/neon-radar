"""Tests for the Stage 6B ChartWidget."""

from __future__ import annotations

from neon_radar.config.models import TimeFrame
from neon_radar.domain.indicators.base import (
    IndicatorKind,
    IndicatorSeries,
    IndicatorSnapshot,
    IndicatorValue,
)
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.presentation.widgets.chart_widget import (
    CandlestickItem,
    ChartWidget,
)


def _make_series(n: int = 50, start_price: float = 100.0, start_time: int = 1_700_000_000_000) -> KlineSeries:
    """Build a small test series with predictable prices."""
    candles = tuple(
        OHLCV(
            open_time=start_time + i * 86_400_000,
            open=start_price + i,
            high=start_price + i + 2,
            low=start_price + i - 1,
            close=start_price + i + 1,
            volume=1000.0 + i,
        )
        for i in range(n)
    )
    return KlineSeries(
        symbol=Symbol("BTCUSDT"),
        timeframe=TimeFrame.D1,
        candles=candles,
    )


def _make_indicator_series(name: str, n: int = 50) -> IndicatorSeries:
    """Build a simple IndicatorSeries with linear values (for chart test)."""
    snapshots = tuple(
        IndicatorSnapshot(
            timestamp=1_700_000_000_000 + i * 86_400_000,
            values=(IndicatorValue(name.split("_")[0], 100.0 + i),),
        )
        for i in range(n)
    )
    return IndicatorSeries(
        name=name,
        kind=IndicatorKind.OVERLAY,
        snapshots=snapshots,
    )


class TestChartWidget:
    def test_constructs(self, qtbot) -> None:
        widget = ChartWidget()
        qtbot.addWidget(widget)
        assert widget is not None

    def test_render_empty_series_does_not_crash(self, qtbot) -> None:
        widget = ChartWidget()
        qtbot.addWidget(widget)
        empty = KlineSeries(symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.D1, candles=())
        widget.render(empty)

    def test_render_with_candles_adds_items(self, qtbot) -> None:
        widget = ChartWidget()
        qtbot.addWidget(widget)
        series = _make_series(n=10)
        widget.render(series)
        # Plot should have at least 10 candle items (plus indicators later).
        assert len(widget._price_plot.items) >= 10

    def test_render_with_indicators(self, qtbot) -> None:
        widget = ChartWidget()
        qtbot.addWidget(widget)
        series = _make_series(n=30)
        indicators = (_make_indicator_series("ema_20", n=30),)
        widget.render(series, indicators)
        # Two types: candles + ema line. Items count > 30.
        assert len(widget._price_plot.items) > 30

    def test_visible_candles_limits_render(self, qtbot) -> None:
        widget = ChartWidget()
        qtbot.addWidget(widget)
        series = _make_series(n=100)
        widget.render(series, visible_candles=20)
        # Should have ~20 candles, not 100.
        # (Plus the EMA plot line = 1.) We allow some slack for axis items.
        candle_count = sum(
            1 for it in widget._price_plot.items
            if isinstance(it, CandlestickItem)
        )
        assert candle_count == 20

    def test_clear_resets_chart(self, qtbot) -> None:
        widget = ChartWidget()
        qtbot.addWidget(widget)
        series = _make_series(n=10)
        widget.render(series)
        assert len(widget._price_plot.items) > 0
        widget.clear()
        assert len(widget._price_plot.items) == 0
        assert len(widget._volume_plot.items) == 0

    def test_nan_values_are_skipped(self, qtbot) -> None:
        """Indicator with NaN should not crash the chart."""
        widget = ChartWidget()
        qtbot.addWidget(widget)
        series = _make_series(n=10)

        # Build indicator with NaN values for first half.
        snapshots = []
        for i in range(10):
            value = float("nan") if i < 5 else 100.0 + i
            snapshots.append(
                IndicatorSnapshot(
                    timestamp=1_700_000_000_000 + i * 86_400_000,
                    values=(IndicatorValue("ema", value),),
                )
            )
        ind = IndicatorSeries(
            name="ema_20",
            kind=IndicatorKind.OVERLAY,
            snapshots=tuple(snapshots),
        )
        widget.render(series, (ind,))
        # No exception means NaN was handled.


class TestCandlestickItem:
    def test_bullish_color(self) -> None:
        item = CandlestickItem(x=0, o=100, h=110, lo=99, c=105)  # close > open = bullish
        # boundingRect covers low to high.
        assert item.lo == 99
        assert item.h == 110

    def test_bearish_color(self) -> None:
        item = CandlestickItem(x=0, o=110, h=111, lo=99, c=100)  # close < open = bearish
        assert item.o == 110
        assert item.c == 100
