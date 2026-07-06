"""Tests for the Indicator abstraction and registry.

Note: this module does NOT auto-clear the registry, because doing so
breaks later tests that rely on the built-in indicators (EMA, RSI,
…) being registered. Each test that registers a custom indicator
uses the ``_restore_registry`` fixture which restores the registry
after the test runs.
"""

from __future__ import annotations

import math

import pytest

from neon_radar.config.models import TimeFrame

# Built-in indicators are imported here so they auto-register on
# module load. Tests that register custom names must clear() first.
from neon_radar.domain.indicators import (  # noqa: F401  -- side-effect import
    EMA,
    RSI,
    SMA,
)
from neon_radar.domain.indicators.base import (
    Indicator,
    IndicatorKind,
    IndicatorRegistry,
    IndicatorSeries,
    IndicatorSnapshot,
    IndicatorValue,
)
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol


def _make_series(n: int = 30, start_price: float = 100.0) -> KlineSeries:
    candles = tuple(
        OHLCV(
            open_time=1_700_000_000_000 + i * 86_400_000,
            open=start_price + i,
            high=start_price + i + 2,
            low=start_price + i - 1,
            close=start_price + i + 1,
            volume=1000.0 + i,
        )
        for i in range(n)
    )
    return KlineSeries(symbol=Symbol("BTCUSDT"), timeframe=TimeFrame.D1, candles=candles)


# ---------------------------------------------------------------------------
# IndicatorValue / Snapshot / Series
# ---------------------------------------------------------------------------


class TestIndicatorValue:
    def test_construct(self) -> None:
        v = IndicatorValue(name="ema", value=123.4)
        assert v.name == "ema"
        assert v.value == 123.4

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            IndicatorValue(name="", value=0.0)


class TestIndicatorSnapshot:
    def test_get_returns_value(self) -> None:
        snap = IndicatorSnapshot(
            timestamp=1_000_000,
            values=(
                IndicatorValue("macd", 1.5),
                IndicatorValue("signal", 1.0),
            ),
        )
        assert snap.get("macd") == 1.5
        assert snap.get("signal") == 1.0
        assert snap.get("histogram") is None

    def test_empty_snapshot(self) -> None:
        snap = IndicatorSnapshot(timestamp=1_000_000, values=())
        assert snap.get("anything") is None


class TestIndicatorSeries:
    def test_length(self) -> None:
        snaps = tuple(
            IndicatorSnapshot(timestamp=1_700_000_000_000 + i, values=(IndicatorValue("v", float(i)),))
            for i in range(5)
        )
        s = IndicatorSeries(name="test", kind=IndicatorKind.META, snapshots=snaps)
        assert len(s) == 5

    def test_latest_returns_last(self) -> None:
        snaps = (
            IndicatorSnapshot(timestamp=1, values=(IndicatorValue("v", 1.0),)),
            IndicatorSnapshot(timestamp=2, values=(IndicatorValue("v", 2.0),)),
        )
        s = IndicatorSeries(name="t", kind=IndicatorKind.META, snapshots=snaps)
        assert s.latest() is snaps[1]
        assert s.latest_value("v") == 2.0

    def test_latest_on_empty(self) -> None:
        s = IndicatorSeries(name="t", kind=IndicatorKind.META, snapshots=())
        assert s.latest() is None
        assert s.latest_value("v") is None

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            IndicatorSeries(name="", kind=IndicatorKind.META, snapshots=())


# ---------------------------------------------------------------------------
# Indicator ABC + Registry
# ---------------------------------------------------------------------------


class _StubIndicator(Indicator):
    KIND = IndicatorKind.META

    def compute(self, series: KlineSeries) -> IndicatorSeries:
        snaps = tuple(
            IndicatorSnapshot(
                timestamp=c.open_time,
                values=(IndicatorValue("v", float(i)),),
            )
            for i, c in enumerate(series)
        )
        return IndicatorSeries(name=self.NAME, kind=self.KIND, snapshots=snaps)


@pytest.fixture
def _restore_registry():
    """Save the registry before the test, restore after. Use this for
    tests that mutate global registry state via :meth:`register`.
    """
    saved = dict(IndicatorRegistry._items)
    yield
    IndicatorRegistry._items.clear()
    IndicatorRegistry._items.update(saved)


class TestIndicatorRegistry:
    def test_register_and_get(self, _restore_registry) -> None:
        @IndicatorRegistry.register("stub")
        class Stub(_StubIndicator):
            pass

        assert IndicatorRegistry.is_registered("stub")
        cls = IndicatorRegistry.get("stub")
        assert cls is Stub
        assert cls.NAME == "stub"

    def test_register_then_use(self, _restore_registry) -> None:
        @IndicatorRegistry.register("stub")
        class Stub(_StubIndicator):
            pass

        instance = Stub()
        assert instance.NAME == "stub"
        result = instance.compute(_make_series(n=10))
        assert len(result) == 10
        assert result.latest_value("v") == 9.0

    def test_rejects_empty_name(self, _restore_registry) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            IndicatorRegistry.register("")(_StubIndicator)

    def test_rejects_duplicate_name(self, _restore_registry) -> None:
        @IndicatorRegistry.register("dup")
        class A(_StubIndicator):
            pass

        with pytest.raises(ValueError, match="Duplicate"):

            @IndicatorRegistry.register("dup")
            class B(_StubIndicator):
                pass

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown indicator"):
            IndicatorRegistry.get("nope")

    def test_all_and_names(self, _restore_registry) -> None:
        @IndicatorRegistry.register("first")
        class First(_StubIndicator):
            pass

        @IndicatorRegistry.register("second")
        class Second(_StubIndicator):
            pass

        names = IndicatorRegistry.names()
        assert names[-2:] == ("first", "second")
        assert len(IndicatorRegistry.all()) >= 2

    def test_clear(self, _restore_registry) -> None:
        @IndicatorRegistry.register("x")
        class X(_StubIndicator):
            pass

        IndicatorRegistry.clear()
        assert IndicatorRegistry.names() == ()
        assert not IndicatorRegistry.is_registered("x")

    def test_indicator_without_decorator_has_no_name(self) -> None:
        class Bare(_StubIndicator):
            pass

        assert Bare.NAME == ""


class TestIndicatorNaNWarmup:
    """The first ``period - 1`` snapshots should be NaN; the rest finite."""

    def test_warmup_via_compute(self, _restore_registry) -> None:
        @IndicatorRegistry.register("warm")
        class Warm(_StubIndicator):
            KIND = IndicatorKind.OVERLAY

            def __init__(self, period: int = 3) -> None:
                self.period = period

            def compute(self, series: KlineSeries) -> IndicatorSeries:
                snaps = []
                for i, c in enumerate(series):
                    val = float("nan") if i < self.period - 1 else float(i)
                    snaps.append(
                        IndicatorSnapshot(
                            timestamp=c.open_time,
                            values=(IndicatorValue("v", val),),
                        )
                    )
                return IndicatorSeries(
                    name=self.NAME, kind=self.KIND, snapshots=tuple(snaps)
                )

        result = Warm(period=3).compute(_make_series(n=5))
        assert math.isnan(result.snapshots[0].get("v"))  # warm
        assert math.isnan(result.snapshots[1].get("v"))  # warm
        assert result.snapshots[2].get("v") == 2.0  # first valid
        assert result.snapshots[4].get("v") == 4.0  # last
