"""Indicator abstractions — the heart of the analysis pipeline.

Design notes
------------
* :class:`Indicator` is an **abstract base class**. Every concrete
  indicator (EMA, RSI, …) subclasses it.
* :class:`IndicatorRegistry` is a tiny, **append-only** registry.
  Adding a new indicator is a one-line decorator::

      @IndicatorRegistry.register("rsi")
      class RSI(Indicator):
          KIND = IndicatorKind.OSCILLATOR
          def __init__(self, period: int = 14) -> None: ...
          def compute(self, series: KlineSeries) -> IndicatorSeries: ...

  No other file in the project needs to be touched. This is the
  Open/Closed Principle in its most concrete form for our domain.

* Indicators are **instances, not classes**. The class is the recipe
  (parameterised, e.g. ``EMA(period=20)``); the registry stores the
  recipe; the engine instantiates with the configured parameters.

* :class:`IndicatorSeries` is aligned to the input :class:`KlineSeries`:
  the i-th snapshot corresponds to the i-th candle. Snapshots whose
  inputs are still "warming up" (e.g. the first 19 candles for an
  EMA(20)) carry ``NaN`` in their values. Callers can detect this via
  :func:`math.isnan`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from neon_radar.domain.models import KlineSeries


class IndicatorKind(StrEnum):
    """How an indicator is rendered on the chart.

    * ``OVERLAY`` — drawn on top of the price candle (EMA, SMA, BB)
    * ``OSCILLATOR`` — drawn in its own pane (RSI, MACD)
    * ``META`` — not drawn; consumed by other indicators or scoring rules
      (ATR, raw volume, funding rate)
    """

    OVERLAY = "overlay"
    OSCILLATOR = "oscillator"
    META = "meta"


@dataclass(slots=True, frozen=True)
class IndicatorValue:
    """A single named value inside an indicator snapshot.

    Using a small dataclass instead of a tuple keeps the API self-
    documenting and works nicely with type-checkers.
    """

    name: str
    value: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("IndicatorValue.name must not be empty")


@dataclass(slots=True, frozen=True)
class IndicatorSnapshot:
    """One point of an indicator, aligned to a single candle.

    Attributes
    ----------
    timestamp
        Open time of the candle this snapshot corresponds to (Unix ms).
    values
        Ordered tuple of named values. Order matters because the chart
        legend relies on it. For multi-output indicators (MACD, BB) the
        tuple contains several values.
    """

    timestamp: int
    values: tuple[IndicatorValue, ...]

    def get(self, name: str) -> float | None:
        """Return the value named ``name`` or ``None`` if not present."""
        for v in self.values:
            if v.name == name:
                return v.value
        return None


@dataclass(slots=True, frozen=True)
class IndicatorSeries:
    """Time-aligned series of indicator snapshots for one indicator run.

    The series has the same length as the input :class:`KlineSeries`
    (snapshots may carry NaN values during the warm-up period).
    """

    name: str
    kind: IndicatorKind
    snapshots: tuple[IndicatorSnapshot, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("IndicatorSeries.name must not be empty")

    def __len__(self) -> int:
        return len(self.snapshots)

    def latest(self) -> IndicatorSnapshot | None:
        """Return the last non-empty snapshot or ``None``."""
        return self.snapshots[-1] if self.snapshots else None

    def latest_value(self, name: str) -> float | None:
        """Convenience: latest value of sub-output ``name``."""
        snap = self.latest()
        return snap.get(name) if snap else None


class Indicator(ABC):
    """Abstract base class for all technical indicators.

    To add a new indicator:

    1. Create ``domain/indicators/<name>.py``.
    2. Subclass :class:`Indicator`.
    3. Decorate with :meth:`IndicatorRegistry.register`.
    4. Implement :meth:`compute`.

    No other file needs to change. The pipeline picks it up
    automatically.
    """

    # Subclasses MUST set these. ``NAME`` is set by the registry decorator.
    NAME: ClassVar[str] = ""
    KIND: ClassVar[IndicatorKind] = IndicatorKind.META

    @abstractmethod
    def compute(
        self,
        series: KlineSeries,
        *,
        name: str | None = None,
    ) -> IndicatorSeries:
        """Compute the indicator over ``series``.

        Parameters
        ----------
        series
            Input candles.
        name
            Override for the resulting :class:`IndicatorSeries` name.
            Useful when several instances of the same class coexist
            (e.g. ``EMA(20)`` and ``EMA(50)`` — the orchestrator
            names them ``"ema_20"`` and ``"ema_50"`` so that rules
            can look them up unambiguously). If ``None``, the
            class's :attr:`NAME` is used.

        Returns
        -------
        IndicatorSeries
            Series of the **same length** as ``series``. Snapshots
            whose values are not yet defined (warm-up period) contain
            ``NaN`` floats.
        """

    @property
    def name(self) -> str:
        """Convenience accessor — usually equal to ``self.NAME``."""
        return self.NAME


class IndicatorRegistry:
    """Central, append-only registry of available indicators.

    The registry is intentionally global. Indicators are singletons in
    the sense that we don't want two EMA implementations fighting over
    the same name. If the global state bothers you in tests, just
    clear :attr:`_items` between tests.
    """

    _items: ClassVar[dict[str, type[Indicator]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Indicator]], type[Indicator]]:
        """Class decorator that registers an :class:`Indicator` subclass.

        Usage::

            @IndicatorRegistry.register("ema")
            class EMA(Indicator):
                KIND = IndicatorKind.OVERLAY
                ...
        """

        def decorator(indicator_cls: type[Indicator]) -> type[Indicator]:
            if not name or not name.strip():
                raise ValueError("Indicator name must be a non-empty string")
            if name in cls._items:
                raise ValueError(
                    f"Duplicate indicator name '{name}'. "
                    f"Already registered: {cls._items[name].__name__}"
                )
            indicator_cls.NAME = name
            cls._items[name] = indicator_cls
            return indicator_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[Indicator]:
        """Return the indicator class registered under ``name``."""
        try:
            return cls._items[name]
        except KeyError as exc:
            raise KeyError(f"Unknown indicator '{name}'. Registered: {sorted(cls._items)}") from exc

    @classmethod
    def all(cls) -> tuple[type[Indicator], ...]:
        """Return all registered indicator classes."""
        return tuple(cls._items.values())

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Return all registered names in insertion order."""
        return tuple(cls._items.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations. For tests only."""
        cls._items.clear()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._items
