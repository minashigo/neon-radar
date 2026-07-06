"""Indicator pipeline — compute multiple indicators over one series.

A thin orchestrator that turns a list of :class:`IndicatorSpec` into
a list of :class:`IndicatorSeries`. The UI and the scoring engine
both consume the result.

Design notes
------------
* This is a **function** plus a small ``IndicatorSpec`` dataclass,
  not a class. State-less orchestration is easier to test and reason
  about.
* Each spec is built **fresh** on every call so spec objects can be
  reused across many ``compute_indicators`` invocations without
  leaking mutable state.
* Empty spec list returns empty list — no overhead.
* ``tag`` on an :class:`IndicatorSpec` controls the
  :class:`IndicatorSeries` name. This is how the scoring engine
  distinguishes, for example, ``EMA(20)`` from ``EMA(50)`` — both
  register as ``"ema"`` in the global registry, but with distinct
  tags they get unique names like ``"ema_20"`` and ``"ema_50"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from neon_radar.domain.indicators.base import Indicator, IndicatorRegistry, IndicatorSeries

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neon_radar.domain.models import KlineSeries


@dataclass(frozen=True)
class IndicatorSpec:
    """Specification for one indicator in a pipeline.

    Attributes
    ----------
    name
        Registered indicator name (e.g. ``"ema"``).
    params
        Constructor kwargs for the indicator class. Defaults to
        ``{}`` which uses the indicator's own defaults.
    tag
        Optional suffix appended to ``name`` to form the resulting
        :class:`IndicatorSeries` ``name`` (``"ema"`` + tag ``"20"``
        → ``"ema_20"``). Use this when multiple instances of the
        same indicator class coexist (different periods, parameters).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    tag: str | None = None

    @property
    def series_name(self) -> str:
        """Name to use for the resulting ``IndicatorSeries``."""
        return f"{self.name}_{self.tag}" if self.tag else self.name

    def build(self) -> Indicator:
        """Instantiate the indicator class with the configured params."""
        try:
            cls = IndicatorRegistry.get(self.name)
        except KeyError as exc:
            raise ValueError(
                f"Cannot build indicator '{self.name}': not registered. "
                f"Available: {IndicatorRegistry.names()}"
            ) from exc
        return cls(**self.params)


def compute_indicators(
    series: KlineSeries,
    specs: Iterable[IndicatorSpec],
) -> list[IndicatorSeries]:
    """Compute every spec's indicator over ``series``.

    Order of the output matches the order of ``specs``. Empty input
    yields an empty list — no work is done.

    Each indicator is computed with its :attr:`IndicatorSpec.series_name`
    so that the resulting :class:`IndicatorSeries` can be looked up
    unambiguously even when multiple instances of the same class
    coexist.
    """
    results: list[IndicatorSeries] = []
    for spec in specs:
        indicator = spec.build()
        results.append(indicator.compute(series, name=spec.series_name))
    return results


def available_indicators() -> tuple[str, ...]:
    """Return all registered indicator names (alphabetical)."""
    return tuple(sorted(IndicatorRegistry.names()))
