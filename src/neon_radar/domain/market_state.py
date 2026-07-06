"""``MarketState`` — the assembled snapshot the scoring engine evaluates.

Design notes
------------
* A :class:`MarketState` is a **value object**: it is fully immutable,
  and two states with the same contents are equal. This means we can
  cache them, hash them, and pass them around without defensive copies.
* The primary candle series is mandatory; the higher-timeframe series
  is optional (some analyses don't need it; some exchanges don't
  expose enough history for a higher TF). When present, it must be
  the *same symbol* and a *strictly higher timeframe*.
* Indicators are stored as a tuple of :class:`IndicatorSeries`. We
  deliberately do not key by indicator name — duplicate names are
  possible (e.g. ``ema_20`` and ``ema_50`` are both ``ema`` from the
  registry's perspective; they are different instances). Lookup is
  provided via :meth:`get_indicator` for the most common case.
* ``funding_rate`` and ``open_interest`` are optional because some
  sources (spot exchanges, some pairs on some venues) do not provide
  them. Scoring rules must tolerate ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neon_radar.domain.models import KlineSeries, Symbol, TickerStats

if TYPE_CHECKING:
    from neon_radar.config.models import TimeFrame
    from neon_radar.domain.funding import FundingRate, OpenInterest
    from neon_radar.domain.indicators.base import IndicatorSeries


@dataclass(slots=True, frozen=True)
class MarketState:
    """The complete, immutable view of a market at one point in time.

    Parameters
    ----------
    symbol
        The trading pair this state describes.
    timestamp
        Unix ms when the state was assembled (not when the data was
        originally produced — assembly time is what the analysis cares
        about).
    primary_series
        The main candle series (e.g. 4H) the user is looking at.
    higher_tf_series
        Optional higher-TF context (e.g. 1D when primary is 4H). Must
        be strictly higher than ``primary_series.timeframe``.
    indicator_series
        Pre-computed indicators. The scoring engine expects the
        indicators it needs to be present; missing indicators yield
        ``None`` from :meth:`get_indicator`.
    ticker
        24h ticker statistics. Optional — not every call site fetches
        it.
    funding_rate
        Current funding rate (decimal). Optional.
    open_interest
        Current open interest. Optional.
    """

    symbol: Symbol
    timestamp: int
    primary_series: KlineSeries
    higher_tf_series: KlineSeries | None = None
    indicator_series: tuple[IndicatorSeries, ...] = ()
    ticker: TickerStats | None = None
    funding_rate: FundingRate | None = None
    open_interest: OpenInterest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, Symbol):
            object.__setattr__(self, "symbol", Symbol(self.symbol))
        if not isinstance(self.primary_series, KlineSeries):
            raise TypeError("primary_series must be a KlineSeries")
        if self.primary_series.symbol != self.symbol:
            raise ValueError(
                f"MarketState symbol ({self.symbol}) does not match "
                f"primary_series.symbol ({self.primary_series.symbol})"
            )
        if self.higher_tf_series is not None:
            if not isinstance(self.higher_tf_series, KlineSeries):
                raise TypeError("higher_tf_series must be a KlineSeries or None")
            if self.higher_tf_series.symbol != self.symbol:
                raise ValueError(
                    f"higher_tf_series symbol ({self.higher_tf_series.symbol}) "
                    f"does not match state symbol ({self.symbol})"
                )
            if not self._is_higher_tf(
                self.primary_series.timeframe, self.higher_tf_series.timeframe
            ):
                raise ValueError(
                    f"higher_tf_series.timeframe ({self.higher_tf_series.timeframe.value}) "
                    f"must be strictly higher than primary_series.timeframe "
                    f"({self.primary_series.timeframe.value})"
                )

    @staticmethod
    def _is_higher_tf(primary: TimeFrame, higher: TimeFrame) -> bool:
        return higher.seconds > primary.seconds

    # -- Lookup helpers ----------------------------------------------------

    def get_indicator(self, name: str) -> IndicatorSeries | None:
        """Return the first indicator series matching ``name`` or ``None``.

        "First" because two indicators with the same name are allowed
        (e.g. EMA(20) and EMA(50)). For most use cases there's only one
        per name.
        """
        for ind in self.indicator_series:
            if ind.name == name:
                return ind
        return None

    def get_indicator_value(self, name: str, field: str = "") -> float | None:
        """Convenience: latest value of an indicator.

        Parameters
        ----------
        name
            Indicator series name (e.g. ``"ema"``).
        field
            Sub-output name for multi-value indicators (e.g. ``"signal"``
            for MACD). If empty, returns the **first** value.
        """
        ind = self.get_indicator(name)
        if ind is None:
            return None
        snap = ind.latest()
        if snap is None:
            return None
        if not field:
            return snap.values[0].value if snap.values else None
        return snap.get(field)
