"""Domain models for Historical Market Intelligence and Point-In-Time evaluation."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neon_radar.domain.market_intelligence.models import IntelligenceSignal


@dataclass(slots=True, frozen=True)
class IntelligenceObservation:
    """A single historical observation of a Market Intelligence signal.

    Attributes
    ----------
    signal
        The raw intelligence signal.
    observation_timestamp
        Unix ms (UTC) when the metric represents the observed market state.
    available_at
        Unix ms (UTC) when the value became physically available for use by the system.
        This must be strictly used for Point-in-Time slicing to prevent Look-Ahead Bias.
    """

    signal: IntelligenceSignal
    observation_timestamp: int
    available_at: int


T_Obs = TypeVar("T_Obs", bound=IntelligenceObservation)


@dataclass(slots=True, frozen=True)
class IntelligenceSignalSeries:
    """A historical series of intelligence observations of a specific type.

    Provides Point-in-Time slicing and forward-fill latest-known-value semantics.
    """

    signal_type: str
    items: tuple[IntelligenceObservation, ...]

    def __post_init__(self) -> None:
        if self.items:
            # Enforce sorting by available_at for binary search slicing
            times = [obs.available_at for obs in self.items]
            if times != sorted(times):
                raise ValueError(
                    f"IntelligenceSignalSeries items are not sorted ascending by available_at for {self.signal_type}"
                )

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[IntelligenceObservation]:
        return iter(self.items)

    def __getitem__(self, index: int | slice) -> IntelligenceObservation | tuple[IntelligenceObservation, ...]:
        return self.items[index]

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    def slice_by_availability(self, max_timestamp: int) -> IntelligenceSignalSeries:
        """Return a new series excluding items that became available after ``max_timestamp``.

        This is the critical Point-in-Time safeguard.
        """
        if not self.items:
            return self

        publish_times = [obs.available_at for obs in self.items]
        idx = bisect.bisect_right(publish_times, max_timestamp)
        valid_items = self.items[:idx]

        return replace(self, items=valid_items)

    def latest_valid_observation(self, timestamp: int, max_staleness_ms: int) -> IntelligenceObservation | None:
        """Get the latest observation available at ``timestamp``, respecting max staleness.

        Implements safe Forward-Fill semantics.
        """
        sliced = self.slice_by_availability(timestamp)
        if not sliced.items:
            return None

        latest = sliced.items[-1]
        if timestamp - latest.available_at > max_staleness_ms:
            return None

        return latest


@dataclass(slots=True, frozen=True)
class HistoricalIntelligenceContext:
    """Aggregate root containing all historical MI series.

    Attributes
    ----------
    timestamp
        Evaluation timestamp (Unix ms). The time this historical context is built for.
    """

    timestamp: int
    series_map: dict[str, IntelligenceSignalSeries]

    def __post_init__(self) -> None:
        """Enforce strict Point-in-Time correctness for all series."""
        safe_map = {}
        for sig_type, series in self.series_map.items():
            safe_map[sig_type] = series.slice_by_availability(self.timestamp)

        # Bypass frozen constraints to set the sliced map
        object.__setattr__(self, "series_map", safe_map)

    def slice_at(self, timestamp: int) -> HistoricalIntelligenceContext:
        """Return a new HistoricalIntelligenceContext sliced exactly at the given timestamp."""
        return replace(self, timestamp=timestamp)
