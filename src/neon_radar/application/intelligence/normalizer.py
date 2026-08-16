"""Neutral Market Intelligence feature normalizer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.market_intelligence.history import IntelligenceSignalSeries


class IntelligenceNormalizer:
    """Calculates point-in-time statistical features from a sliced IntelligenceSignalSeries.

    Does NOT map features to specific directions or bounds.
    Requires the series to ALREADY be sliced by `available_at <= T` to avoid Look-Ahead Bias.
    """

    @staticmethod
    def extract_raw_value(series: IntelligenceSignalSeries, index: int = -1) -> float | None:
        """Extract the raw value from the specified observation."""
        if not series.items:
            return None

        try:
            obs = series.items[index]
            raw_str = obs.signal.metadata.get("raw_value")
            if raw_str is None:
                return None
            return float(raw_str)
        except (ValueError, TypeError, IndexError):
            return None

    @staticmethod
    def calculate_rolling_z_score(series: IntelligenceSignalSeries, window_size: int) -> float | None:
        """Calculate the Z-Score of the most recent observation against a rolling window.

        Formula: (Current_Value - Mean(Window)) / StdDev(Window)
        Returns None if not enough data points (needs at least 2 points).
        """
        if window_size < 2 or len(series.items) < 2:
            return None

        # Take up to `window_size` items from the end of the SLICED series
        window_items = series.items[-window_size:]

        values = []
        for obs in window_items:
            try:
                raw_str = obs.signal.metadata.get("raw_value")
                if raw_str is not None:
                    values.append(float(raw_str))
            except (ValueError, TypeError):
                continue

        if len(values) < 2:
            return None

        current_val = values[-1]

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1) # Sample variance

        if variance == 0.0:
            return 0.0

        std_dev = math.sqrt(variance)
        return (current_val - mean) / std_dev

    @staticmethod
    def calculate_percentile(series: IntelligenceSignalSeries, window_size: int) -> float | None:
        """Calculate the percentile rank of the most recent observation within the rolling window.

        Returns a float between 0.0 and 1.0.
        """
        if window_size < 1 or not series.items:
            return None

        window_items = series.items[-window_size:]

        values = []
        for obs in window_items:
            try:
                raw_str = obs.signal.metadata.get("raw_value")
                if raw_str is not None:
                    values.append(float(raw_str))
            except (ValueError, TypeError):
                continue

        if not values:
            return None

        current_val = values[-1]

        # Count strictly less
        less_count = sum(1 for v in values if v < current_val)
        # Count equal
        equal_count = sum(1 for v in values if v == current_val)

        # Standard percentile formula
        percentile = (less_count + 0.5 * equal_count) / len(values)
        return percentile
