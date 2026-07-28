"""Noise filtering logic for Market Intelligence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.market_intelligence.enums import SourceReliability

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neon_radar.domain.market_intelligence.models import SignalEvidence


class NoiseFilter:
    """Filters out noise, hype, and duplicate signals from intelligence feeds."""

    def __init__(
        self,
        min_reliability_threshold: float = 0.2,
        time_window_ms: int = 3600000,  # 1 hour by default for deduplication
        require_independent_confirmation: bool = True,
    ) -> None:
        """Initialize the noise filter.

        Args:
            min_reliability_threshold: Minimum weight/reliability a source must have.
            time_window_ms: Time window in ms to consider signals as duplicates.
            require_independent_confirmation: If True, requires at least two distinct
                providers for a specific signal type and direction to pass it through,
                unless the source is highly reliable.
        """
        self.min_reliability_threshold = min_reliability_threshold
        self.time_window_ms = time_window_ms
        self.require_independent_confirmation = require_independent_confirmation

    def filter_signals(self, signals: Iterable[SignalEvidence]) -> tuple[SignalEvidence, ...]:
        """Apply noise filtering rules to a stream of signals."""
        signals_list = sorted(signals, key=lambda s: s.timestamp)

        if not signals_list:
            return ()

        # 1. Filter out low reliability
        reliable_signals = [
            s for s in signals_list
            if s.source.weight >= self.min_reliability_threshold
            or s.source.reliability in (SourceReliability.OFFICIAL, SourceReliability.INSTITUTIONAL)
        ]

        # 2. Deduplicate: same type, same direction, same provider_name within time_window
        deduped: list[SignalEvidence] = []
        for current in reliable_signals:
            is_duplicate = False
            for prev in reversed(deduped):
                time_diff = current.timestamp - prev.timestamp
                if time_diff > self.time_window_ms:
                    break  # Outside window, no more duplicates to check

                # Check for exact duplicate criteria
                if (
                    prev.type == current.type
                    and prev.source.provider_name == current.source.provider_name
                    # Consider same direction as duplicate if from same provider
                    and (prev.direction * current.direction > 0 or prev.direction == current.direction)
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped.append(current)

        # 3. Independent confirmation
        if self.require_independent_confirmation:
            confirmed: list[SignalEvidence] = []
            # Group by (type, direction_sign)
            # Find distinct providers for each group
            for s in deduped:
                # High reliability sources bypass confirmation requirement
                if s.source.reliability in (SourceReliability.OFFICIAL, SourceReliability.INSTITUTIONAL):
                    confirmed.append(s)
                    continue

                direction_sign = 1 if s.direction > 0 else (-1 if s.direction < 0 else 0)

                # Count distinct providers in the same group within window
                providers = set()
                for other in deduped:
                    other_sign = 1 if other.direction > 0 else (-1 if other.direction < 0 else 0)
                    if (
                        other.type == s.type
                        and other_sign == direction_sign
                        and abs(other.timestamp - s.timestamp) <= self.time_window_ms
                    ):
                        providers.add(other.source.provider_name)

                if len(providers) >= 2:
                    confirmed.append(s)

            return tuple(confirmed)

        return tuple(deduped)
