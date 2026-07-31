"""Noise filtering logic for Market Intelligence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.market_intelligence.enums import SourceReliability

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neon_radar.domain.market_intelligence.models import IntelligenceSignal


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

    def filter_signals(
        self, signals: Iterable[IntelligenceSignal]
    ) -> tuple[IntelligenceSignal, ...]:
        """Apply noise filtering rules to a stream of signals."""
        from collections import defaultdict

        # Use event_timestamp for deduplication logic as it represents when the event actually happened
        signals_list = sorted(signals, key=lambda s: s.event_timestamp)

        if not signals_list:
            return ()

        # 1. Filter out low reliability
        reliable_signals = [
            s
            for s in signals_list
            if s.weight >= self.min_reliability_threshold
            or s.reliability in (SourceReliability.OFFICIAL, SourceReliability.INSTITUTIONAL)
        ]

        # 2. Deduplicate: same type, same direction, same provider_name within time_window
        provider_signals: dict[tuple[str, str], list[tuple[int, IntelligenceSignal]]] = defaultdict(
            list
        )

        for current in reliable_signals:
            key = (current.type.name, current.provider_name)
            group = provider_signals[key]

            if group:
                anchor_ts, prev_sig = group[-1]
                time_diff = current.event_timestamp - anchor_ts
                if time_diff <= self.time_window_ms and (
                    prev_sig.direction * current.direction > 0
                    or prev_sig.direction == current.direction
                ):
                    # Duplicate found. Replace the old one with the newer one, but keep the anchor!
                    group[-1] = (anchor_ts, current)
                    continue

            # If not a duplicate within the window, add it
            group.append((current.event_timestamp, current))

        # Flatten and re-sort
        deduped = [s for group in provider_signals.values() for _, s in group]
        deduped.sort(key=lambda s: s.event_timestamp)

        # 3. Independent confirmation
        if self.require_independent_confirmation:
            confirmed: list[IntelligenceSignal] = []

            # Group by (type, direction_sign)
            direction_groups: dict[tuple[str, int], list[IntelligenceSignal]] = defaultdict(list)
            for s in deduped:
                direction_sign = 1 if s.direction > 0 else (-1 if s.direction < 0 else 0)
                direction_groups[(s.type.name, direction_sign)].append(s)

            for s in deduped:
                # High reliability sources bypass confirmation requirement
                if s.reliability in (SourceReliability.OFFICIAL, SourceReliability.INSTITUTIONAL):
                    confirmed.append(s)
                    continue

                direction_sign = 1 if s.direction > 0 else (-1 if s.direction < 0 else 0)
                group = direction_groups[(s.type.name, direction_sign)]

                # Count distinct providers in the same group within window
                providers = set()
                for other in group:
                    if abs(other.event_timestamp - s.event_timestamp) <= self.time_window_ms:
                        providers.add(other.provider_name)

                if len(providers) >= 2:
                    confirmed.append(s)

            return tuple(confirmed)

        return tuple(deduped)
