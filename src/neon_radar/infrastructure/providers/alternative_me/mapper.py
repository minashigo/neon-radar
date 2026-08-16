"""Mapper for Alternative.me API responses."""

import logging
from typing import Any

from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType, SourceReliability
from neon_radar.domain.market_intelligence.history import IntelligenceObservation
from neon_radar.domain.market_intelligence.models import IntelligenceSignal

logger = logging.getLogger(__name__)


def map_fng_to_signal(data: dict[str, Any], ingestion_timestamp: int) -> IntelligenceSignal | None:
    """Map Alternative.me Fear & Greed API response to IntelligenceSignal.

    Formula:
    Fear & Greed index ranges from 0 (Extreme Fear) to 100 (Extreme Greed).
    We normalize this to a direction of [-1.0, 1.0].

    direction = (value - 50) / 50
    strength = abs(direction)
    """
    try:
        if "data" not in data or not data["data"]:
            return None

        # Check for logical errors indicated by the API
        metadata = data.get("metadata", {})
        if metadata.get("error") is not None:
            logger.warning(f"Alternative.me API returned an error in metadata: {metadata['error']}")
            return None

        latest_item = data["data"][0]
        value_str = latest_item.get("value")

        if value_str is None:
            return None

        value = float(value_str)

        # Calculate direction and strength
        direction = (value - 50.0) / 50.0

        # Bound the direction just in case API returns unexpected values
        direction = max(-1.0, min(1.0, direction))
        strength = abs(direction)

        return IntelligenceSignal(
            type=IntelligenceSignalType.FEAR_AND_GREED,
            direction=direction,
            strength=strength,
            event_timestamp=int(latest_item.get("timestamp", ingestion_timestamp)) * 1000 if "timestamp" in latest_item else ingestion_timestamp,
            ingestion_timestamp=ingestion_timestamp,
            source_id="alternative_me",
            provider_name="AlternativeMe",
            provider_type="API",
            reliability=SourceReliability.ANALYTICS,
            weight=1.0,
            metadata={
                "symbol": "BTC",
                "raw_value": value_str,
                "value_classification": latest_item.get("value_classification", "Unknown"),
            },
        )
    except Exception as e:
        logger.warning(f"Failed to map Fear & Greed data: {e}")
        return None

def map_historical_fng_to_observations(data: dict[str, Any], ingestion_timestamp: int) -> tuple[IntelligenceObservation, ...]:
    """Map historical F&G data to point-in-time observations."""
    try:
        if "data" not in data or not data["data"]:
            return ()

        observations = []
        for item in data["data"]:
            value_str = item.get("value")
            if value_str is None:
                continue
            value = float(value_str)
            direction = max(-1.0, min(1.0, (value - 50.0) / 50.0))

            # Alternative.me timestamps are seconds, we need ms
            evt_sec = int(item["timestamp"])
            obs_time = evt_sec * 1000

            # Conservative available_at: The index for the day is published exactly at the timestamp (00:00:00 UTC).
            available_at = obs_time

            sig = IntelligenceSignal(
                type=IntelligenceSignalType.FEAR_AND_GREED,
                direction=direction,
                strength=abs(direction),
                event_timestamp=obs_time,
                ingestion_timestamp=ingestion_timestamp,
                source_id="alternative_me",
                provider_name="AlternativeMe",
                provider_type="API",
                reliability=SourceReliability.ANALYTICS,
                weight=1.0,
                metadata={
                    "symbol": "BTC",
                    "raw_value": value_str,
                    "value_classification": item.get("value_classification", "Unknown"),
                },
            )

            observations.append(
                IntelligenceObservation(
                    signal=sig,
                    observation_timestamp=obs_time,
                    available_at=available_at,
                )
            )

        # API returns newest first, we want chronological
        observations.sort(key=lambda o: o.available_at)
        return tuple(observations)
    except Exception as e:
        logger.warning(f"Failed to map historical Fear & Greed data: {e}")
        return ()
