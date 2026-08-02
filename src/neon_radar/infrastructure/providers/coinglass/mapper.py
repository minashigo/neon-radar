"""Mappers for converting CoinGlass API responses into Domain Models."""

from __future__ import annotations

from typing import Any

from neon_radar.domain.market_intelligence.enums import (
    IntelligenceSignalType,
    SourceReliability,
)
from neon_radar.domain.market_intelligence.models import IntelligenceSignal


def map_long_short_ratio_to_signal(
    data: dict[str, Any], symbol: str, ingestion_timestamp: int
) -> IntelligenceSignal | None:
    """Map CoinGlass long/short ratio data to an IntelligenceSignal.

    Args:
        data: The JSON payload from the API containing the 'data' array.
        symbol: The symbol this data corresponds to.
        ingestion_timestamp: The time the data was fetched.

    Returns:
        An IntelligenceSignal if valid data is found, otherwise None.
    """
    history = data.get("data")
    if not history or not isinstance(history, list):
        return None

    # Find the latest entry
    try:
        latest = max(history, key=lambda x: x.get("time", 0))
    except (ValueError, TypeError):
        return None

    event_time = latest.get("time")
    long_percent = latest.get("global_account_long_percent")
    short_percent = latest.get("global_account_short_percent")

    if event_time is None or long_percent is None or short_percent is None:
        return None

    try:
        long_pct = float(long_percent)
        short_pct = float(short_percent)
    except (ValueError, TypeError):
        return None

    # Calculate direction and strength
    net_long_ratio = (long_pct - short_pct) / 100.0

    # Cap between -1.0 and 1.0
    direction = max(-1.0, min(1.0, net_long_ratio))
    strength = abs(direction)

    return IntelligenceSignal(
        type=IntelligenceSignalType.LONG_SHORT_RATIO,
        direction=direction,
        strength=strength,
        event_timestamp=int(event_time),
        ingestion_timestamp=ingestion_timestamp,
        source_id="coinglass",
        provider_name="CoinGlass",
        provider_type="API",
        reliability=SourceReliability.ANALYTICS,
        weight=0.8,
        metadata={
            "symbol": symbol,
            "long_percent": str(long_pct),
            "short_percent": str(short_pct),
        },
    )
