"""Mappers for converting Deribit API responses into Domain Models."""

from __future__ import annotations

from typing import Any

from neon_radar.domain.market_intelligence.enums import (
    IntelligenceSignalType,
    SourceReliability,
)
from neon_radar.domain.market_intelligence.models import IntelligenceSignal

DEFAULT_WEIGHT = 0.8


def map_dvol_to_signal(
    data: dict[str, Any], symbol: str, ingestion_timestamp: int
) -> IntelligenceSignal | None:
    """Map Deribit Volatility Index (DVOL) data to an IntelligenceSignal.

    Expected data format is the result of public/get_volatility_index_data.
    """
    result = data.get("result")
    if not result:
        return None

    history = result.get("data")
    if not history or not isinstance(history, list):
        return None

    # Get the most recent candle (last element in the list)
    latest = history[-1]
    if len(latest) < 5:
        return None

    event_time = latest[0]
    close_val = latest[4]

    try:
        dvol_val = float(close_val)
    except (ValueError, TypeError):
        return None

    # We cannot correctly normalize direction without historical baseline (Z-Score).
    # Therefore, we yield a signal with 0 direction and 0 strength,
    # but include the raw value in metadata for future processing.
    direction = 0.0
    strength = 0.0

    return IntelligenceSignal(
        type=IntelligenceSignalType.DVOL,
        direction=direction,
        strength=strength,
        event_timestamp=int(event_time),
        ingestion_timestamp=ingestion_timestamp,
        source_id="deribit",
        provider_name="Deribit",
        provider_type="API",
        reliability=SourceReliability.ANALYTICS,
        weight=DEFAULT_WEIGHT,
        metadata={
            "symbol": symbol,
            "raw_value": str(dvol_val),
        },
    )


def map_put_call_ratio_to_signal(
    data: dict[str, Any], symbol: str, ingestion_timestamp: int
) -> IntelligenceSignal | None:
    """Map Deribit options book summary to Put/Call Ratio IntelligenceSignal.

    Expected data format is the result of public/get_book_summary_by_currency.
    """
    result = data.get("result")
    if not result or not isinstance(result, list):
        return None

    puts_oi = 0.0
    calls_oi = 0.0

    for item in result:
        instrument_name = item.get("instrument_name", "")
        oi = item.get("open_interest")

        if oi is None:
            continue

        try:
            oi_val = float(oi)
        except (ValueError, TypeError):
            continue

        if instrument_name.endswith("-P"):
            puts_oi += oi_val
        elif instrument_name.endswith("-C"):
            calls_oi += oi_val

    if puts_oi == 0.0 and calls_oi == 0.0:
        return None

    if calls_oi == 0.0:
        # Division by zero scenario, treated as rejected signal due to anomaly
        return None

    ratio = puts_oi / calls_oi

    # We cannot correctly normalize direction without historical baseline.
    # Therefore, we yield a signal with 0 direction and 0 strength,
    # but include the raw ratio in metadata for future processing.
    direction = 0.0
    strength = 0.0

    # Since we are fetching book summary, there is no single event_time for the whole book.
    # We use ingestion_timestamp (current time) as event_time.
    event_time = ingestion_timestamp

    return IntelligenceSignal(
        type=IntelligenceSignalType.PUT_CALL_RATIO,
        direction=direction,
        strength=strength,
        event_timestamp=event_time,
        ingestion_timestamp=ingestion_timestamp,
        source_id="deribit",
        provider_name="Deribit",
        provider_type="API",
        reliability=SourceReliability.ANALYTICS,
        weight=DEFAULT_WEIGHT,
        metadata={
            "symbol": symbol,
            "puts_oi": str(puts_oi),
            "calls_oi": str(calls_oi),
            "put_call_ratio": str(ratio),
        },
    )
