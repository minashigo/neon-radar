"""Mappers for converting CoinGlass API responses into Domain Models."""

from __future__ import annotations

from typing import Any

from neon_radar.domain.market_intelligence.enums import (
    IntelligenceSignalType,
    SourceReliability,
)
from neon_radar.domain.market_intelligence.models import IntelligenceSignal

DEFAULT_WEIGHT = 0.8


def _get_history(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    history = data.get("data")
    if not history or not isinstance(history, list):
        return None
    return history


def _sort_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        return sorted(history, key=lambda x: x.get("time", 0))
    except (ValueError, TypeError):
        return []


def _get_latest_item(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    sorted_history = _sort_history(history)
    return sorted_history[-1] if sorted_history else None


def map_long_short_ratio_to_signal(
    data: dict[str, Any], symbol: str, ingestion_timestamp: int
) -> IntelligenceSignal | None:
    """Map CoinGlass long/short ratio data to an IntelligenceSignal."""
    history = _get_history(data)
    if not history:
        return None

    latest = _get_latest_item(history)
    if not latest:
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

    net_long_ratio = (long_pct - short_pct) / 100.0
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
        weight=DEFAULT_WEIGHT,
        metadata={
            "symbol": symbol,
            "long_percent": str(long_pct),
            "short_percent": str(short_pct),
        },
    )


def map_funding_rate_to_signal(
    data: dict[str, Any], symbol: str, ingestion_timestamp: int
) -> IntelligenceSignal | None:
    """Map CoinGlass funding rate data to an IntelligenceSignal."""
    history = _get_history(data)
    if not history:
        return None

    latest = _get_latest_item(history)
    if not latest:
        return None

    event_time = latest.get("time")
    funding_rate_str = latest.get("close")

    if event_time is None or funding_rate_str is None:
        return None

    try:
        funding_rate = float(funding_rate_str)
    except (ValueError, TypeError):
        return None

    direction = 1.0 if funding_rate > 0 else -1.0 if funding_rate < 0 else 0.0

    # CoinGlass (like Binance) returns funding rate as a decimal (e.g., 0.0001 for 0.01%).
    # A multiplier of 1000.0 maps a 0.1% funding rate (extremely overheated) to 1.0 strength.
    strength = max(0.0, min(1.0, abs(funding_rate) * 1000.0))

    return IntelligenceSignal(
        type=IntelligenceSignalType.FUNDING,
        direction=direction,
        strength=strength,
        event_timestamp=int(event_time),
        ingestion_timestamp=ingestion_timestamp,
        source_id="coinglass",
        provider_name="CoinGlass",
        provider_type="API",
        reliability=SourceReliability.ANALYTICS,
        weight=DEFAULT_WEIGHT,
        metadata={
            "symbol": symbol,
            "funding_rate": str(funding_rate),
        },
    )


def map_open_interest_to_signal(
    data: dict[str, Any], symbol: str, ingestion_timestamp: int
) -> IntelligenceSignal | None:
    """Map CoinGlass open interest data to an IntelligenceSignal."""
    history = _get_history(data)
    if not history or len(history) < 2:
        return None

    sorted_history = _sort_history(history)
    if len(sorted_history) < 2:
        return None

    latest = sorted_history[-1]
    previous = sorted_history[-2]

    event_time = latest.get("time")
    latest_oi_str = latest.get("close")
    prev_oi_str = previous.get("close")

    if event_time is None or latest_oi_str is None or prev_oi_str is None:
        return None

    try:
        latest_oi = float(latest_oi_str)
        prev_oi = float(prev_oi_str)
    except (ValueError, TypeError):
        return None

    if prev_oi == 0:
        return None

    delta_pct = (latest_oi - prev_oi) / prev_oi
    direction = 1.0 if delta_pct > 0 else -1.0 if delta_pct < 0 else 0.0
    strength = max(0.0, min(1.0, abs(delta_pct) * 20.0))

    return IntelligenceSignal(
        type=IntelligenceSignalType.OPEN_INTEREST,
        direction=direction,
        strength=strength,
        event_timestamp=int(event_time),
        ingestion_timestamp=ingestion_timestamp,
        source_id="coinglass",
        provider_name="CoinGlass",
        provider_type="API",
        reliability=SourceReliability.ANALYTICS,
        weight=DEFAULT_WEIGHT,
        metadata={
            "symbol": symbol,
            "latest_oi": str(latest_oi),
            "previous_oi": str(prev_oi),
            "delta_pct": str(delta_pct),
        },
    )


def map_liquidations_to_signal(
    data: dict[str, Any], symbol: str, ingestion_timestamp: int
) -> IntelligenceSignal | None:
    """Map CoinGlass liquidations data to an IntelligenceSignal."""
    history = _get_history(data)
    if not history:
        return None

    latest = _get_latest_item(history)
    if not latest:
        return None

    event_time = latest.get("time")
    long_liq_str = latest.get("long_liquidation_usd")
    short_liq_str = latest.get("short_liquidation_usd")

    if event_time is None or long_liq_str is None or short_liq_str is None:
        return None

    try:
        long_liq = float(long_liq_str)
        short_liq = float(short_liq_str)
    except (ValueError, TypeError):
        return None

    if long_liq < 0 or short_liq < 0:
        return None

    total_liq = long_liq + short_liq

    if total_liq <= 0:
        return None

    direction = (short_liq - long_liq) / total_liq
    strength = abs(direction)

    return IntelligenceSignal(
        type=IntelligenceSignalType.LIQUIDATIONS,
        direction=direction,
        strength=strength,
        event_timestamp=int(event_time),
        ingestion_timestamp=ingestion_timestamp,
        source_id="coinglass",
        provider_name="CoinGlass",
        provider_type="API",
        reliability=SourceReliability.ANALYTICS,
        weight=DEFAULT_WEIGHT,
        metadata={
            "symbol": symbol,
            "long_liquidation_usd": str(long_liq),
            "short_liquidation_usd": str(short_liq),
            "total_liquidation_usd": str(total_liq),
        },
    )
