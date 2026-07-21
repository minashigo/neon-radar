"""Domain models for the Market Context layer.

This module defines immutable dataclasses representing Binance Futures
microstructure data. These context objects provide point-in-time correctness
via explicit time markers, isolating the domain from raw API structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.models import Symbol


class SchemaVersion(str, Enum):
    """Schema versioning to ensure backward compatibility in backtests."""

    V1 = "FeatureSchema_v1"


@dataclass(slots=True, frozen=True)
class TimeContext:
    """Provides point-in-time correctness guarantees for context data.

    Attributes
    ----------
    event_time
        Unix ms (UTC) when the event actually occurred on the exchange.
    publish_time
        Unix ms (UTC) when the exchange made this data publicly accessible.
    ingest_time
        Unix ms (UTC) when Neon Radar received and normalized the data.
    """

    event_time: int
    publish_time: int
    ingest_time: int


@dataclass(slots=True, frozen=True)
class FundingContext:
    """Current funding rate context for perpetual swaps."""

    raw_funding: float
    funding_8h_equiv: float
    annualized_apr: float
    mark_price: float
    next_funding_time_utc: int
    time_context: TimeContext


@dataclass(slots=True, frozen=True)
class OpenInterestContext:
    """Open interest context measured in both base and quote assets."""

    oi_coin: float
    oi_usd_notional: float
    time_context: TimeContext


@dataclass(slots=True, frozen=True)
class LongShortRatioContext:
    """Ratio of long vs short positions."""

    long_pct: float
    short_pct: float
    ls_ratio: float
    time_context: TimeContext


@dataclass(slots=True, frozen=True)
class TakerFlowContext:
    """Aggressive market buy/sell volume."""

    buy_volume: float
    sell_volume: float
    net_buy_volume: float
    time_context: TimeContext


@dataclass(slots=True, frozen=True)
class LiquidationContext:
    """Liquidated volumes in the base asset."""

    long_liquidations: float
    short_liquidations: float
    time_context: TimeContext


@dataclass(slots=True, frozen=True)
class MarketContext:
    """Aggregate root containing all context models for a symbol at a specific time.

    Attributes
    ----------
    symbol
        The trading pair (e.g. BTCUSDT).
    schema_version
        Version of the schema for backward compatibility.
    timestamp
        Evaluation timestamp (Unix ms). The time this context is built for.
    """

    symbol: Symbol
    timestamp: int
    schema_version: SchemaVersion = SchemaVersion.V1

    funding: FundingContext | None = None
    open_interest: OpenInterestContext | None = None
    ls_ratio: LongShortRatioContext | None = None
    taker_flow: TakerFlowContext | None = None
    liquidations: LiquidationContext | None = None
