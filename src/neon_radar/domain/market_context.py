"""Domain models for the Market Context layer.

This module defines immutable dataclasses representing Binance Futures
microstructure data. These context objects provide point-in-time correctness
via explicit time markers, isolating the domain from raw API structures.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Generic, TypeVar

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
    long_liquidations_usd: float = 0.0
    short_liquidations_usd: float = 0.0
    total_liquidations_usd: float = 0.0
    time_context: TimeContext = None


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


T_Context = TypeVar("T_Context")

@dataclass(slots=True, frozen=True)
class ContextSeries(Generic[T_Context]):
    """Base class for all historical context series.
    
    Provides iterable API, validation, slicing, latest(), and window().
    """

    symbol: Symbol
    items: list[T_Context]

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, Symbol):
            object.__setattr__(self, "symbol", Symbol(self.symbol))

        if self.items:
            # All context models must have `time_context`
            times = [c.time_context.event_time for c in self.items]
            if times != sorted(times):
                raise ValueError(
                    f"ContextSeries items are not sorted ascending by event_time "
                    f"for {self.symbol}"
                )
            if len(set(times)) != len(times):
                raise ValueError(
                    f"ContextSeries items contain duplicate event_time values "
                    f"for {self.symbol}"
                )

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T_Context]:
        return iter(self.items)

    def __getitem__(self, index: int | slice) -> T_Context | list[T_Context]:
        return self.items[index]

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    def latest(self) -> T_Context | None:
        """Return the most recent item or ``None`` if empty."""
        return self.items[-1] if self.items else None

    def window(self, n: int) -> ContextSeries[T_Context]:
        """Return a new series with only the last ``n`` items."""
        if n <= 0:
            raise ValueError("n must be positive")
        return replace(self, items=self.items[-n:])

    def slice_by_publish_time(self, max_timestamp: int) -> ContextSeries[T_Context]:
        """Return a new series excluding items published after ``max_timestamp``.
        
        This is a critical Point-in-Time safeguard to prevent look-ahead bias.
        """
        if not self.items:
            return self

        # Fast binary search. We extract publish_times and use bisect_right.
        import bisect
        publish_times = [c.time_context.publish_time for c in self.items]
        # We assume publish_times are also monotonically increasing (or roughly so).
        # Wait, bisect requires strict sorting. If publish_times are not strictly sorted,
        # it might not work perfectly, but for Binance they almost always are.
        # Let's ensure publish_times are sorted, otherwise we fallback to linear.
        if publish_times != sorted(publish_times):
            # Fallback to linear filtering if publish times are slightly out of order
            valid_items = tuple(c for c in self.items if c.time_context.publish_time <= max_timestamp)
        else:
            idx = bisect.bisect_right(publish_times, max_timestamp)
            valid_items = self.items[:idx]

        return replace(self, items=valid_items)


@dataclass(slots=True, frozen=True)
class FundingSeries(ContextSeries[FundingContext]):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FundingSeries:
        items = [FundingContext(
            raw_funding=i["raw_funding"],
            funding_8h_equiv=i["funding_8h_equiv"],
            annualized_apr=i["annualized_apr"],
            mark_price=i["mark_price"],
            next_funding_time_utc=i["next_funding_time_utc"],
            time_context=TimeContext(**i["time_context"])
        ) for i in data["items"]]
        return cls(symbol=Symbol(data["symbol"]), items=tuple(items))

@dataclass(slots=True, frozen=True)
class OpenInterestSeries(ContextSeries[OpenInterestContext]):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenInterestSeries:
        items = [OpenInterestContext(
            oi_coin=i["oi_coin"],
            oi_usd_notional=i["oi_usd_notional"],
            time_context=TimeContext(**i["time_context"])
        ) for i in data["items"]]
        return cls(symbol=Symbol(data["symbol"]), items=tuple(items))

@dataclass(slots=True, frozen=True)
class LongShortSeries(ContextSeries[LongShortRatioContext]):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LongShortSeries:
        items = [LongShortRatioContext(
            long_pct=i["long_pct"],
            short_pct=i["short_pct"],
            ls_ratio=i["ls_ratio"],
            time_context=TimeContext(**i["time_context"])
        ) for i in data["items"]]
        return cls(symbol=Symbol(data["symbol"]), items=tuple(items))

@dataclass(slots=True, frozen=True)
class TakerFlowSeries(ContextSeries[TakerFlowContext]):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TakerFlowSeries:
        items = [TakerFlowContext(
            buy_volume=i["buy_volume"],
            sell_volume=i["sell_volume"],
            net_buy_volume=i["net_buy_volume"],
            time_context=TimeContext(**i["time_context"])
        ) for i in data["items"]]
        return cls(symbol=Symbol(data["symbol"]), items=tuple(items))

@dataclass(slots=True, frozen=True)
class LiquidationSeries(ContextSeries[LiquidationContext]):
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiquidationSeries:
        items = [LiquidationContext(
            long_liquidations=i.get("long_liquidations", 0.0),
            short_liquidations=i.get("short_liquidations", 0.0),
            long_liquidations_usd=i.get("long_liquidations_usd", 0.0),
            short_liquidations_usd=i.get("short_liquidations_usd", 0.0),
            total_liquidations_usd=i.get("total_liquidations_usd", 0.0),
            time_context=TimeContext(**i["time_context"])
        ) for i in data["items"]]
        return cls(symbol=Symbol(data["symbol"]), items=tuple(items))


@dataclass(slots=True, frozen=True)
class HistoricalMarketContext:
    """Aggregate root containing all historical series for a symbol at a specific time.

    Attributes
    ----------
    symbol
        The trading pair (e.g. BTCUSDT).
    schema_version
        Version of the schema for backward compatibility.
    timestamp
        Evaluation timestamp (Unix ms). The time this historical context is built for.
    """

    symbol: Symbol
    timestamp: int
    schema_version: SchemaVersion = SchemaVersion.V1

    funding_history: FundingSeries | None = None
    open_interest_history: OpenInterestSeries | None = None
    ls_ratio_history: LongShortSeries | None = None
    taker_flow_history: TakerFlowSeries | None = None
    liquidations_history: LiquidationSeries | None = None

    def __post_init__(self) -> None:
        """Enforce strict Point-in-Time correctness for all series.
        
        This prevents any downstream rule from accidentally accessing data 
        published after this context's timestamp, completely eliminating look-ahead bias.
        """
        if self.funding_history:
            object.__setattr__(self, "funding_history", self.funding_history.slice_by_publish_time(self.timestamp))
        if self.open_interest_history:
            object.__setattr__(self, "open_interest_history", self.open_interest_history.slice_by_publish_time(self.timestamp))
        if self.ls_ratio_history:
            object.__setattr__(self, "ls_ratio_history", self.ls_ratio_history.slice_by_publish_time(self.timestamp))
        if self.taker_flow_history:
            object.__setattr__(self, "taker_flow_history", self.taker_flow_history.slice_by_publish_time(self.timestamp))
        if self.liquidations_history:
            object.__setattr__(self, "liquidations_history", self.liquidations_history.slice_by_publish_time(self.timestamp))

    def slice_at(self, timestamp: int) -> HistoricalMarketContext:
        """Return a new HistoricalMarketContext sliced exactly at the given timestamp.
        
        This enables efficient point-in-time evaluation during backtesting by 
        fetching the full history once and slicing it in memory for each candle.
        """
        from dataclasses import replace
        
        # replace() will invoke __post_init__ on the new instance, 
        # which will safely re-slice all underlying series using slice_by_publish_time
        # up to the new timestamp.
        return replace(self, timestamp=timestamp)
