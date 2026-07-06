"""Domain models — immutable data shapes used throughout the application.

Design notes
------------
* All models use ``@dataclass(slots=True, frozen=True)`` — they are
  cheap to allocate, hashable (so they can live in sets/dicts), and
  immutable (so they cannot be accidentally mutated by downstream code).
* We intentionally use ``int`` (Unix milliseconds) for timestamps instead
  of ``datetime``. Datetimes are timezone-aware but slow to compare;
  integers are cheap to compare, easy to send over signals, and the
  ``datetime`` view is available via the ``OHLCV.datetime`` property.
* ``Symbol`` is a thin ``str`` subclass for type-safety: you cannot
  accidentally pass a free-form string where a symbol is expected. The
  validation lives in ``__new__`` so it is enforced at construction time.
* ``KlineSeries`` is a thin wrapper around a tuple of ``OHLCV`` plus a
  ``Symbol`` and a ``TimeFrame``. We do **not** use ``pandas.DataFrame``
  here because:
    - The domain should not depend on pandas.
    - pandas DataFrames are mutable and unhashable, breaking our
      immutability invariants.
    - Conversion to pandas happens at the application/presentation
      boundary, where the heavy machinery is justified.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from neon_radar.config.models import TimeFrame


class Symbol(str):
    """Binance Futures trading symbol, e.g. ``"BTCUSDT"``.

    A ``Symbol`` is a ``str`` subclass that validates its value on
    construction. It can be used everywhere a ``str`` is expected —
    for example as a dict key — but type-checkers will refuse a
    random ``str`` where ``Symbol`` is required.
    """

    __slots__ = ()

    def __new__(cls, value: object) -> Symbol:
        s = str(value).strip().upper()
        if not s or not s.isalnum() or len(s) > 32:
            raise ValueError(f"Invalid symbol: {value!r}")
        return super().__new__(cls, s)

    def base(self) -> str:
        """Return the base asset, assuming ``BASEQUOTE`` naming.

        For ``BTCUSDT`` → ``BTC``. For exotic symbols that do not follow
        the pattern, returns the original string.
        """
        quote_candidates = ("USDT", "BUSD", "USDC", "USD", "BTC", "ETH")
        for quote in quote_candidates:
            if self.endswith(quote) and len(self) > len(quote):
                return self[: -len(quote)]
        return str(self)

    def quote(self) -> str:
        """Return the quote asset, assuming ``BASEQUOTE`` naming."""
        quote_candidates = ("USDT", "BUSD", "USDC", "USD", "BTC", "ETH")
        for quote in quote_candidates:
            if self.endswith(quote) and len(self) > len(quote):
                return quote
        return ""


@dataclass(slots=True, frozen=True)
class OHLCV:
    """A single OHLCV candle.

    Attributes
    ----------
    open_time
        Candle open time, in Unix milliseconds (UTC).
    open, high, low, close
        Prices in quote currency (e.g. USDT).
    volume
        Base asset volume traded during the candle.
    close_time
        Candle close time, in Unix milliseconds (UTC). Optional —
        Binance provides it; we keep it for completeness.
    quote_volume
        Quote asset volume (optional).
    trades
        Number of trades during the candle (optional).
    """

    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int | None = None
    quote_volume: float | None = None
    trades: int | None = None

    def __post_init__(self) -> None:
        # Field-level sanity checks. We do not raise for the optional
        # fields if they are None.
        for name in ("open", "high", "low", "close", "volume"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"OHLCV.{name} must be non-negative, got {value}")
        if self.high < self.low:
            raise ValueError(
                f"OHLCV.high ({self.high}) must be >= low ({self.low})"
            )
        if self.close_time is not None and self.close_time < self.open_time:
            raise ValueError("OHLCV.close_time must be >= open_time")

    @property
    def datetime(self) -> datetime:
        """Open time as a timezone-aware UTC ``datetime``."""
        return datetime.fromtimestamp(self.open_time / 1000, tz=UTC)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def range(self) -> float:
        """High minus low."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """The absolute size of the candle body: abs(close - open)."""
        return abs(self.close - self.open)


@dataclass(slots=True, frozen=True)
class KlineSeries:
    """An ordered series of ``OHLCV`` candles for a symbol/timeframe pair.

    The series is immutable. To build a derived series, use ``with_candles``
    which returns a new instance — this keeps the immutability contract.
    """

    symbol: Symbol
    timeframe: TimeFrame
    candles: tuple[OHLCV, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, Symbol):
            # Be lenient on input: allow plain strings.
            object.__setattr__(self, "symbol", Symbol(self.symbol))
        # Enforce ascending time order — Binance guarantees it but we
        # double-check because downstream indicators assume it.
        times = [c.open_time for c in self.candles]
        if times != sorted(times):
            raise ValueError(
                f"KlineSeries candles are not sorted ascending by open_time "
                f"for {self.symbol}@{self.timeframe.value}"
            )

    def __len__(self) -> int:
        return len(self.candles)

    def __iter__(self) -> Iterator[OHLCV]:
        return iter(self.candles)

    def __getitem__(self, index: int | slice) -> OHLCV | list[OHLCV]:
        return self.candles[index]

    @property
    def is_empty(self) -> bool:
        return len(self.candles) == 0

    def latest(self) -> OHLCV | None:
        """Return the most recent candle or ``None`` if empty."""
        return self.candles[-1] if self.candles else None

    def last_n(self, n: int) -> KlineSeries:
        """Return a new series with only the last ``n`` candles."""
        if n <= 0:
            raise ValueError("n must be positive")
        return replace(self, candles=self.candles[-n:])


@dataclass(slots=True, frozen=True)
class TickerStats:
    """24h ticker statistics for a symbol.

    Returned by ``GET /fapi/v1/ticker/24hr`` on Binance Futures. The fields
    are optional because Binance may omit some on less-traded pairs.
    """

    symbol: Symbol
    last_price: float
    price_change_percent: float
    high_24h: float
    low_24h: float
    volume_24h: float
    quote_volume_24h: float
    open_interest: float | None = None
    timestamp: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, Symbol):
            object.__setattr__(self, "symbol", Symbol(self.symbol))

    @property
    def is_bullish_24h(self) -> bool:
        return self.price_change_percent > 0
