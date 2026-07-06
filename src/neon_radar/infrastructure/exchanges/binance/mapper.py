"""Binance JSON → domain model mappers.

These functions are deliberately **pure** (no I/O, no logging, no
clock). They take a raw JSON value (dict or list) from Binance and
return a domain object. Keeping them pure means:

* They are trivially unit-testable with hand-crafted fixtures.
* They can be reused outside the client (e.g. in a backfill script).
* Failure modes are explicit: each mapper raises a specific exception
  with the field name when input is malformed.

Design notes
------------
* Binance klines come as **lists of lists**, not dicts. The column
  order is documented in their API but we map by index — adding a
  field would require updating both sides.
* All numeric values arrive as **strings** in Binance responses. We
  parse them with ``float()`` and never trust the wire type. A
  ``ValueError`` from ``float()`` becomes :class:`ParseError` here.
* ``timestamp`` fields are in **milliseconds** — we keep them as
  ``int`` (no conversion to datetime — see :class:`OHLCV`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neon_radar.domain.exceptions import DataValidationError, ParseError
from neon_radar.domain.funding import FundingRate, OpenInterest
from neon_radar.domain.models import (
    OHLCV,
    KlineSeries,
    Symbol,
    TickerStats,
)

if TYPE_CHECKING:
    from neon_radar.config.models import TimeFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_float(value: Any, *, path: str) -> float:
    """Parse a Binance numeric (which arrives as string) into float.

    Raises :class:`ParseError` (not raw ``ValueError``) so the caller
    can catch a single exception type for "bad Binance response".
    """
    if value is None:
        raise ParseError(f"Missing required numeric field: {path}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"Invalid numeric value for {path}: {value!r}") from exc


def _require_int(value: Any, *, path: str) -> int:
    if value is None:
        raise ParseError(f"Missing required integer field: {path}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"Invalid integer value for {path}: {value!r}") from exc


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Klines
# ---------------------------------------------------------------------------


# Column indices for the array-of-arrays kline format.
# See: https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
_KLINE_OPEN_TIME = 0
_KLINE_OPEN = 1
_KLINE_HIGH = 2
_KLINE_LOW = 3
_KLINE_CLOSE = 4
_KLINE_VOLUME = 5
_KLINE_CLOSE_TIME = 6
_KLINE_QUOTE_VOLUME = 7
_KLINE_TRADES = 8
# 9, 10, 11 are taker buy volumes and an "ignore" field — we skip them.


def map_kline(raw: list[Any]) -> OHLCV:
    """Convert one Binance kline row to :class:`OHLCV`."""
    if not isinstance(raw, list):
        raise ParseError(f"Kline row must be a list, got {type(raw).__name__}")
    if len(raw) < 6:
        raise ParseError(f"Kline row has {len(raw)} fields, expected at least 6")

    try:
        ohlcv = OHLCV(
            open_time=_require_int(raw[_KLINE_OPEN_TIME], path="kline.openTime"),
            open=_require_float(raw[_KLINE_OPEN], path="kline.open"),
            high=_require_float(raw[_KLINE_HIGH], path="kline.high"),
            low=_require_float(raw[_KLINE_LOW], path="kline.low"),
            close=_require_float(raw[_KLINE_CLOSE], path="kline.close"),
            volume=_require_float(raw[_KLINE_VOLUME], path="kline.volume"),
            close_time=_optional_int(raw[_KLINE_CLOSE_TIME]) if len(raw) > _KLINE_CLOSE_TIME else None,
            quote_volume=(
                _optional_float(raw[_KLINE_QUOTE_VOLUME])
                if len(raw) > _KLINE_QUOTE_VOLUME
                else None
            ),
            trades=_optional_int(raw[_KLINE_TRADES]) if len(raw) > _KLINE_TRADES else None,
        )
    except DataValidationError:
        # Re-raise validation errors directly — they already carry useful info.
        raise
    except ValueError as exc:
        raise ParseError(f"Malformed kline: {exc}") from exc

    return ohlcv


def map_klines(
    raw: list[Any],
    *,
    symbol: Symbol,
    timeframe: TimeFrame,
) -> KlineSeries:
    """Convert a list of Binance kline rows to a :class:`KlineSeries`.

    Empty input is **not** an error — it returns an empty series.
    """
    if not isinstance(raw, list):
        raise ParseError(f"Klines response must be a list, got {type(raw).__name__}")

    candles = tuple(map_kline(row) for row in raw)
    return KlineSeries(symbol=symbol, timeframe=timeframe, candles=candles)


# ---------------------------------------------------------------------------
# Ticker 24h
# ---------------------------------------------------------------------------


def map_ticker(raw: dict[str, Any]) -> TickerStats:
    """Convert a 24h ticker response to :class:`TickerStats`."""
    if not isinstance(raw, dict):
        raise ParseError(f"Ticker response must be a dict, got {type(raw).__name__}")
    try:
        return TickerStats(
            symbol=Symbol(raw["symbol"]),
            last_price=_require_float(raw.get("lastPrice"), path="ticker.lastPrice"),
            price_change_percent=_require_float(
                raw.get("priceChangePercent"), path="ticker.priceChangePercent"
            ),
            high_24h=_require_float(raw.get("highPrice"), path="ticker.highPrice"),
            low_24h=_require_float(raw.get("lowPrice"), path="ticker.lowPrice"),
            volume_24h=_require_float(raw.get("volume"), path="ticker.volume"),
            quote_volume_24h=_require_float(raw.get("quoteVolume"), path="ticker.quoteVolume"),
            open_interest=_optional_float(raw.get("sumOpenInterest")),
            timestamp=_optional_int(raw.get("time")),
        )
    except DataValidationError:
        raise
    except KeyError as exc:
        raise ParseError(f"Ticker response missing field: {exc}") from exc


# ---------------------------------------------------------------------------
# Funding rate (via /fapi/v1/premiumIndex)
# ---------------------------------------------------------------------------


def map_funding_rate_from_premium_index(
    raw: dict[str, Any],
    *,
    symbol: Symbol,
) -> FundingRate:
    """Convert a ``/fapi/v1/premiumIndex`` response to :class:`FundingRate`.

    This endpoint gives the **last settled** funding rate, which is
    the most recent known value at query time.
    """
    if not isinstance(raw, dict):
        raise ParseError(
            f"PremiumIndex response must be a dict, got {type(raw).__name__}"
        )
    return FundingRate(
        symbol=symbol,
        rate=_require_float(raw.get("lastFundingRate"), path="premiumIndex.lastFundingRate"),
        mark_price=_optional_float(raw.get("markPrice")),
        next_funding_time=_optional_int(raw.get("nextFundingTime")),
        timestamp=_optional_int(raw.get("time")),
    )


# ---------------------------------------------------------------------------
# Open interest
# ---------------------------------------------------------------------------


def map_open_interest(raw: dict[str, Any], *, symbol: Symbol) -> OpenInterest:
    """Convert an open-interest response to :class:`OpenInterest`."""
    if not isinstance(raw, dict):
        raise ParseError(
            f"OpenInterest response must be a dict, got {type(raw).__name__}"
        )
    return OpenInterest(
        symbol=symbol,
        value=_require_float(raw.get("sumOpenInterest"), path="openInterest.sumOpenInterest"),
        value_quote=_optional_float(raw.get("sumOpenInterestValue")),
        timestamp=_optional_int(raw.get("time")),
    )
