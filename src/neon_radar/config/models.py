"""Pydantic models for application configuration.

Design notes
------------
* All models are **frozen** (``model_config = ConfigDict(frozen=True)``) so the
  loaded configuration cannot be accidentally mutated by services.
* ``TimeFrame`` is a string ``Enum`` whose values match Binance API interval
  strings (``"1d"``, ``"4h"`` …). This lets us pass them straight to the API
  without an extra mapping layer.
* Symbol names are upper-cased and whitespace-trimmed on validation so users
  can write ``"btcusdt"`` in the JSON without surprises.
* Defaults are conservative — appropriate for a daily-use analytical tool,
  not a high-frequency trader.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class TimeFrame(StrEnum):
    """Supported kline intervals (subset of Binance Futures intervals).

    Values are exactly the strings Binance's REST API expects, so a
    ``TimeFrame`` instance can be passed to the API without conversion.
    """

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MN = "1M"

    @property
    def seconds(self) -> int:
        """Length of one candle in seconds. Used for caching TTL calculations."""
        mapping: dict[TimeFrame, int] = {
            TimeFrame.M1: 60,
            TimeFrame.M5: 300,
            TimeFrame.M15: 900,
            TimeFrame.M30: 1800,
            TimeFrame.H1: 3600,
            TimeFrame.H2: 7200,
            TimeFrame.H4: 14_400,
            TimeFrame.H6: 21_600,
            TimeFrame.H8: 28_800,
            TimeFrame.H12: 43_200,
            TimeFrame.D1: 86_400,
            TimeFrame.D3: 259_200,
            TimeFrame.W1: 604_800,
            TimeFrame.MN: 2_592_000,
        }
        return mapping[self]

    @property
    def higher_timeframe(self) -> TimeFrame | None:
        """Logical higher timeframe for macro trend analysis."""
        mapping: dict[TimeFrame, TimeFrame | None] = {
            TimeFrame.M1: TimeFrame.M5,
            TimeFrame.M5: TimeFrame.M15,
            TimeFrame.M15: TimeFrame.H1,
            TimeFrame.M30: TimeFrame.H4,
            TimeFrame.H1: TimeFrame.H4,
            TimeFrame.H2: TimeFrame.H8,
            TimeFrame.H4: TimeFrame.D1,
            TimeFrame.H6: TimeFrame.D1,
            TimeFrame.H8: TimeFrame.D1,
            TimeFrame.H12: TimeFrame.D3,
            TimeFrame.D1: TimeFrame.W1,
            TimeFrame.D3: TimeFrame.W1,
            TimeFrame.W1: TimeFrame.MN,
            TimeFrame.MN: None,
        }
        return mapping.get(self)


class SymbolConfig(BaseModel):
    """A single tradable instrument on Binance Futures.

    ``symbol`` is the exchange ticker (e.g. ``"BTCUSDT"``) — not a base/quote
    pair structure. We keep it as a flat string because that is what Binance
    uses internally; splitting would force every downstream component to
    re-join it.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1, max_length=32)
    enabled: bool = True
    note: str | None = Field(default=None, max_length=120)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        """Symbols are case-insensitive on Binance but the API expects upper-case."""
        cleaned = value.strip().upper()
        if not cleaned.isalnum():
            raise ValueError(
                f"Invalid symbol '{value}': must be alphanumeric (e.g. 'BTCUSDT')"
            )
        return cleaned


class RefreshConfig(BaseModel):
    """Background refresh behaviour."""

    model_config = ConfigDict(frozen=True)

    interval_seconds: int = Field(default=60, ge=5, le=86_400)
    auto_refresh: bool = True


class ApiConfig(BaseModel):
    """Network configuration for the Binance Futures public REST API."""

    model_config = ConfigDict(frozen=True)

    base_url: str = "https://fapi.binance.com"
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=1.5, ge=0.1, le=30.0)
    rate_limit_per_minute: int = Field(default=1200, ge=1)


class CacheConfig(BaseModel):
    """On-disk cache for fetched klines.

    The cache is optional. If disabled, every refresh hits the API.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    directory: Path = Path("~/.neon_radar/cache")
    ttl_seconds: int = Field(default=300, ge=0)

    @field_validator("directory")
    @classmethod
    def _expand_user(cls, value: Path) -> Path:
        return value.expanduser().resolve()


class UiConfig(BaseModel):
    """Presentation-layer configuration.

    Kept here (rather than in ``presentation/``) so the UI can be configured
    without modifying UI code. The presentation layer reads this on startup.
    """

    model_config = ConfigDict(frozen=True)

    theme: str = "neon_dark"
    default_timeframe: TimeFrame = TimeFrame.D1
    window_size: tuple[int, int] = Field(default=(1600, 1000))
    show_volume_pane: bool = True
    default_candles: int = Field(default=200, ge=50, le=2000)

    @field_validator("window_size")
    @classmethod
    def _validate_window_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        w, h = value
        if w < 800 or h < 600:
            raise ValueError("Window size too small (min 800x600)")
        return value


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(frozen=True)

    level: str = Field(default="INFO")
    file: Path | None = None
    json_format: bool = False

    @field_validator("level")
    @classmethod
    def _validate_level(cls, value: str) -> str:
        normalized = value.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid:
            raise ValueError(f"Invalid log level '{value}'. Must be one of: {valid}")
        return normalized

    @field_validator("file")
    @classmethod
    def _expand_file(cls, value: Path | None) -> Path | None:
        return value.expanduser().resolve() if value else None


class AppConfig(BaseModel):
    """Root configuration model.

    This is the only object the rest of the application should depend on.
    It is composed of the smaller models above so each subsystem reads only
    the slice it needs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbols: list[SymbolConfig]
    timeframes: list[TimeFrame] = Field(default_factory=lambda: [TimeFrame.D1, TimeFrame.H4])
    refresh: RefreshConfig = Field(default_factory=RefreshConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    ui: UiConfig = Field(default_factory=UiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _validate_unique_symbols(self) -> AppConfig:
        """Symbols must be unique regardless of the case in the JSON file."""
        seen: set[str] = set()
        for s in self.symbols:
            if s.symbol in seen:
                raise ValueError(f"Duplicate symbol in config: {s.symbol}")
            seen.add(s.symbol)
        return self

    @model_validator(mode="after")
    def _validate_default_timeframe(self) -> AppConfig:
        """The default timeframe must be present in ``timeframes``."""
        if self.ui.default_timeframe not in self.timeframes:
            raise ValueError(
                f"ui.default_timeframe={self.ui.default_timeframe.value} "
                f"must be one of {[t.value for t in self.timeframes]}"
            )
        return self

    def enabled_symbols(self) -> list[SymbolConfig]:
        """Convenience accessor for the presentation layer."""
        return [s for s in self.symbols if s.enabled]


# Public constants — useful for type-checking and IDE help.
SUPPORTED_TIMEFRAMES: Final[tuple[TimeFrame, ...]] = tuple(TimeFrame)
MIN_CANDLES: Final[int] = 50
MAX_CANDLES: Final[int] = 2000
