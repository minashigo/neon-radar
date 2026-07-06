"""Application-wide exception hierarchy.

Every error raised inside Neon Radar inherits from :class:`NeonRadarError`.
This gives the presentation layer a single base class to catch for the
"show user-friendly message" path while still allowing fine-grained
handling where needed.

Hierarchy::

    NeonRadarError
    ├── ConfigError
    ├── ApiError
    │   ├── NetworkError
    │   ├── RateLimitError
    │   └── ServerError
    ├── DataError
    │   ├── ParseError
    │   └── DataValidationError
    ├── IndicatorError
    └── ExchangeError
"""

from __future__ import annotations


class NeonRadarError(Exception):
    """Base class for all errors raised by Neon Radar.

    Catching this in the UI guarantees no Neon Radar exception will
    bubble up uncaught.
    """


class ConfigError(NeonRadarError):
    """Configuration file is missing, malformed, or invalid."""


class ApiError(NeonRadarError):
    """Base class for errors coming from the Binance API layer."""


class NetworkError(ApiError):
    """Network-level failure: connection refused, DNS, timeout, etc."""


class RateLimitError(ApiError):
    """Binance rate limit (HTTP 429) was exceeded even after retries."""


class ServerError(ApiError):
    """Binance returned 5xx — the server is having a bad time."""


class DataError(NeonRadarError):
    """The data we received cannot be used as-is."""


class ParseError(DataError):
    """A response from the API could not be parsed into our domain models."""


class DataValidationError(DataError):
    """The parsed data violates domain invariants (e.g. negative volume)."""


class IndicatorError(NeonRadarError):
    """An indicator could not be computed (insufficient data, bad input)."""


class ExchangeError(NeonRadarError):
    """An exchange client does not support a requested operation.

    For example: ``OpenInterest`` is requested from a spot-only exchange.
    This is **not** a transient network failure; the caller should not
    retry.
    """
