"""Shared pytest fixtures and helper functions.

Test files can ``from .conftest import make_candles, make_series`` or
just rely on the implicit ``conftest`` import by pytest.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from neon_radar.config.loader import ConfigLoader

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neon_radar.config.models import AppConfig, TimeFrame
    from neon_radar.domain.models import KlineSeries


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def example_config_path() -> Path:
    """Path to the checked-in ``config.example.json``."""
    return Path(__file__).resolve().parent.parent / "config.example.json"


@pytest.fixture
def loaded_config(example_config_path: Path) -> AppConfig:
    """A validated :class:`AppConfig` from the example file."""
    return ConfigLoader(example_config_path).load()


# ---------------------------------------------------------------------------
# Indicator-test data helpers
# ---------------------------------------------------------------------------


def make_candles(
    closes: Sequence[float],
    *,
    start_time: int = 1_700_000_000_000,
    interval_ms: int = 86_400_000,
    base_volume: float = 1_000.0,
) -> tuple:
    """Build a tuple of OHLCV candles from a list of close prices.

    Each candle has:
      * ``open_time`` = ``start_time + i * interval_ms``
      * ``open``      = previous candle's close (or ``closes[0]``)
      * ``high``      = ``close + 1``
      * ``low``       = ``close - 1``
      * ``close``     = ``closes[i]``
      * ``volume``    = ``base_volume + i``

    The result is suitable for indicator tests where predictable
    shape matters more than realistic OHLC relationships.
    """
    from neon_radar.domain.models import OHLCV

    candles: list[OHLCV] = []
    prev = closes[0]
    for i, close in enumerate(closes):
        candles.append(
            OHLCV(
                open_time=start_time + i * interval_ms,
                open=prev,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=base_volume + i,
            )
        )
        prev = close
    return tuple(candles)


def make_series(
    closes: Sequence[float],
    *,
    symbol: str = "BTCUSDT",
    timeframe: TimeFrame | None = None,
) -> KlineSeries:
    """Build a KlineSeries from close prices."""
    from neon_radar.config.models import TimeFrame
    from neon_radar.domain.models import KlineSeries, Symbol

    if timeframe is None:
        timeframe = TimeFrame.D1
    return KlineSeries(
        symbol=Symbol(symbol),
        timeframe=timeframe,
        candles=make_candles(closes),
    )
