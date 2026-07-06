"""Tests for the ExchangeClient abstract base class."""

from __future__ import annotations

import pytest

from neon_radar.config.models import TimeFrame
from neon_radar.domain.exceptions import ExchangeError
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol, TickerStats
from neon_radar.infrastructure.exchanges.base import ExchangeClient, ExchangeInfo


class _FakeClient(ExchangeClient):
    """Minimal concrete implementation for testing the ABC contract."""

    name = "fake"

    def __init__(self) -> None:
        self.closed = False

    async def info(self) -> ExchangeInfo:
        return ExchangeInfo(
            name="fake",
            display_name="Fake Exchange",
            website="https://example.com",
            supports_funding=False,
            supports_open_interest=False,
        )

    async def get_klines(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        *,
        limit: int = 500,
        end_time: int | None = None,
    ) -> KlineSeries:
        candles = tuple(
            OHLCV(
                open_time=1_700_000_000_000 + i,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0,
            )
            for i in range(min(limit, 3))
        )
        return KlineSeries(symbol=symbol, timeframe=timeframe, candles=candles)

    async def get_ticker(self, symbol: Symbol) -> TickerStats:
        return TickerStats(
            symbol=symbol,
            last_price=100.0,
            price_change_percent=0.0,
            high_24h=101.0,
            low_24h=99.0,
            volume_24h=1000.0,
            quote_volume_24h=100_000.0,
        )

    async def close(self) -> None:
        self.closed = True


class TestExchangeClient:
    @pytest.mark.asyncio
    async def test_basic_methods(self) -> None:
        client = _FakeClient()
        info = await client.info()
        assert info.name == "fake"

        series = await client.get_klines(Symbol("BTCUSDT"), TimeFrame.H4, limit=3)
        assert len(series) == 3

        ticker = await client.get_ticker(Symbol("BTCUSDT"))
        assert ticker.symbol == "BTCUSDT"

        await client.close()
        assert client.closed

    @pytest.mark.asyncio
    async def test_funding_defaults_to_raising(self) -> None:
        client = _FakeClient()
        with pytest.raises(ExchangeError, match="does not support"):
            await client.get_funding_rate(Symbol("BTCUSDT"))

    @pytest.mark.asyncio
    async def test_open_interest_defaults_to_raising(self) -> None:
        client = _FakeClient()
        with pytest.raises(ExchangeError, match="does not support"):
            await client.get_open_interest(Symbol("BTCUSDT"))

    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            ExchangeClient()  # type: ignore[abstract]


class TestExchangeInfo:
    def test_basic(self) -> None:
        info = ExchangeInfo(
            name="binance",
            display_name="Binance",
            website="https://binance.com",
            supports_funding=True,
            supports_open_interest=True,
        )
        assert info.name == "binance"
        assert info.supports_funding

    def test_immutable(self) -> None:
        info = ExchangeInfo(
            name="x",
            display_name="X",
            website="https://x.com",
            supports_funding=False,
            supports_open_interest=False,
        )
        with pytest.raises((AttributeError, Exception)):
            info.name = "y"  # type: ignore[misc]
