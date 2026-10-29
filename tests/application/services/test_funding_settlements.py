"""Variable funding calendars, event prices, complete downloads and paper integration."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_forward_paper_trading import make_candle

from neon_radar.application.services.execution import PaperExecutionEngine
from neon_radar.application.services.portfolio_engine import PortfolioEngine
from neon_radar.domain.enums import Bias
from neon_radar.domain.execution_costs import SettlementFundingModel
from neon_radar.domain.funding import FundingSettlement
from neon_radar.domain.market_context import FundingContext, FundingSeries, TimeContext
from neon_radar.domain.models import Symbol
from neon_radar.domain.portfolio import OpenPosition
from neon_radar.infrastructure.providers.binance_funding import BinanceHistoricalFundingProvider

SYM = Symbol("BTCUSDT")
HOUR = 3_600_000
DAY = date(1970, 1, 1)


def row(time, rate="0.001", mark="100"):
    return {"fundingTime": time, "fundingRate": rate, "markPrice": mark}


async def history(rows):
    client = MagicMock()
    client._get_json = AsyncMock(return_value=rows)
    provider = BinanceHistoricalFundingProvider(client=client)
    await provider.prefetch((SYM,), DAY, DAY)
    return provider


@pytest.mark.parametrize("direction,expected", [(Bias.BULLISH, -1.0), (Bias.BEARISH, 1.0)])
async def test_actual_variable_times_marks_and_signs(direction, expected):
    provider = await history(
        [
            row(0, "0.99", "10000"),  # excluded: entry and settlement coincide
            row(8 * HOUR + 1, "0.001", "100"),
            row(12 * HOUR, "-0.002", "200"),
            row(13 * HOUR, "0.001", "50"),
            row(14 * HOUR, "0.5", "10000"),  # after exit; must not leak
        ]
    )
    model = SettlementFundingModel()
    # 4 * (100*.001 - 200*.002 + 50*.001) = -1.0.
    for exit_notional in (10, 10_000_000):
        assert model.calculate_funding_cost(
            exit_notional, SYM, direction, 0, 13 * HOUR, provider, quantity=4
        ) == pytest.approx(expected)
    assert provider.get_funding_rate_at(SYM, 8 * HOUR).timestamp == 0
    assert [e.timestamp for e in provider.get_funding_events(SYM, 0, 13 * HOUR)] == [
        8 * HOUR + 1,
        12 * HOUR,
        13 * HOUR,
    ]


async def test_jitter_is_not_rounded_into_an_earlier_exit():
    provider = await history([row(8 * HOUR + 1)])
    model = SettlementFundingModel()
    assert (
        model.calculate_funding_cost(100, SYM, Bias.BULLISH, 0, 8 * HOUR, provider, quantity=1) == 0
    )
    assert (
        model.calculate_funding_cost(100, SYM, Bias.BULLISH, 0, 8 * HOUR + 1, provider, quantity=1)
        == 0.1
    )


@pytest.mark.parametrize("entry,exit", [(0, 24 * HOUR), (-2, HOUR)])
async def test_outside_prefetched_coverage_is_not_zero_cost(entry, exit):
    provider = await history([])
    with pytest.raises(ValueError, match="does not cover"):
        provider.get_funding_events(SYM, entry, exit)


def test_unfetched_history_is_not_zero_cost():
    with pytest.raises(ValueError, match="does not cover"):
        BinanceHistoricalFundingProvider().get_funding_events(SYM, 0, HOUR)


async def test_empty_successful_history_is_distinct_from_unavailable():
    provider = await history([])
    assert provider.get_funding_events(SYM, 0, HOUR) == ()


@pytest.mark.parametrize("mark", [None, "0", "-1", "nan", "inf"])
async def test_missing_or_invalid_mark_cannot_fall_back_to_exit_notional(mark):
    provider = await history([row(HOUR, mark=mark)])
    with pytest.raises(ValueError, match=r"[Mm]ark price"):
        SettlementFundingModel().calculate_funding_cost(
            999999, SYM, Bias.BULLISH, 0, HOUR, provider, quantity=1
        )


@pytest.mark.parametrize("quantity", [None, 0, -1, float("nan"), float("inf")])
def test_quantity_must_be_explicit_and_valid(quantity):
    with pytest.raises(ValueError, match="base quantity"):
        SettlementFundingModel().calculate_funding_cost(
            100, SYM, Bias.BULLISH, 0, HOUR, MagicMock(), quantity=quantity
        )


def test_zero_duration_does_not_query_provider_and_reversed_time_is_rejected():
    provider = MagicMock()
    model = SettlementFundingModel()
    assert model.calculate_funding_cost(1, SYM, Bias.BULLISH, HOUR, HOUR, provider, quantity=1) == 0
    provider.get_funding_events.assert_not_called()
    with pytest.raises(ValueError, match="precedes"):
        model.calculate_funding_cost(1, SYM, Bias.BULLISH, HOUR, 0, provider, quantity=1)


@pytest.mark.parametrize("rate", [float("nan"), float("inf")])
def test_nonfinite_rates_rejected(rate):
    with pytest.raises(ValueError, match="rate"):
        FundingSettlement(SYM, HOUR, rate, 100)


@pytest.mark.parametrize("timestamp", [-1, 1.5, True])
def test_invalid_event_timestamps_rejected(timestamp):
    with pytest.raises(ValueError, match="timestamp"):
        FundingSettlement(SYM, timestamp, 0.001, 100)


@pytest.mark.parametrize("times", [[HOUR, HOUR], [2 * HOUR, HOUR], [0], [4 * HOUR]])
def test_duplicate_reversed_or_out_of_range_provider_events_are_rejected(times):
    provider = MagicMock()
    provider.get_funding_events.return_value = tuple(
        FundingSettlement(SYM, t, 0.001, 100) for t in times
    )
    with pytest.raises(ValueError, match="strictly ordered"):
        SettlementFundingModel().calculate_funding_cost(
            1, SYM, Bias.BULLISH, 0, 3 * HOUR, provider, quantity=1
        )


def test_wrong_symbol_events_rejected():
    provider = MagicMock()
    provider.get_funding_events.return_value = (
        FundingSettlement(Symbol("ETHUSDT"), HOUR, 0.001, 100),
    )
    with pytest.raises(ValueError, match="match symbol"):
        SettlementFundingModel().calculate_funding_cost(
            1, SYM, Bias.BULLISH, 0, HOUR, provider, quantity=1
        )


async def test_full_page_is_paginated_from_last_event_not_fixed_calendar_chunk():
    client = MagicMock()
    client._get_json = AsyncMock(
        side_effect=[
            [row(t) for t in range(1000)],
            [row(1000)],
        ]
    )
    provider = BinanceHistoricalFundingProvider(client=client)
    await provider.prefetch((SYM,), DAY, DAY)
    assert len(provider.get_funding_events(SYM, -1, 1000)) == 1001
    assert client._get_json.await_args_list[1].kwargs["params"] == {
        "symbol": "BTCUSDT",
        "startTime": 1000,
        "endTime": 24 * HOUR - 1,
        "limit": 1000,
    }


async def test_failed_second_page_does_not_publish_partial_history():
    client = MagicMock()
    client._get_json = AsyncMock(
        side_effect=[[row(t) for t in range(1000)], RuntimeError("offline")]
    )
    provider = BinanceHistoricalFundingProvider(client=client)
    with pytest.raises(RuntimeError, match="offline"):
        await provider.prefetch((SYM,), DAY, DAY)
    assert provider.get_funding_rate_at(SYM, HOUR) is None
    with pytest.raises(ValueError, match="does not cover"):
        provider.get_funding_events(SYM, 0, HOUR)


async def test_failed_second_symbol_does_not_refresh_first():
    client = MagicMock()
    client._get_json = AsyncMock(side_effect=[[row(HOUR)], RuntimeError("offline")])
    provider = BinanceHistoricalFundingProvider(client=client)
    with pytest.raises(RuntimeError):
        await provider.prefetch((SYM, Symbol("ETHUSDT")), DAY, DAY)
    assert provider._coverage == {}
    assert provider._rates == {}


@pytest.mark.parametrize("times", [[HOUR, HOUR], [2 * HOUR, HOUR], [24 * HOUR]])
async def test_malformed_pages_rejected(times):
    with pytest.raises(ValueError, match="timestamps"):
        await history([row(t) for t in times])


async def test_context_adapter_preserves_raw_rate_event_time_and_mark_price():
    source = MagicMock()
    source.get_funding_history = AsyncMock(
        return_value=FundingSeries(
            symbol=SYM,
            items=(
                FundingContext(
                    0.003, 0.999, 10, 123, 9 * HOUR, TimeContext(HOUR, HOUR + 1, HOUR + 2)
                ),
            ),
        )
    )
    provider = BinanceHistoricalFundingProvider(funding_provider=source)
    await provider.prefetch((SYM,), DAY, DAY)
    assert provider.get_funding_events(SYM, 0, HOUR) == (FundingSettlement(SYM, HOUR, 0.003, 123),)


async def test_unavailable_context_history_propagates():
    source = MagicMock()
    source.get_funding_history = AsyncMock(return_value=None)
    provider = BinanceHistoricalFundingProvider(funding_provider=source)
    with pytest.raises(ValueError, match="unavailable"):
        await provider.prefetch((SYM,), DAY, DAY)


@pytest.mark.parametrize(
    "direction,price,funding", [(Bias.BULLISH, 120, 0.2), (Bias.BEARISH, 80, -0.2)]
)
async def test_paper_close_uses_position_quantity_and_event_mark(direction, price, funding):
    provider = await history([row(HOUR, mark="50")])
    portfolio = PortfolioEngine(10000)
    portfolio.open_position(
        OpenPosition(
            symbol=SYM,
            direction=direction,
            entry_price=100,
            quantity=4,
            position_size=400,
            stop_loss=90 if direction == Bias.BULLISH else 110,
            take_profit=110 if direction == Bias.BULLISH else 90,
            opened_at=0,
        )
    )
    execution = PaperExecutionEngine(
        portfolio, funding_model=SettlementFundingModel(), funding_provider=provider
    )
    execution.process_market_batch(
        {SYM: make_candle(HOUR, open_price=price, high=price + 1, low=price - 1, close=price)}
    )
    (closed,) = portfolio.history
    assert closed.execution_summary.funding_cost == pytest.approx(funding)
    assert closed.execution_summary.gross_pnl == 80
    assert portfolio.state.account.total_capital == pytest.approx(
        10000 + 80 - 4 * price * 0.0002 - funding
    )


async def test_bounded_windows_include_every_hour_for_more_than_1000_events():
    client = MagicMock()
    rows = [row(h * HOUR) for h in range(60 * 24)]

    async def fetch(_path, *, params, weight):
        return [r for r in rows if params["startTime"] <= r["fundingTime"] <= params["endTime"]][
            : params["limit"]
        ]

    client._get_json = AsyncMock(side_effect=fetch)
    provider = BinanceHistoricalFundingProvider(client=client)
    await provider.prefetch((SYM,), DAY, date(1970, 3, 1))
    assert len(provider.get_funding_events(SYM, -1, 60 * 24 * HOUR - 1)) == 1440
    assert client._get_json.await_count == 2


def test_neutral_direction_cannot_be_charged_as_short():
    with pytest.raises(ValueError, match="long or short"):
        SettlementFundingModel().calculate_funding_cost(
            100, SYM, Bias.NEUTRAL, 0, HOUR, MagicMock(), quantity=1
        )
