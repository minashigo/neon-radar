"""Tests for the TradeBacktester."""

from datetime import date
from unittest.mock import AsyncMock

import pytest

from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.trading.backtest import TradeStatus


@pytest.fixture
def mock_exchange():
    return AsyncMock()


@pytest.fixture
def scoring_config():
    return ScoringRulesConfig(min_confidence=0.5, rules=[])


@pytest.fixture
def empty_rules():
    return tuple()


@pytest.mark.asyncio
async def test_trade_backtester_empty_result(mock_exchange, scoring_config, empty_rules):
    tester = TradeBacktester(
        exchange=mock_exchange,
        scoring_config=scoring_config,
        rules=empty_rules,
    )

    mock_exchange.get_klines.return_value = KlineSeries(
        symbol=Symbol("BTCUSDT"), timeframe="1d", candles=()
    )

    trades = await tester.run(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        symbols=[Symbol("BTCUSDT")],
        timeframe="1d",
        min_history_candles=5,
    )

    assert isinstance(trades, tuple)
    assert len(trades) == 0


@pytest.mark.asyncio
async def test_trade_backtester_with_mocked_analysis(
    mock_exchange, scoring_config, empty_rules, monkeypatch
):
    # Create fake candles
    candles = []
    base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
    for i in range(10):
        c = OHLCV(
            open_time=base_time + i * 86400000,
            open=100.0 + i,
            high=105.0 + i,
            low=95.0 + i,
            close=101.0 + i,
            volume=1000.0,
            quote_volume=100000.0,
            trades=100,
        )
        candles.append(c)

    series = KlineSeries(symbol=Symbol("BTCUSDT"), timeframe="1d", candles=tuple(candles))
    mock_exchange.get_klines.return_value = series

    # Mock pipeline evaluate
    from neon_radar.domain.risk import RiskDecision
    from neon_radar.domain.trading.setup import FinalTradeSetup

    def fake_evaluate(self, series, *args, **kwargs):
        # Trigger a trade on candle index 5 (which is the 6th candle)
        # We look at history up to index 5, so history length is 6.
        if len(series.candles) == 6:
            return FinalTradeSetup(
                symbol=Symbol("BTCUSDT"),
                direction=Bias.BULLISH,
                entry=106.0,  # The next candle (i=6) has low=101.0, high=111.0 -> will hit
                stop_loss=100.0,
                take_profit=110.0,
                confidence=1.0,
                score=1.0,
                risk_decision=RiskDecision(True),
                position_size=1.0,
                quote_size=106.0,
                risk_amount=6.0,
                expected_reward=4.0,
                diagnostics=None
            )
        return None

    monkeypatch.setattr(
        "neon_radar.application.services.trading_pipeline.TradingPipeline.evaluate", fake_evaluate
    )

    tester = TradeBacktester(
        exchange=mock_exchange,
        scoring_config=scoring_config,
        rules=empty_rules,
    )

    trades = await tester.run(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        symbols=[Symbol("BTCUSDT")],
        timeframe="1d",
        min_history_candles=5,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade.symbol == Symbol("BTCUSDT")
    assert trade.direction == Bias.BULLISH
    assert trade.entry_price == 106.0
    assert trade.exit_price is not None
    assert trade.status in (TradeStatus.WIN, TradeStatus.LOSS, TradeStatus.BREAK_EVEN)
    assert trade.status == TradeStatus.WIN
    assert trade.exit_price == 110.0
