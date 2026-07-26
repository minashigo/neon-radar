"""Integration tests for the Trading Pipeline."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from neon_radar.application.services.risk.drawdown import DrawdownMonitor
from neon_radar.application.services.risk.manager import RiskManager, RiskManagerConfig
from neon_radar.application.services.risk.sizing import FixedRiskStrategy, PositionSizingEngine
from neon_radar.application.services.trading_pipeline import TradingPipeline
from neon_radar.domain.enums import Bias
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.models import Kline, KlineSeries, Symbol
from neon_radar.domain.risk import AccountState, PortfolioState, PositionState
from neon_radar.domain.scoring.value_objects import AnalysisResult, Score
from neon_radar.domain.trading.backtest import TradeDiagnostics, TradeEntryReason
from neon_radar.domain.trading.setup import FinalTradeSetup, TradeSetup, TradeSetupEngine


@pytest.fixture
def dummy_series():
    return KlineSeries(
        symbol=Symbol("BTC", "USDT"),
        timeframe="1d",
        candles=(
            Kline(
                open_time=int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1000),
                open=10000.0,
                high=10500.0,
                low=9500.0,
                close=10000.0,
                volume=100.0,
            ),
        ),
    )

@pytest.fixture
def mock_analyze_result(dummy_series):
    market_state = MarketState(
        symbol=dummy_series.symbol,
        timestamp=dummy_series.candles[0].open_time,
        primary_series=dummy_series,
        indicator_series=()
    )
    return AnalysisResult(
        score=Score(value=1.0, confidence=0.8, bias=Bias.BULLISH),
        signals=(),
        summary="Mock",
        computed_at=dummy_series.candles[0].open_time,
        market_state=market_state
    )

@pytest.fixture
def mock_trade_setup():
    return TradeSetup(
        direction=Bias.BULLISH,
        entry_price=10000.0,
        stop_loss=9000.0,
        take_profit_1=11500.0,
        take_profit_2=13000.0,
        risk_reward=(1.5, 3.0),
        diagnostics=TradeDiagnostics(
            adx=25.0,
            atr=500.0,
            rsi=60.0,
            ema_spread_pct=1.0,
            htf_trend=1.0,
            confidence=0.8,
            final_score=1.0,
            triggered_rules="mock",
            entry_reason=TradeEntryReason.CONFIDENCE_THRESHOLD,
            regime="bullish",
            regime_reason="mock",
        )
    )

@pytest.fixture
def pipeline():
    setup_engine = MagicMock(spec=TradeSetupEngine)
    setup_engine.required_indicators.return_value = ()

    risk_manager = RiskManager(RiskManagerConfig(max_open_positions=1))
    sizing_engine = PositionSizingEngine(FixedRiskStrategy())

    return TradingPipeline(
        rules=(),
        setup_engine=setup_engine,
        risk_manager=risk_manager,
        sizing_engine=sizing_engine,
    )

@patch("neon_radar.application.services.trading_pipeline.analyze_series")
def test_pipeline_valid_buy(mock_analyze_series, pipeline, dummy_series, mock_analyze_result, mock_trade_setup):
    """Test a valid buy signal successfully creates a FinalTradeSetup."""
    mock_analyze_series.return_value = mock_analyze_result
    pipeline.setup_engine.build_setup.return_value = mock_trade_setup

    portfolio = PortfolioState(account=AccountState(total_capital=10000.0, free_capital=10000.0))

    final_setup = pipeline.evaluate(dummy_series, portfolio=portfolio)

    assert final_setup is not None
    assert isinstance(final_setup, FinalTradeSetup)
    assert final_setup.direction == Bias.BULLISH
    assert final_setup.risk_decision.is_allowed is True
    # Default risk budget is 0.01 * 10000 = 100
    # Stop loss distance = 1000
    # Base size = 100 / 1000 = 0.1
    assert final_setup.position_size == 0.1
    assert final_setup.quote_size == 1000.0
    assert final_setup.risk_amount == 100.0

@patch("neon_radar.application.services.trading_pipeline.analyze_series")
def test_pipeline_risk_rejection(mock_analyze_series, pipeline, dummy_series, mock_analyze_result, mock_trade_setup):
    """Test that the RiskManager correctly blocks the trade if max positions are reached."""
    mock_analyze_series.return_value = mock_analyze_result
    pipeline.setup_engine.build_setup.return_value = mock_trade_setup

    # Portfolio already has 1 open position, max is 1
    existing_pos = PositionState(
        symbol=Symbol("ETH", "USDT"),
        side=Bias.BULLISH,
        entry_price=2000.0,
        size=1.0,
        stop_loss=1900.0
    )
    portfolio = PortfolioState(
        account=AccountState(total_capital=10000.0, free_capital=8000.0),
        positions=(existing_pos,)
    )

    final_setup = pipeline.evaluate(dummy_series, portfolio=portfolio)

    assert final_setup is None

@patch("neon_radar.application.services.trading_pipeline.analyze_series")
def test_pipeline_drawdown_block(mock_analyze_series, pipeline, dummy_series, mock_analyze_result, mock_trade_setup):
    """Test that Drawdown penalty blocks the trade."""
    mock_analyze_series.return_value = mock_analyze_result
    pipeline.setup_engine.build_setup.return_value = mock_trade_setup

    # Configure RiskManager with max drawdown allowed
    pipeline.risk_manager.config = RiskManagerConfig(max_drawdown_pct=0.10)

    portfolio = PortfolioState(account=AccountState(total_capital=8000.0, free_capital=8000.0))
    # ATH was 10000.0, current is 8000.0 => 20% drawdown
    monitor = DrawdownMonitor(initial_capital=10000.0)
    drawdown = monitor.update(8000.0, dummy_series.candles[0].open_time)

    final_setup = pipeline.evaluate(dummy_series, portfolio=portfolio, drawdown=drawdown)

    assert final_setup is None
