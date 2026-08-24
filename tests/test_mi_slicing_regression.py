"""Look-Ahead Bias regression tests for Market Intelligence."""

from datetime import UTC, datetime

import pytest

from neon_radar.application.services.trade_backtester import TradeBacktester
from neon_radar.config.models import TimeFrame
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType, SourceReliability
from neon_radar.domain.market_intelligence.history import (
    IntelligenceObservation,
    IntelligenceSignalSeries,
)
from neon_radar.domain.market_intelligence.models import IntelligenceSignal
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol


def _create_observation(sig_type: str, obs_time_ms: int, val: float, available_at_ms: int) -> IntelligenceObservation:
    sig = IntelligenceSignal(
        type=IntelligenceSignalType(sig_type),
        direction=0.5,
        strength=0.5,
        event_timestamp=obs_time_ms,
        ingestion_timestamp=obs_time_ms,
        weight=1.0,
        provider_name="test",
        provider_type="test",
        source_id="test",
        reliability=SourceReliability.OFFICIAL,
        metadata={"raw_value": str(val)}
    )
    return IntelligenceObservation(
        signal=sig,
        observation_timestamp=obs_time_ms,
        available_at=available_at_ms
    )


class MockIntelligenceStore:
    def __init__(self, fng_series=None, dvol_series=None, pcr_series=None):
        self.fng = fng_series
        self.dvol = dvol_series
        self.pcr = pcr_series

    def load_series(self, sig_type: str):
        if sig_type == "fear_and_greed":
            return self.fng
        if sig_type == "dvol":
            return self.dvol
        if sig_type == "put_call_ratio":
            return self.pcr
        return None

class MockExchange:
    pass


@pytest.mark.asyncio
async def test_no_look_ahead_bias_in_mi_features():
    """Prove that an MI observation with available_at > T cannot change MarketIntelligenceFeatures at T."""

    # We want to test a point in time t_stamp.
    # t_stamp = 1000. Candle open time is 1000.
    t_stamp = 1000

    # Obs 1: available at 900
    obs1 = _create_observation("fear_and_greed", 900, 50.0, 900)
    # Obs 2: available at 1000 (just in time)
    obs2 = _create_observation("fear_and_greed", 1000, 60.0, 1000)
    # Obs 3: available at 1001 (FUTURE)
    obs3 = _create_observation("fear_and_greed", 1001, 99.0, 1001)

    fng_safe = IntelligenceSignalSeries("fear_and_greed", (obs1, obs2))
    fng_future = IntelligenceSignalSeries("fear_and_greed", (obs1, obs2, obs3))

    store_safe = MockIntelligenceStore(fng_series=fng_safe)
    store_future = MockIntelligenceStore(fng_series=fng_future)

    # We don't need a full run, we just need to see how _simulate_symbol builds MarketIntelligenceFeatures.

    series = KlineSeries(
        symbol=Symbol("BTCUSDT"),
        timeframe=TimeFrame("1d"),
        candles=(
            OHLCV(open_time=t_stamp, open=100, high=110, low=90, close=105, volume=1000, close_time=t_stamp+86399999),
        )
    )

    cfg = ScoringRulesConfig(min_confidence=0.5, rules=[])

    bt_safe = TradeBacktester(
        exchange=MockExchange(),
        scoring_config=cfg,
        rules=(),
        preloaded_series={("BTCUSDT", "1d"): series},
        intelligence_store=store_safe
    )

    # Hack into _simulate_symbol's feature construction code to capture the constructed features.
    features_safe = None

    def mock_eval(*args, **kwargs):
        nonlocal features_safe
        features_safe = kwargs.get("intelligence")
        return None
    bt_safe._pipeline.evaluate = mock_eval

    bt_safe._simulate_symbol(
        symbol=Symbol("BTCUSDT"),
        timeframe="1d",
        start_date=datetime.fromtimestamp(0, UTC).date(),
        end_date=datetime.fromtimestamp(100, UTC).date(),
        min_history_candles=0
    )

    # Run 2: future store
    bt_future = TradeBacktester(
        exchange=MockExchange(),
        scoring_config=cfg,
        rules=(),
        preloaded_series={("BTCUSDT", "1d"): series},
        intelligence_store=store_future
    )
    features_future = None
    def mock_eval_future(*args, **kwargs):
        nonlocal features_future
        features_future = kwargs.get("intelligence")
        return None
    bt_future._pipeline.evaluate = mock_eval_future

    bt_future._simulate_symbol(
        symbol=Symbol("BTCUSDT"),
        timeframe="1d",
        start_date=datetime.fromtimestamp(0, UTC).date(),
        end_date=datetime.fromtimestamp(100, UTC).date(),
        min_history_candles=0
    )

    print("Features safe:", features_safe)
    print("Features future:", features_future)
    assert features_safe is not None
    assert features_future is not None

    # The extracted value should be exactly the same, ignoring the future value of 99.0.
    assert features_safe.fng_value == 60.0
    assert features_future.fng_value == 60.0
    # Meaning future data was successfully sliced out.


@pytest.mark.asyncio
async def test_missing_pcr_does_not_affect_history():
    """Verify missing PCR doesn't affect historical construction, and missing F&G/DVOL omits them."""

    t_stamp = 1000

    # Only DVOL exists
    dvol_obs = _create_observation("dvol", 900, 45.0, 900)
    store = MockIntelligenceStore(dvol_series=IntelligenceSignalSeries("dvol", (dvol_obs,)))

    series = KlineSeries(
        symbol=Symbol("BTCUSDT"),
        timeframe=TimeFrame("1d"),
        candles=(
            OHLCV(open_time=t_stamp, open=100, high=110, low=90, close=105, volume=1000, close_time=t_stamp+86399999),
        )
    )
    cfg = ScoringRulesConfig(min_confidence=0.5, rules=[])

    bt = TradeBacktester(
        exchange=MockExchange(),
        scoring_config=cfg,
        rules=(),
        preloaded_series={("BTCUSDT", "1d"): series},
        intelligence_store=store
    )

    features = None
    def mock_eval(*args, **kwargs):
        nonlocal features
        features = kwargs.get("intelligence")
        return None
    bt._pipeline.evaluate = mock_eval

    bt._simulate_symbol(
        symbol=Symbol("BTCUSDT"),
        timeframe="1d",
        start_date=datetime.fromtimestamp(0, UTC).date(),
        end_date=datetime.fromtimestamp(100, UTC).date(),
        min_history_candles=0
    )

    assert features is not None
    assert getattr(features, "fng_value", None) is None
    assert getattr(features, "pcr_value", None) is None
    assert features.dvol_value == 45.0
