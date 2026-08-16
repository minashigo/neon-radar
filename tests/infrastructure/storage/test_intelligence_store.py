
from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType, SourceReliability
from neon_radar.domain.market_intelligence.history import IntelligenceObservation
from neon_radar.domain.market_intelligence.models import IntelligenceSignal
from neon_radar.infrastructure.storage.intelligence_store import HistoricalIntelligenceStore


def make_obs(value: float, obs_time: int, avail_time: int) -> IntelligenceObservation:
    sig = IntelligenceSignal(
        type=IntelligenceSignalType.DVOL,
        direction=0.0,
        strength=0.0,
        event_timestamp=obs_time,
        ingestion_timestamp=avail_time,
        source_id="test",
        provider_name="Test",
        provider_type="API",
        reliability=SourceReliability.ANALYTICS,
        weight=1.0,
        metadata={"value": str(value)},
    )
    return IntelligenceObservation(
        signal=sig, observation_timestamp=obs_time, available_at=avail_time
    )


def test_historical_intelligence_store_append_and_load(tmp_path):
    store = HistoricalIntelligenceStore(tmp_path)

    # Init empty
    assert store.load_series("DVOL") is None

    # Append new
    obs1 = make_obs(10.0, 1000, 1500)
    obs2 = make_obs(20.0, 2000, 2500)
    store.append_series("DVOL", [obs1, obs2])

    series = store.load_series("DVOL")
    assert series is not None
    assert len(series) == 2
    assert series.items[0].available_at == 1500
    assert series.items[1].available_at == 2500

    # Append overlapping + new
    obs2_duplicate = make_obs(99.0, 2000, 2500)  # Same available_at as obs2
    obs3 = make_obs(30.0, 3000, 3500)

    store.append_series("DVOL", [obs2_duplicate, obs3])

    series2 = store.load_series("DVOL")
    assert len(series2) == 3
    # Existing observation should be kept over duplicate
    assert float(series2.items[1].signal.metadata["value"]) == 20.0
    assert series2.items[2].available_at == 3500

def test_historical_intelligence_store_determinism(tmp_path):
    store = HistoricalIntelligenceStore(tmp_path)

    # Append out of order
    obs3 = make_obs(30.0, 3000, 3500)
    obs1 = make_obs(10.0, 1000, 1500)
    obs2 = make_obs(20.0, 2000, 2500)

    store.append_series("DVOL", [obs3, obs1, obs2])

    series = store.load_series("DVOL")
    # Should automatically sort by available_at
    assert len(series) == 3
    assert series.items[0].available_at == 1500
    assert series.items[1].available_at == 2500
    assert series.items[2].available_at == 3500
