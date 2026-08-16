import pytest

from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType, SourceReliability
from neon_radar.domain.market_intelligence.history import (
    HistoricalIntelligenceContext,
    IntelligenceObservation,
    IntelligenceSignalSeries,
)
from neon_radar.domain.market_intelligence.models import IntelligenceSignal


def make_signal(value: float, timestamp: int) -> IntelligenceSignal:
    return IntelligenceSignal(
        type=IntelligenceSignalType.DVOL,
        direction=0.0,
        strength=0.0,
        event_timestamp=timestamp,
        ingestion_timestamp=timestamp,
        source_id="test",
        provider_name="TestProvider",
        provider_type="API",
        reliability=SourceReliability.ANALYTICS,
        weight=1.0,
        metadata={"value": str(value)},
    )


def test_intelligence_series_slice_by_availability():
    obs1 = IntelligenceObservation(
        signal=make_signal(10.0, 1000), observation_timestamp=1000, available_at=1000
    )
    obs2 = IntelligenceObservation(
        signal=make_signal(20.0, 2000), observation_timestamp=2000, available_at=2000
    )
    obs3 = IntelligenceObservation(
        signal=make_signal(30.0, 3000), observation_timestamp=3000, available_at=3000
    )

    series = IntelligenceSignalSeries(signal_type="DVOL", items=(obs1, obs2, obs3))

    # Slice exactly at 2000 (inclusive)
    sliced = series.slice_by_availability(2000)
    assert len(sliced) == 2
    assert sliced.items[-1].available_at == 2000

    # Slice before 1000
    sliced = series.slice_by_availability(999)
    assert len(sliced) == 0

    # Slice after 3000
    sliced = series.slice_by_availability(4000)
    assert len(sliced) == 3


def test_intelligence_series_latest_valid_observation():
    obs1 = IntelligenceObservation(
        signal=make_signal(10.0, 1000), observation_timestamp=1000, available_at=1000
    )
    series = IntelligenceSignalSeries(signal_type="DVOL", items=(obs1,))

    max_staleness = 500

    # Request at 1200 (within staleness)
    obs = series.latest_valid_observation(1200, max_staleness)
    assert obs is not None
    assert obs.available_at == 1000

    # Request at 1600 (exceeds staleness)
    obs = series.latest_valid_observation(1600, max_staleness)
    assert obs is None


def test_intelligence_series_unsorted_raises():
    obs1 = IntelligenceObservation(
        signal=make_signal(10.0, 1000), observation_timestamp=1000, available_at=2000
    )
    obs2 = IntelligenceObservation(
        signal=make_signal(20.0, 2000), observation_timestamp=2000, available_at=1000
    )
    with pytest.raises(ValueError, match="not sorted ascending"):
        IntelligenceSignalSeries(signal_type="DVOL", items=(obs1, obs2))


def test_historical_context_post_init_slicing():
    obs1 = IntelligenceObservation(
        signal=make_signal(10.0, 1000), observation_timestamp=1000, available_at=1000
    )
    obs2 = IntelligenceObservation(
        signal=make_signal(20.0, 2000), observation_timestamp=2000, available_at=2000
    )
    series = IntelligenceSignalSeries(signal_type="DVOL", items=(obs1, obs2))

    context = HistoricalIntelligenceContext(
        timestamp=1500, series_map={"DVOL": series}
    )

    # Should automatically slice at 1500
    sliced_series = context.series_map["DVOL"]
    assert len(sliced_series) == 1
    assert sliced_series.items[-1].available_at == 1000


def test_historical_context_slice_at():
    obs1 = IntelligenceObservation(
        signal=make_signal(10.0, 1000), observation_timestamp=1000, available_at=1000
    )
    obs2 = IntelligenceObservation(
        signal=make_signal(20.0, 2000), observation_timestamp=2000, available_at=2000
    )
    series = IntelligenceSignalSeries(signal_type="DVOL", items=(obs1, obs2))

    context = HistoricalIntelligenceContext(
        timestamp=3000, series_map={"DVOL": series}
    )

    # Context contains both
    assert len(context.series_map["DVOL"]) == 2

    # Slice at 1500
    sliced_context = context.slice_at(1500)
    assert sliced_context.timestamp == 1500
    assert len(sliced_context.series_map["DVOL"]) == 1
