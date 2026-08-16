import math

from neon_radar.application.intelligence.normalizer import IntelligenceNormalizer
from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType, SourceReliability
from neon_radar.domain.market_intelligence.history import (
    IntelligenceObservation,
    IntelligenceSignalSeries,
)
from neon_radar.domain.market_intelligence.models import IntelligenceSignal


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
        metadata={"raw_value": str(value)},
    )
    return IntelligenceObservation(
        signal=sig, observation_timestamp=obs_time, available_at=avail_time
    )


def test_normalizer_z_score():
    # Values: 10, 20, 30
    obs1 = make_obs(10.0, 1000, 1000)
    obs2 = make_obs(20.0, 2000, 2000)
    obs3 = make_obs(30.0, 3000, 3000)

    series = IntelligenceSignalSeries(signal_type="DVOL", items=(obs1, obs2, obs3))

    # mean = 20
    # variance = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 2 = (100 + 0 + 100) / 2 = 100
    # std_dev = 10
    # current = 30
    # z_score = (30 - 20) / 10 = 1.0

    z = IntelligenceNormalizer.calculate_rolling_z_score(series, window_size=3)
    assert z is not None
    assert math.isclose(z, 1.0)


def test_normalizer_percentile():
    # Values: 10, 20, 30, 40
    # If window is 4 and current is 40.
    # less=3, equal=1
    # perc = (3 + 0.5) / 4 = 3.5 / 4 = 0.875

    obs1 = make_obs(10.0, 1000, 1000)
    obs2 = make_obs(20.0, 2000, 2000)
    obs3 = make_obs(30.0, 3000, 3000)
    obs4 = make_obs(40.0, 4000, 4000)

    series = IntelligenceSignalSeries(signal_type="DVOL", items=(obs1, obs2, obs3, obs4))

    p = IntelligenceNormalizer.calculate_percentile(series, window_size=4)
    assert p is not None
    assert math.isclose(p, 0.875)


def test_normalizer_look_ahead_bias_regression():
    """Ensure normalizer cannot see future data if series is sliced correctly."""
    # Data up to 3000 is normal (10, 20, 30)
    # At 4000 there is a massive spike (1000.0) which would skew the mean heavily
    obs1 = make_obs(10.0, 1000, 1000)
    obs2 = make_obs(20.0, 2000, 2000)
    obs3 = make_obs(30.0, 3000, 3000)
    obs4_future_spike = make_obs(1000.0, 4000, 4000)

    full_series = IntelligenceSignalSeries(signal_type="DVOL", items=(obs1, obs2, obs3, obs4_future_spike))

    # We are evaluating at T = 3500.
    # We MUST slice the series before normalizing.
    sliced_series = full_series.slice_by_availability(3500)

    # The normalizer only receives the sliced series
    z = IntelligenceNormalizer.calculate_rolling_z_score(sliced_series, window_size=3)

    # If look-ahead bias occurred, the future spike (1000.0) would drastically alter the mean and Z-Score.
    # But since it's sliced, the result should be exactly 1.0 (mean=20, std=10, current=30).
    assert z is not None
    assert math.isclose(z, 1.0)

    # Also verify that the future spike is completely invisible to the normalizer's raw extract
    raw = IntelligenceNormalizer.extract_raw_value(sliced_series)
    assert raw == 30.0
