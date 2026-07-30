import pytest

from neon_radar.domain.market_intelligence.enums import (
    IntelligenceSignalType,
    SourceReliability,
)
from neon_radar.domain.market_intelligence.models import SignalEvidence, SignalSource
from neon_radar.domain.market_intelligence.noise_filter import NoiseFilter


@pytest.fixture
def base_source():
    return SignalSource("id", "Base", "API", SourceReliability.NEWS, 0.5)


@pytest.fixture
def low_rel_source():
    return SignalSource("id2", "Anon", "Social", SourceReliability.ANONYMOUS, 0.1)


@pytest.fixture
def official_source():
    return SignalSource("id3", "Official", "API", SourceReliability.OFFICIAL, 1.0)


def test_filters_out_low_reliability(base_source, low_rel_source):
    filter_engine = NoiseFilter(min_reliability_threshold=0.2)

    sig1 = SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.0, 1000, base_source)
    sig2 = SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.0, 1000, low_rel_source)

    filtered = filter_engine.filter_signals([sig1, sig2])
    # Note: wait, require_independent_confirmation is True by default.
    # We should disable it for this test to strictly test reliability.
    filter_engine.require_independent_confirmation = False

    filtered = filter_engine.filter_signals([sig1, sig2])
    assert len(filtered) == 1
    assert filtered[0].source.provider_name == "Base"


def test_deduplicates_same_provider_within_window(base_source):
    filter_engine = NoiseFilter(time_window_ms=1000, require_independent_confirmation=False)

    sig1 = SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.0, 1000, base_source)
    sig2 = SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.0, 1500, base_source) # Duplicate
    sig3 = SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.0, 2500, base_source) # Outside window

    filtered = filter_engine.filter_signals([sig1, sig2, sig3])
    assert len(filtered) == 2
    assert filtered[0].timestamp == 1500
    assert filtered[1].timestamp == 2500


def test_requires_independent_confirmation(base_source):
    filter_engine = NoiseFilter(time_window_ms=1000, require_independent_confirmation=True)

    # Only one provider
    sig1 = SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.0, 1000, base_source)
    filtered = filter_engine.filter_signals([sig1])
    assert len(filtered) == 0

    # Two providers
    other_source = SignalSource("id4", "Other", "API", SourceReliability.NEWS, 0.5)
    sig2 = SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.0, 1500, other_source)

    filtered2 = filter_engine.filter_signals([sig1, sig2])
    assert len(filtered2) == 2


def test_official_bypasses_confirmation(official_source):
    filter_engine = NoiseFilter(require_independent_confirmation=True)

    # Only one provider but it's official
    sig1 = SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.0, 1000, official_source)
    filtered = filter_engine.filter_signals([sig1])
    assert len(filtered) == 1
