from neon_radar.domain.market_intelligence.enums import (
    IntelligenceSignalType,
    SourceReliability,
)
from neon_radar.domain.market_intelligence.models import IntelligenceSignal
from neon_radar.domain.market_intelligence.noise_filter import NoiseFilter


def create_signal(
    provider_name: str,
    reliability: SourceReliability,
    weight: float,
    timestamp: int,
    direction: float = 1.0,
    strength: float = 1.0,
    type_: IntelligenceSignalType = IntelligenceSignalType.RSI,
) -> IntelligenceSignal:
    return IntelligenceSignal(
        type=type_,
        direction=direction,
        strength=strength,
        event_timestamp=timestamp,
        ingestion_timestamp=timestamp + 10,
        source_id=f"{provider_name}_id",
        provider_name=provider_name,
        provider_type="API",
        reliability=reliability,
        weight=weight,
    )


def test_filters_out_low_reliability():
    filter_engine = NoiseFilter(min_reliability_threshold=0.2)

    sig1 = create_signal("Base", SourceReliability.NEWS, 0.5, 1000)
    sig2 = create_signal("Anon", SourceReliability.ANONYMOUS, 0.1, 1000)

    # We should disable require_independent_confirmation for this test to strictly test reliability.
    filter_engine.require_independent_confirmation = False

    filtered = filter_engine.filter_signals([sig1, sig2])
    assert len(filtered) == 1
    assert filtered[0].provider_name == "Base"


def test_deduplicates_same_provider_within_window():
    filter_engine = NoiseFilter(time_window_ms=1000, require_independent_confirmation=False)

    sig1 = create_signal("Base", SourceReliability.NEWS, 0.5, 1000)
    sig2 = create_signal("Base", SourceReliability.NEWS, 0.5, 1500)  # Duplicate
    sig3 = create_signal("Base", SourceReliability.NEWS, 0.5, 2500)  # Outside window

    filtered = filter_engine.filter_signals([sig1, sig2, sig3])
    assert len(filtered) == 2
    assert filtered[0].event_timestamp == 1500
    assert filtered[1].event_timestamp == 2500


def test_requires_independent_confirmation():
    filter_engine = NoiseFilter(time_window_ms=1000, require_independent_confirmation=True)

    # Only one provider
    sig1 = create_signal("Base", SourceReliability.NEWS, 0.5, 1000)
    filtered = filter_engine.filter_signals([sig1])
    assert len(filtered) == 0

    # Two providers
    sig2 = create_signal("Other", SourceReliability.NEWS, 0.5, 1500)

    filtered2 = filter_engine.filter_signals([sig1, sig2])
    assert len(filtered2) == 2


def test_official_bypasses_confirmation():
    filter_engine = NoiseFilter(require_independent_confirmation=True)

    # Only one provider but it's official
    sig1 = create_signal("Official", SourceReliability.OFFICIAL, 1.0, 1000)
    filtered = filter_engine.filter_signals([sig1])
    assert len(filtered) == 1


def test_exempt_signals_bypass_confirmation():
    filter_engine = NoiseFilter(
        require_independent_confirmation=True, exempt_signal_types={"liquidations"}
    )

    sig1 = create_signal(
        "CoinGlass",
        SourceReliability.ANALYTICS,
        1.0,
        1000,
        type_=IntelligenceSignalType.LIQUIDATIONS,
    )
    sig2 = create_signal(
        "CoinGlass",
        SourceReliability.ANALYTICS,
        1.0,
        1000,
        type_=IntelligenceSignalType.RSI,
    )

    filtered = filter_engine.filter_signals([sig1, sig2])
    # Liquidations passes because it's exempt, RSI gets dropped because it requires confirmation
    assert len(filtered) == 1
    assert filtered[0].type == IntelligenceSignalType.LIQUIDATIONS
