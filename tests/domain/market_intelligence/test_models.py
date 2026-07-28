import pytest

from neon_radar.domain.market_intelligence.enums import (
    ConsensusDirection,
    IntelligenceSignalType,
    NarrativeType,
    SourceReliability,
)
from neon_radar.domain.market_intelligence.models import (
    IntelligenceScore,
    MarketConsensus,
    MarketNarrative,
    SignalEvidence,
    SignalSource,
)


def test_signal_source_validation():
    # Valid
    source = SignalSource(
        id="test_1",
        provider_name="TestProvider",
        provider_type="API",
        reliability=SourceReliability.OFFICIAL,
        weight=0.8,
    )
    assert source.weight == 0.8

    # Invalid weight
    with pytest.raises(ValueError, match="weight must be in"):
        SignalSource("id", "Name", "API", SourceReliability.ANONYMOUS, 1.5)


def test_signal_evidence_validation():
    source = SignalSource("id", "Name", "API", SourceReliability.OFFICIAL, 1.0)

    # Valid
    sig = SignalEvidence(
        type=IntelligenceSignalType.RSI,
        direction=1.0,
        strength=0.5,
        timestamp=1000,
        source=source,
        metadata={"key": "value"}
    )
    assert sig.direction == 1.0

    # Check immutability of metadata
    with pytest.raises(TypeError):
        sig.metadata["key"] = "new_value"

    # Invalid direction
    with pytest.raises(ValueError, match="direction must be in"):
        SignalEvidence(IntelligenceSignalType.RSI, 1.5, 0.5, 1000, source)

    # Invalid strength
    with pytest.raises(ValueError, match="strength must be in"):
        SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.5, 1000, source)


def test_market_narrative_validation():
    # Valid
    narrative = MarketNarrative(NarrativeType.RISK_ON, 0.8, 3600, 5)
    assert narrative.strength == 0.8

    # Invalid strength
    with pytest.raises(ValueError, match="strength must be in"):
        MarketNarrative(NarrativeType.RISK_ON, -0.1, 3600, 5)

    # Invalid evidence count
    with pytest.raises(ValueError, match="evidence_count must be non-negative"):
        MarketNarrative(NarrativeType.RISK_ON, 0.5, 3600, -1)


def test_market_consensus_validation():
    # Valid
    consensus = MarketConsensus(ConsensusDirection.BULLISH, 0.9, 0.1)
    assert consensus.confidence == 0.9

    # Invalid confidence
    with pytest.raises(ValueError, match="confidence must be in"):
        MarketConsensus(ConsensusDirection.BULLISH, 1.1, 0.1)

    # Invalid conflict
    with pytest.raises(ValueError, match="conflict_level must be in"):
        MarketConsensus(ConsensusDirection.BULLISH, 0.5, -0.2)


def test_intelligence_score_validation():
    # Valid
    score = IntelligenceScore(
        value=0.5,
        direction=ConsensusDirection.BULLISH,
        confidence=0.8,
        conflict=0.2,
        noise=0.1,
        coverage=0.9
    )
    assert score.value == 0.5

    # Invalid value
    with pytest.raises(ValueError, match="value must be in"):
        IntelligenceScore(1.5, ConsensusDirection.BULLISH, 0.8, 0.2, 0.1, 0.9)

    # Invalid noise
    with pytest.raises(ValueError, match="noise must be in"):
        IntelligenceScore(0.5, ConsensusDirection.BULLISH, 0.8, 0.2, -0.1, 0.9)
