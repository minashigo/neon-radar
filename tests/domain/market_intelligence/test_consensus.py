import pytest

from neon_radar.domain.market_intelligence.consensus import ConsensusEngine
from neon_radar.domain.market_intelligence.enums import (
    ConsensusDirection,
    IntelligenceSignalType,
    SourceReliability,
)
from neon_radar.domain.market_intelligence.models import SignalEvidence, SignalSource


@pytest.fixture
def source_strong():
    return SignalSource("1", "Strong", "API", SourceReliability.OFFICIAL, 1.0)


@pytest.fixture
def source_weak():
    return SignalSource("2", "Weak", "News", SourceReliability.NEWS, 0.4)


def test_bullish_consensus(source_strong):
    engine = ConsensusEngine()
    sig1 = SignalEvidence(IntelligenceSignalType.RSI, 1.0, 1.0, 1000, source_strong)
    sig2 = SignalEvidence(IntelligenceSignalType.MACD, 1.0, 0.8, 1000, source_strong)

    consensus = engine.compute_consensus([sig1, sig2])
    assert consensus.direction == ConsensusDirection.BULLISH
    assert consensus.confidence > 0.0
    assert consensus.conflict_level == 0.0


def test_bearish_consensus(source_strong):
    engine = ConsensusEngine()
    sig1 = SignalEvidence(IntelligenceSignalType.RSI, -1.0, 1.0, 1000, source_strong)

    consensus = engine.compute_consensus([sig1])
    assert consensus.direction == ConsensusDirection.BEARISH
    assert consensus.conflict_level == 0.0


def test_conflicting_consensus(source_strong, source_weak):
    engine = ConsensusEngine(conflict_threshold=0.4)
    # Strong bearish and slightly weaker bullish
    sig1 = SignalEvidence(IntelligenceSignalType.RSI, -1.0, 1.0, 1000, source_strong)  # power: -1.0
    sig2 = SignalEvidence(IntelligenceSignalType.MACD, 1.0, 1.0, 1000, source_strong)  # power: 1.0

    consensus = engine.compute_consensus([sig1, sig2])
    assert consensus.direction == ConsensusDirection.CONFLICTING
    assert consensus.conflict_level == 1.0  # Perfect conflict


def test_neutral_consensus(source_weak):
    engine = ConsensusEngine(bullish_threshold=0.5, bearish_threshold=-0.5)
    # Very weak signal doesn't break threshold
    sig1 = SignalEvidence(IntelligenceSignalType.RSI, 0.2, 0.1, 1000, source_weak)

    consensus = engine.compute_consensus([sig1])
    assert consensus.direction == ConsensusDirection.NEUTRAL
