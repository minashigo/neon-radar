import pytest

from neon_radar.domain.market_intelligence.enums import (
    IntelligenceSignalType,
    NarrativeType,
    SourceReliability,
)
from neon_radar.domain.market_intelligence.models import SignalEvidence, SignalSource
from neon_radar.domain.market_intelligence.narrative import NarrativeEngine


@pytest.fixture
def source():
    return SignalSource("1", "Provider", "API", SourceReliability.OFFICIAL, 1.0)


def test_etf_accumulation(source):
    engine = NarrativeEngine(min_evidence_count=2)
    sig1 = SignalEvidence(IntelligenceSignalType.ETF_FLOW, 1.0, 0.8, 1000, source)
    sig2 = SignalEvidence(IntelligenceSignalType.ETF_FLOW, 1.0, 0.9, 1000, source)

    narratives = engine.compute_narratives([sig1, sig2], 2000)
    assert len(narratives) == 1
    assert narratives[0].type == NarrativeType.ETF_ACCUMULATION
    assert narratives[0].evidence_count == 2


def test_multiple_narratives(source):
    engine = NarrativeEngine(min_evidence_count=1)

    # ETF Accumulation
    sig1 = SignalEvidence(IntelligenceSignalType.ETF_FLOW, 1.0, 0.8, 1000, source)
    # Alt season
    sig2 = SignalEvidence(IntelligenceSignalType.BTC_DOMINANCE, -1.0, 0.9, 1000, source)

    narratives = engine.compute_narratives([sig1, sig2], 2000)
    assert len(narratives) == 2
    types = {n.type for n in narratives}
    assert NarrativeType.ETF_ACCUMULATION in types
    assert NarrativeType.ALT_SEASON in types


def test_ignores_weak_narratives(source):
    engine = NarrativeEngine(min_evidence_count=2, min_strength_threshold=0.9)
    sig1 = SignalEvidence(IntelligenceSignalType.ETF_FLOW, 1.0, 0.4, 1000, source)
    sig2 = SignalEvidence(IntelligenceSignalType.ETF_FLOW, 1.0, 0.5, 1000, source)

    narratives = engine.compute_narratives([sig1, sig2], 2000)
    assert len(narratives) == 0
