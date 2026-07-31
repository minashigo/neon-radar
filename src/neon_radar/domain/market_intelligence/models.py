"""Immutable domain models for Market Intelligence."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from neon_radar.domain.market_intelligence.enums import (
        ConsensusDirection,
        IntelligenceSignalType,
        NarrativeType,
        SourceReliability,
    )


@dataclass(slots=True, frozen=True)
class DataQuality:
    """Metrics regarding the quality and reliability of the fetched data."""

    latency_ms: float
    error_count: int
    is_stale: bool


@dataclass(slots=True, frozen=True)
class IntelligenceSignal:
    """Internal model for a raw intelligence signal within the pipeline."""

    type: IntelligenceSignalType
    direction: float  # -1.0 to 1.0
    strength: float  # 0.0 to 1.0
    event_timestamp: int
    ingestion_timestamp: int
    source_id: str
    provider_name: str
    provider_type: str
    reliability: SourceReliability
    weight: float
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError(
                f"IntelligenceSignal.direction must be in [-1, 1], got {self.direction}"
            )
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"IntelligenceSignal.strength must be in [0, 1], got {self.strength}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"IntelligenceSignal.weight must be in [0, 1], got {self.weight}")
        if isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(slots=True, frozen=True)
class ProviderResult:
    """Result of a provider's fetch operation."""

    signals: tuple[IntelligenceSignal, ...]
    quality: DataQuality


@dataclass(slots=True, frozen=True)
class PipelineContext:
    """Immutable context passed through the pipeline."""

    run_id: str
    timestamp: int
    active_providers: tuple[str, ...]
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(slots=True, frozen=True)
class SignalSource:
    """Represents the origin of an intelligence signal."""

    id: str
    provider_name: str
    provider_type: str  # e.g., 'API', 'Scraper', 'Webhook'
    reliability: SourceReliability
    weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"SignalSource.weight must be in [0, 1], got {self.weight}")


@dataclass(slots=True, frozen=True)
class SignalEvidence:
    """A specific piece of evidence/signal from a source."""

    type: IntelligenceSignalType
    direction: float  # -1.0 to 1.0
    strength: float  # 0.0 to 1.0
    timestamp: int
    source: SignalSource
    # Metadata is immutable via MappingProxyType to prevent modification.
    # It allows lookups but no modifications.
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError(f"SignalEvidence.direction must be in [-1, 1], got {self.direction}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"SignalEvidence.strength must be in [0, 1], got {self.strength}")
        # Enforce metadata immutability type check (not foolproof at runtime but catches direct dicts)
        if isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(slots=True, frozen=True)
class MarketNarrative:
    """Represents an active market narrative."""

    type: NarrativeType
    strength: float  # 0.0 to 1.0
    duration: int  # Milliseconds or specific time representation
    evidence_count: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"MarketNarrative.strength must be in [0, 1], got {self.strength}")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be non-negative")


@dataclass(slots=True, frozen=True)
class MarketConsensus:
    """Represents the overall consensus of the market signals."""

    direction: ConsensusDirection
    confidence: float  # 0.0 to 1.0
    conflict_level: float  # 0.0 to 1.0 (how much opposing evidence exists)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"MarketConsensus.confidence must be in [0, 1], got {self.confidence}")
        if not 0.0 <= self.conflict_level <= 1.0:
            raise ValueError(
                f"MarketConsensus.conflict_level must be in [0, 1], got {self.conflict_level}"
            )


@dataclass(slots=True, frozen=True)
class IntelligenceScore:
    """Aggregated score representing the quality and direction of the intelligence context."""

    value: float  # -1.0 to 1.0 (overall bullish/bearish score)
    direction: ConsensusDirection
    confidence: float  # 0.0 to 1.0 (how confident we are in the value)
    conflict: float  # 0.0 to 1.0 (amount of opposing evidence)
    noise: float  # 0.0 to 1.0 (ratio of low-quality/unverified signals)
    coverage: float  # 0.0 to 1.0 (how many signal categories are covered)

    def __post_init__(self) -> None:
        if not -1.0 <= self.value <= 1.0:
            raise ValueError(f"IntelligenceScore.value must be in [-1, 1], got {self.value}")
        for field_name in ("confidence", "conflict", "noise", "coverage"):
            val = getattr(self, field_name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"IntelligenceScore.{field_name} must be in [0, 1], got {val}")


@dataclass(slots=True, frozen=True)
class IntelligenceReport:
    """The final structured intelligence context provided to Scoring Engine."""

    score: IntelligenceScore
    consensus: MarketConsensus
    narratives: tuple[MarketNarrative, ...]
    signals: tuple[SignalEvidence, ...]
    timestamp: int
