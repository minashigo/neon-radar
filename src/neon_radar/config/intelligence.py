"""Configuration for the Market Intelligence layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for an individual intelligence provider."""

    enabled: bool = True
    priority: int = 1
    timeout_seconds: float = 10.0
    options: dict[str, Any] = Field(default_factory=dict)


class NoiseFilterConfig(BaseModel):
    """Configuration for the noise filter."""

    min_reliability_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    time_window_ms: int = Field(default=3600000, ge=0)
    require_independent_confirmation: bool = True


class ConsensusConfig(BaseModel):
    """Configuration for the consensus engine."""

    bullish_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    bearish_threshold: float = Field(default=-0.3, ge=-1.0, le=0.0)
    conflict_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class NarrativeConfig(BaseModel):
    """Configuration for the narrative engine."""

    min_strength_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    min_evidence_count: int = Field(default=2, ge=1)


class IntelligenceConfig(BaseModel):
    """Main configuration for the Intelligence Service."""

    enabled: bool = True
    cache_ttl_seconds: int = Field(default=300, ge=0)

    noise_filter: NoiseFilterConfig = Field(default_factory=NoiseFilterConfig)
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)
    narrative: NarrativeConfig = Field(default_factory=NarrativeConfig)

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
