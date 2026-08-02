"""Orchestration service for Market Intelligence."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

from neon_radar.application.intelligence.pipeline import PipelineStep, SignalPipeline
from neon_radar.application.intelligence.registry import provider_registry
from neon_radar.domain.market_intelligence.consensus import ConsensusEngine
from neon_radar.domain.market_intelligence.enums import ConsensusDirection, IntelligenceSignalType
from neon_radar.domain.market_intelligence.models import (
    IntelligenceReport,
    IntelligenceScore,
    IntelligenceSignal,
    PipelineContext,
    SignalEvidence,
    SignalSource,
)
from neon_radar.domain.market_intelligence.narrative import NarrativeEngine
from neon_radar.domain.market_intelligence.noise_filter import NoiseFilter
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neon_radar.application.intelligence.providers import IntelligenceProvider
    from neon_radar.config.intelligence import IntelligenceConfig

logger = get_logger(__name__)


class NoiseFilterStep(PipelineStep):
    """Adapter to wrap NoiseFilter into the signal pipeline."""

    def __init__(self, filter: NoiseFilter) -> None:
        self._filter = filter

    async def process(
        self, context: PipelineContext, signals: Sequence[IntelligenceSignal]
    ) -> tuple[IntelligenceSignal, ...]:
        """Filter noise out of signals."""
        return self._filter.filter_signals(signals)


class MarketIntelligenceService:
    """Orchestrates the gathering and processing of market intelligence."""

    def __init__(
        self,
        config: IntelligenceConfig,
        pipeline: SignalPipeline | None = None,
    ) -> None:
        """Initialize the intelligence service.

        Args:
            config: The intelligence configuration.
            pipeline: Overridable pipeline for custom steps.
        """
        self._config = config

        # Instantiate providers dynamically from registry
        self._providers: list[IntelligenceProvider] = []
        for name in provider_registry.get_registered_names():
            p_config = config.providers.get(name)
            if p_config is not None and p_config.enabled:
                self._providers.append(provider_registry.create_provider(name, p_config))

        noise_filter = NoiseFilter(
            min_reliability_threshold=config.noise_filter.min_reliability_threshold,
            time_window_ms=config.noise_filter.time_window_ms,
            require_independent_confirmation=config.noise_filter.require_independent_confirmation,
        )

        self._pipeline = pipeline or SignalPipeline([NoiseFilterStep(noise_filter)])

        self._consensus = ConsensusEngine(
            bullish_threshold=config.consensus.bullish_threshold,
            bearish_threshold=config.consensus.bearish_threshold,
            conflict_threshold=config.consensus.conflict_threshold,
        )

        self._narrative = NarrativeEngine(
            min_strength_threshold=config.narrative.min_strength_threshold,
            min_evidence_count=config.narrative.min_evidence_count,
        )

    async def generate_report(self) -> IntelligenceReport:
        """Fetch signals, run pipeline, and generate an intelligence report."""
        if not self._config.enabled:
            raise RuntimeError("Market Intelligence is disabled in config.")

        timestamp = int(time.time() * 1000)
        run_id = str(uuid.uuid4())

        active_names = tuple(p.provider_name for p in self._providers)

        if not active_names:
            logger.warning("No active intelligence providers found.")
            return self._build_empty_report(timestamp)

        context = PipelineContext(
            run_id=run_id,
            timestamp=timestamp,
            active_providers=active_names,
        )

        # 1. Fetch from providers
        raw_signals = await self._fetch_all(context)

        if not raw_signals:
            return self._build_empty_report(timestamp)

        # 2. Run pipeline
        pipeline_signals = await self._pipeline.execute(context, raw_signals)

        if not pipeline_signals:
            return self._build_empty_report(timestamp)

        # 3. Convert IntelligenceSignal -> SignalEvidence
        evidence_signals = tuple(
            SignalEvidence(
                type=sig.type,
                direction=sig.direction,
                strength=sig.strength,
                timestamp=sig.event_timestamp,
                source=SignalSource(
                    id=sig.source_id,
                    provider_name=sig.provider_name,
                    provider_type=sig.provider_type,
                    reliability=sig.reliability,
                    weight=sig.weight,
                ),
                metadata=sig.metadata,
            )
            for sig in pipeline_signals
        )

        # 4. Compute Consensus
        consensus = self._consensus.compute_consensus(evidence_signals)

        # 5. Extract Narratives
        narratives = self._narrative.compute_narratives(evidence_signals, timestamp)

        # 6. Build Final Score
        noise_level = (len(raw_signals) - len(pipeline_signals)) / len(raw_signals)

        unique_types = {s.type for s in pipeline_signals}
        coverage = len(unique_types) / len(IntelligenceSignalType)

        if consensus.direction == ConsensusDirection.BULLISH:
            base_value = consensus.confidence
        elif consensus.direction == ConsensusDirection.BEARISH:
            base_value = -consensus.confidence
        else:
            base_value = 0.0

        score = IntelligenceScore(
            value=base_value,
            direction=consensus.direction,
            confidence=consensus.confidence,
            conflict=consensus.conflict_level,
            noise=noise_level,
            coverage=coverage,
        )

        return IntelligenceReport(
            score=score,
            consensus=consensus,
            narratives=narratives,
            signals=evidence_signals,
            timestamp=timestamp,
        )

    async def _fetch_all(self, context: PipelineContext) -> tuple[IntelligenceSignal, ...]:
        """Fetch signals from all active providers concurrently."""
        # Using asyncio.TaskGroup or asyncio.gather for fault tolerance
        # The user mentioned Fault Tolerance using TaskGroup or gather + timeout.
        tasks = [self._fetch_safe(provider, context) for provider in self._providers]
        results = await asyncio.gather(*tasks)

        raw_signals: list[IntelligenceSignal] = []
        for res in results:
            raw_signals.extend(res)

        return tuple(raw_signals)

    async def _fetch_safe(
        self, provider: IntelligenceProvider, context: PipelineContext
    ) -> tuple[IntelligenceSignal, ...]:
        """Fetch signals from a provider, suppressing exceptions."""
        try:
            p_config = self._config.providers.get(provider.provider_name)
            timeout = p_config.timeout_seconds if p_config is not None else 10.0

            async with asyncio.timeout(timeout):
                result = await provider.fetch_signals(context)
                return result.signals
        except TimeoutError:
            logger.warning(
                "Provider %s timed out after %s seconds", provider.provider_name, timeout
            )
            return ()
        except Exception as exc:
            logger.warning("Provider %s failed: %s", provider.provider_name, exc, exc_info=exc)
            return ()

    def _build_empty_report(self, timestamp: int) -> IntelligenceReport:
        from neon_radar.domain.market_intelligence.models import MarketConsensus

        return IntelligenceReport(
            score=IntelligenceScore(
                value=0.0,
                direction=ConsensusDirection.NEUTRAL,
                confidence=0.0,
                conflict=0.0,
                noise=0.0,
                coverage=0.0,
            ),
            consensus=MarketConsensus(
                direction=ConsensusDirection.NEUTRAL,
                confidence=0.0,
                conflict_level=0.0,
            ),
            narratives=(),
            signals=(),
            timestamp=timestamp,
        )
