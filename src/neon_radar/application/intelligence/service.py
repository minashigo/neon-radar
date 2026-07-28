"""Orchestration service for Market Intelligence."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from neon_radar.domain.market_intelligence.consensus import ConsensusEngine
from neon_radar.domain.market_intelligence.models import (
    IntelligenceReport,
    IntelligenceScore,
    SignalEvidence,
)
from neon_radar.domain.market_intelligence.narrative import NarrativeEngine
from neon_radar.domain.market_intelligence.noise_filter import NoiseFilter
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neon_radar.application.intelligence.providers import IntelligenceProvider
    from neon_radar.config.intelligence import IntelligenceConfig

logger = get_logger(__name__)


class MarketIntelligenceService:
    """Orchestrates the gathering and processing of market intelligence."""

    def __init__(
        self,
        config: IntelligenceConfig,
        providers: Iterable[IntelligenceProvider],
    ) -> None:
        """Initialize the intelligence service.

        Args:
            config: The intelligence configuration.
            providers: A collection of intelligence providers.
        """
        self._config = config
        self._providers = tuple(providers)

        self._filter = NoiseFilter(
            min_reliability_threshold=config.noise_filter.min_reliability_threshold,
            time_window_ms=config.noise_filter.time_window_ms,
            require_independent_confirmation=config.noise_filter.require_independent_confirmation,
        )

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
        """Fetch signals, filter noise, and generate an intelligence report."""
        if not self._config.enabled:
            raise RuntimeError("Market Intelligence is disabled in config.")

        timestamp = int(time.time() * 1000)

        # 1. Fetch from all active providers concurrently
        active_providers = []
        for p in self._providers:
            p_config = self._config.providers.get(p.provider_name)
            if p_config is None or p_config.enabled:
                active_providers.append(p)

        if not active_providers:
            logger.warning("No active intelligence providers found.")
            return self._build_empty_report(timestamp)

        tasks = [
            self._fetch_safe(provider, timestamp)
            for provider in active_providers
        ]
        results = await asyncio.gather(*tasks)

        raw_signals: list[SignalEvidence] = []
        for res in results:
            raw_signals.extend(res)

        if not raw_signals:
            return self._build_empty_report(timestamp)

        # 2. Filter Noise
        filtered_signals = self._filter.filter_signals(raw_signals)

        if not filtered_signals:
            return self._build_empty_report(timestamp)

        # 3. Compute Consensus
        consensus = self._consensus.compute_consensus(filtered_signals)

        # 4. Extract Narratives
        narratives = self._narrative.compute_narratives(filtered_signals, timestamp)

        # 5. Build Final Score
        # noise = (raw_signals - filtered_signals) / raw_signals
        noise_level = (len(raw_signals) - len(filtered_signals)) / len(raw_signals)

        # Coverage = distinct signal types out of total known
        from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType
        unique_types = {s.type for s in filtered_signals}
        coverage = len(unique_types) / len(IntelligenceSignalType)

        # Map consensus to overall value
        if consensus.direction.name == "BULLISH":
            base_value = consensus.confidence
        elif consensus.direction.name == "BEARISH":
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
            signals=filtered_signals,
            timestamp=timestamp,
        )

    async def _fetch_safe(self, provider: IntelligenceProvider, timestamp: int) -> tuple[SignalEvidence, ...]:
        """Fetch signals from a provider, suppressing exceptions."""
        try:
            # We enforce timeout at the service level as a fallback
            provider_cfg = self._config.providers.get(provider.provider_name)
            timeout = provider_cfg.timeout_seconds if provider_cfg is not None else 10.0

            async with asyncio.timeout(timeout):
                return await provider.fetch_signals(timestamp)
        except TimeoutError:
            logger.warning("Provider %s timed out after %s seconds", provider.provider_name, timeout)
            return ()
        except Exception as exc:
            logger.warning("Provider %s failed: %s", provider.provider_name, exc, exc_info=exc)
            return ()

    def _build_empty_report(self, timestamp: int) -> IntelligenceReport:
        from neon_radar.domain.market_intelligence.enums import ConsensusDirection
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
