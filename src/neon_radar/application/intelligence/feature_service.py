"""Service for constructing MarketIntelligenceFeatures for Live/CLI mode."""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

from neon_radar.application.intelligence.normalizer import IntelligenceNormalizer
from neon_radar.domain.market_intelligence.features import MarketIntelligenceFeatures
from neon_radar.domain.market_intelligence.history import (
    IntelligenceSignalSeries,
)

if TYPE_CHECKING:
    from neon_radar.application.intelligence.service import MarketIntelligenceService
    from neon_radar.infrastructure.storage.intelligence_store import HistoricalIntelligenceStore


class IntelligenceFeatureService:
    """Constructs MarketIntelligenceFeatures using recent history and live data.

    This ensures the Live/CLI analysis path uses the exact same rolling
    normalization mathematics as the Historical path.
    """

    def __init__(
        self,
        intelligence_service: MarketIntelligenceService,
        historical_store: HistoricalIntelligenceStore | None = None,
    ) -> None:
        self._intelligence_service = intelligence_service
        self._historical_store = historical_store

        self._historical_series_map: dict[str, IntelligenceSignalSeries] = {}
        if self._historical_store is not None:
            for sig_type in ("fear_and_greed", "dvol", "put_call_ratio"):
                series = self._historical_store.load_series(sig_type)
                if series is not None:
                    self._historical_series_map[sig_type] = series

    async def get_features(self) -> MarketIntelligenceFeatures | None:
        """Fetch live signals, merge with history, and compute features."""
        # 1. Fetch live report
        try:
            report = await self._intelligence_service.generate_report()
        except Exception:
            # If the service is disabled or fails, return empty features if we have history
            report = None

        current_time = int(time.time() * 1000)

        # 2. Extract live signals
        live_signals = {}
        if report is not None:
            for evidence in report.signals:
                if evidence.type not in live_signals:
                    live_signals[evidence.type] = evidence

        features = {}

        # 3. Process FNG
        fng_series = self._merge_live("fear_and_greed", live_signals.get("fear_and_greed"))
        if fng_series is not None and not fng_series.is_empty:
            fng_sliced = fng_series.slice_by_availability(current_time)
            features["fng_value"] = IntelligenceNormalizer.extract_raw_value(fng_sliced)
            features["fng_z_score_30d"] = IntelligenceNormalizer.calculate_rolling_z_score(fng_sliced, 30)
            features["fng_percentile_30d"] = IntelligenceNormalizer.calculate_percentile(fng_sliced, 30)

        # 4. Process DVOL
        dvol_series = self._merge_live("dvol", live_signals.get("dvol"))
        if dvol_series is not None and not dvol_series.is_empty:
            dvol_sliced = dvol_series.slice_by_availability(current_time)
            features["dvol_value"] = IntelligenceNormalizer.extract_raw_value(dvol_sliced)
            features["dvol_z_score_30d"] = IntelligenceNormalizer.calculate_rolling_z_score(dvol_sliced, 30)
            features["dvol_percentile_30d"] = IntelligenceNormalizer.calculate_percentile(dvol_sliced, 30)

        # 5. Process PCR (forward only, no rolling history needed right now)
        # We just get the live value if available
        pcr_evidence = live_signals.get("put_call_ratio")
        if pcr_evidence is not None:
            raw_str = pcr_evidence.metadata.get("raw_value")
            if raw_str is not None:
                with contextlib.suppress(ValueError, TypeError):
                    features["pcr_value"] = float(raw_str)

        if not features:
            return None

        return MarketIntelligenceFeatures(**features)

    def _merge_live(self, sig_type: str, live_evidence) -> IntelligenceSignalSeries | None:
        """Merge a live signal evidence into the historical series."""
        base_series = self._historical_series_map.get(sig_type)

        if live_evidence is None:
            return base_series

        # Convert SignalEvidence to IntelligenceObservation
        from neon_radar.domain.market_intelligence.history import IntelligenceObservation
        from neon_radar.domain.market_intelligence.models import IntelligenceSignal

        # We must re-create the IntelligenceSignal from SignalEvidence
        signal = IntelligenceSignal(
            type=live_evidence.type,
            direction=live_evidence.direction,
            strength=live_evidence.strength,
            event_timestamp=live_evidence.timestamp,
            provider_name=live_evidence.source.provider_name,
            provider_type=live_evidence.source.provider_type,
            source_id=live_evidence.source.id,
            reliability=live_evidence.source.reliability,
            metadata=live_evidence.metadata,
        )

        # In live mode, available_at is effectively now.
        obs = IntelligenceObservation(
            signal=signal,
            observation_timestamp=signal.event_timestamp,
            available_at=int(time.time() * 1000)
        )

        if base_series is None:
            return IntelligenceSignalSeries(signal_type=sig_type, items=(obs,))

        # Append to the tuple
        return IntelligenceSignalSeries(
            signal_type=sig_type,
            items=(*base_series.items, obs)
        )
