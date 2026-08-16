"""Filesystem-based append/merge store for Historical Market Intelligence."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType, SourceReliability
from neon_radar.domain.market_intelligence.history import (
    IntelligenceObservation,
    IntelligenceSignalSeries,
)
from neon_radar.domain.market_intelligence.models import IntelligenceSignal

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _serialize_observation(obs: IntelligenceObservation) -> dict[str, Any]:
    sig = obs.signal
    return {
        "observation_timestamp": obs.observation_timestamp,
        "available_at": obs.available_at,
        "signal": {
            "type": str(sig.type.value),
            "direction": sig.direction,
            "strength": sig.strength,
            "event_timestamp": sig.event_timestamp,
            "ingestion_timestamp": sig.ingestion_timestamp,
            "source_id": sig.source_id,
            "provider_name": sig.provider_name,
            "provider_type": sig.provider_type,
            "reliability": str(sig.reliability.value),
            "weight": sig.weight,
            "metadata": dict(sig.metadata),
        }
    }


def _deserialize_observation(data: dict[str, Any]) -> IntelligenceObservation:
    s_data = data["signal"]
    signal = IntelligenceSignal(
        type=IntelligenceSignalType(s_data["type"]),
        direction=float(s_data["direction"]),
        strength=float(s_data["strength"]),
        event_timestamp=int(s_data["event_timestamp"]),
        ingestion_timestamp=int(s_data["ingestion_timestamp"]),
        source_id=str(s_data["source_id"]),
        provider_name=str(s_data["provider_name"]),
        provider_type=str(s_data["provider_type"]),
        reliability=SourceReliability(s_data["reliability"]),
        weight=float(s_data["weight"]),
        metadata=s_data.get("metadata", {}),
    )
    return IntelligenceObservation(
        signal=signal,
        observation_timestamp=int(data["observation_timestamp"]),
        available_at=int(data["available_at"]),
    )


class HistoricalIntelligenceStore:
    """JSONL-based append/merge storage for intelligence series.

    Guarantees deterministic sorting by `available_at` and deduplication.
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, signal_type: str) -> Path:
        # e.g. "dvol.jsonl"
        return self._dir / f"{signal_type.lower()}.jsonl"

    def load_series(self, signal_type: str) -> IntelligenceSignalSeries | None:
        """Load all historical observations for a signal type."""
        path = self._get_path(signal_type)
        if not path.is_file():
            return None

        observations = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    obs = _deserialize_observation(data)
                    observations.append(obs)
        except Exception as exc:
            logger.error("Failed to read intelligence store for %s: %s", signal_type, exc)
            return None

        if not observations:
            return None

        # Sort strictly by available_at to ensure deterministic behavior
        observations.sort(key=lambda o: o.available_at)

        return IntelligenceSignalSeries(
            signal_type=signal_type,
            items=tuple(observations),
        )

    def append_series(self, signal_type: str, new_observations: list[IntelligenceObservation]) -> None:
        """Merge new observations into the historical file.

        Deduplicates by `available_at` timestamp.
        """
        if not new_observations:
            return

        # Load existing
        existing = self.load_series(signal_type)
        all_obs = list(existing.items) if existing else []

        # Merge
        # Use available_at as the primary unique key for deduplication
        existing_map = {obs.available_at: obs for obs in all_obs}

        changed = False
        for new_obs in new_observations:
            key = new_obs.available_at
            if key not in existing_map:
                existing_map[key] = new_obs
                changed = True
            else:
                # Resolve conflict by keeping the existing one, or overriding if we want?
                # Generally, history shouldn't change. We keep existing.
                pass

        if not changed:
            return

        merged_obs = list(existing_map.values())
        merged_obs.sort(key=lambda o: o.available_at)

        # Write back full file
        path = self._get_path(signal_type)
        tmp = path.with_suffix(".jsonl.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for obs in merged_obs:
                    line = json.dumps(_serialize_observation(obs), ensure_ascii=False)
                    f.write(f"{line}\n")
            tmp.replace(path)
        except Exception as exc:
            logger.error("Failed to write intelligence store for %s: %s", signal_type, exc)
            if tmp.exists():
                tmp.unlink()
