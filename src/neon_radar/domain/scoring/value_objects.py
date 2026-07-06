"""Scoring value objects — moved here from the old single ``scoring.py``.

Kept as a separate module so the package layout is consistent:

* :mod:`neon_radar.domain.scoring.value_objects`  — pure data
* :mod:`neon_radar.domain.scoring.factor_rule`    — abstract base
* :mod:`neon_radar.domain.scoring.registry`       — rule registry
* :mod:`neon_radar.domain.scoring.aggregator`     — score math
* :mod:`neon_radar.domain.scoring.engine`         — concrete engines
* :mod:`neon_radar.domain.scoring.rules`          — built-in rules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neon_radar.domain.enums import Bias

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@dataclass(slots=True, frozen=True)
class EvidenceItem:
    """A single (key, value) pair explaining a :class:`Signal`."""

    key: str
    value: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("EvidenceItem.key must not be empty")


@dataclass(slots=True, frozen=True)
class Signal:
    """One factor's contribution to the final score."""

    name: str
    weight: float
    value: float
    confidence: float
    description: str
    evidence: tuple[EvidenceItem, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Signal.weight must be in [0, 1], got {self.weight}")
        if not -1.0 <= self.value <= 1.0:
            raise ValueError(f"Signal.value must be in [-1, 1], got {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Signal.confidence must be in [0, 1], got {self.confidence}"
            )

    @property
    def is_bullish(self) -> bool:
        return self.value > 0

    @property
    def is_bearish(self) -> bool:
        return self.value < 0


@dataclass(slots=True, frozen=True)
class Score:
    """Aggregate score from all contributing signals."""

    value: float
    confidence: float
    long_score: float
    short_score: float
    contributing_signals: int

    def __post_init__(self) -> None:
        if not -1.0 <= self.value <= 1.0:
            raise ValueError(f"Score.value must be in [-1, 1], got {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Score.confidence must be in [0, 1], got {self.confidence}"
            )
        if self.long_score < 0 or self.short_score < 0:
            raise ValueError("long_score and short_score must be non-negative")
        if self.contributing_signals < 0:
            raise ValueError("contributing_signals must be non-negative")

    @property
    def bias(self) -> Bias:
        if self.value > 0.2:
            return Bias.BULLISH
        if self.value < -0.2:
            return Bias.BEARISH
        return Bias.NEUTRAL

    @property
    def magnitude(self) -> float:
        return abs(self.value)


@dataclass(slots=True, frozen=True)
class AnalysisResult:
    """Final output of the scoring engine for one :class:`MarketState`."""

    score: Score
    signals: tuple[Signal, ...]
    summary: str
    computed_at: int
    market_state: MarketState | None = None

    @property
    def bias(self) -> Bias:
        return self.score.bias

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    def signals_by_name(self) -> dict[str, Signal]:
        out: dict[str, Signal] = {}
        for sig in self.signals:
            out.setdefault(sig.name, sig)
        return out

    def breakdown(self) -> tuple[FactorBreakdown, ...]:
        """Per-signal explanation of the score.

        Each entry shows the **signed contribution** (``value * weight``)
        and the underlying value/weight/confidence. Useful for
        debugging and for the ``--explain`` CLI mode / score panel UI.
        """
        return tuple(
            FactorBreakdown(
                factor=sig.name,
                contribution=sig.value * sig.weight,
                weight=sig.weight,
                value=sig.value,
                confidence=sig.confidence,
                description=sig.description,
            )
            for sig in self.signals
        )


@dataclass(slots=True, frozen=True)
class FactorBreakdown:
    """One row of a score breakdown — what one factor contributed."""

    factor: str
    contribution: float  # value * weight (signed)
    weight: float  # factor weight from config
    value: float  # raw factor direction * strength
    confidence: float  # factor's own confidence
    description: str  # human-readable one-liner

    @property
    def is_bullish(self) -> bool:
        return self.contribution > 0

    @property
    def is_bearish(self) -> bool:
        return self.contribution < 0
