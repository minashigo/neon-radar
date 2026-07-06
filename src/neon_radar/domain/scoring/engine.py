"""Scoring engine — composes rules and a market state into an :class:`AnalysisResult`.

Two implementations:

* :class:`ScoringEngine` — abstract base.
* :class:`RuleBasedEngine` — evaluates every configured rule in
  order, aggregates the resulting signals.

The engine is deliberately small. Heavy logic lives in the rules and
the aggregator; the engine is glue.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neon_radar.domain.scoring.aggregator import aggregate
from neon_radar.domain.scoring.value_objects import AnalysisResult, Score, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState
    from neon_radar.domain.scoring.factor_rule import FactorRule


class ScoringEngine(ABC):
    """Abstract scoring engine."""

    name: str = ""

    @abstractmethod
    def evaluate(self, state: MarketState) -> AnalysisResult:
        """Score one :class:`MarketState`."""

    def evaluate_signals(self, state: MarketState) -> tuple[Signal, ...]:
        """Return just the raw signals, without aggregation.

        Useful when callers want to apply their own aggregation
        (e.g. custom min_confidence, weighting scheme).
        """
        signals: list[Signal] = []
        for rule in self._rules_iter(state):  # type: ignore[attr-defined]
            try:
                sig = rule.evaluate(state)
            except Exception:
                continue
            if sig is not None:
                signals.append(sig)
        return tuple(signals)

    def _rules_iter(self, state: MarketState):  # type: ignore[no-untyped-def]
        """Hook for engines that store rules differently."""
        raise NotImplementedError


@dataclass
class RuleBasedEngine(ScoringEngine):
    """Scores by evaluating each rule in order and aggregating."""

    name: str = "rule_based"
    rules: tuple[FactorRule, ...] = ()
    min_confidence: float = 0.0

    def evaluate(self, state: MarketState) -> AnalysisResult:
        signals = self.evaluate_signals(state)
        score = aggregate(signals, min_confidence=self.min_confidence)
        summary = summarise(score, signals)
        return AnalysisResult(
            market_state=state,
            score=score,
            signals=signals,
            summary=summary,
            computed_at=int(time.time() * 1000),
        )

    def _rules_iter(self, state: MarketState):  # type: ignore[override]
        return iter(self.rules)


def summarise(score: Score, signals: list[Signal]) -> str:
    """One-line summary for the CLI and UI."""
    if not signals:
        return "no contributing signals"
    bias_word = score.bias.value.lower()
    return (
        f"{bias_word} bias; "
        f"{len(signals)} contributing signals; "
        f"long={score.long_score:.2f} short={score.short_score:.2f}"
    )
