"""Backtest value objects — pure data shapes, no I/O.

These dataclasses capture everything the :class:`WalkForwardBacktester`
produces. They are intentionally framework-free so they can be:

* serialised to JSON for further analysis (``--output json``)
* rendered in the CLI / UI
* unit-tested with hand-crafted fixtures

Metrics
-------
* **Hit rate** — fraction of evaluations where the signal's direction
  matched the actual price move over the forward horizon.
* **Avg return** — arithmetic mean of price changes for evaluations
  bucketed by signal direction (Long / Short / Neutral).
* **Per-rule hit rate** — same as overall hit rate but restricted to
  evaluations where *this rule* voted (non-zero ``value``). Tells us
  which rules are predictive vs. noise.
* **Correlation matrix** — Pearson correlation of per-day signal
  values between rule pairs. High correlation indicates factor
  crowding.
* **Confidence calibration** — bucketed hit rates by confidence. If
  high-confidence predictions are accurate, the bucketed hit rate
  rises monotonically with confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    from neon_radar.domain.models import Symbol


@dataclass(slots=True, frozen=True)
class BacktestConfig:
    """Parameters used to run a backtest — kept for reproducibility."""

    start_date: date
    end_date: date
    timeframe: str  # e.g. "1d"
    symbols: tuple[str, ...]
    horizons: tuple[int, ...] = (1, 3, 7)  # forward days
    min_confidence: float = 0.0  # mirrors ScoringRulesConfig


@dataclass(slots=True, frozen=True)
class EvaluationResult:
    """One (symbol, day) evaluation: signal at T + outcome over horizon."""

    symbol: Symbol
    day: date
    score_value: float  # in [-1, 1]
    confidence: float  # in [0, 1]
    price_at_signal: float
    price_after_horizon: float  # price at T + horizon (or last available)
    horizon_days: int  # 1, 3, 7, ...
    rule_values: tuple[tuple[str, float], ...]  # (rule_name, value) per rule

    @property
    def actual_return_pct(self) -> float:
        """Realised return over the horizon, as a fraction."""
        if self.price_at_signal == 0:
            return 0.0
        return (self.price_after_horizon - self.price_at_signal) / self.price_at_signal

    @property
    def direction(self) -> int:
        """``+1`` if signal bullish, ``-1`` if bearish, ``0`` if neutral."""
        if self.score_value > 0.05:
            return 1
        if self.score_value < -0.05:
            return -1
        return 0

    @property
    def hit(self) -> bool | None:
        """``True`` if direction matched actual return.

        ``None`` if direction was neutral (no claim to evaluate).
        """
        d = self.direction
        if d == 0:
            return None
        return (d > 0 and self.actual_return_pct > 0) or (d < 0 and self.actual_return_pct < 0)


@dataclass(slots=True, frozen=True)
class SymbolBacktestResult:
    """Per-symbol aggregates over the backtest window."""

    symbol: Symbol
    n_evaluations: int
    hit_rate: dict[int, float]  # horizon -> hit rate (0..1)
    avg_return_long: float  # avg % return when Long signal
    avg_return_short: float  # avg % return when Short signal
    avg_return_neutral: float
    n_long: int
    n_short: int
    n_neutral: int


@dataclass(slots=True, frozen=True)
class RuleMetrics:
    """Per-rule aggregates over the backtest window.

    Used to answer: "which rules actually predict price direction?"
    """

    rule_name: str
    n_votes: int  # evaluations where the rule produced non-zero value
    hit_rate_by_horizon: dict[int, float]  # horizon -> hit rate when voted
    avg_abs_value: float  # average |value| over all days (including zero-votes)


@dataclass(slots=True, frozen=True)
class CorrelationMatrix:
    """Pairwise Pearson correlation between per-day signal values."""

    rule_names: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]  # symmetric, diagonal = 1.0

    def get(self, rule_a: str, rule_b: str) -> float | None:
        try:
            i = self.rule_names.index(rule_a)
            j = self.rule_names.index(rule_b)
        except ValueError:
            return None
        return self.matrix[i][j]


@dataclass(slots=True, frozen=True)
class ConfidenceCalibration:
    """Hit rate by confidence bucket — does confidence predict accuracy?"""

    buckets: tuple[tuple[float, float, float], ...]
    """One entry per bucket: (low, high, hit_rate)."""

    @classmethod
    def from_pairs(cls, pairs: list[tuple[float, float, int, int]]) -> ConfidenceCalibration:
        """Build from (low, high, hits, total) tuples."""
        b = tuple((lo, hi, hits / total if total else 0.0) for lo, hi, hits, total in pairs)
        return cls(buckets=b)


@dataclass(slots=True, frozen=True)
class BacktestResult:
    """Complete output of one :class:`WalkForwardBacktester` run."""

    config: BacktestConfig
    n_evaluations: int

    # Per-symbol aggregates
    symbol_results: dict[str, SymbolBacktestResult] = field(default_factory=dict)

    # Overall (across all symbols)
    overall_hit_rate: dict[int, float] = field(default_factory=dict)
    overall_avg_return_long: float = 0.0
    overall_avg_return_short: float = 0.0
    overall_n_long: int = 0
    overall_n_short: int = 0

    # Per-rule metrics
    rule_metrics: dict[str, RuleMetrics] = field(default_factory=dict)

    # Signal correlation matrix
    correlation: CorrelationMatrix | None = None

    # Confidence calibration
    calibration: ConfidenceCalibration | None = None

    # Free-form recommendations produced by heuristics
    recommendations: tuple[str, ...] = ()

    def hit_rate(self, horizon: int) -> float:
        return self.overall_hit_rate.get(horizon, 0.0)

    @property
    def summary(self) -> str:
        """Short, human-readable summary for the backtest outcome."""
        if self.n_evaluations == 0:
            return "No evaluations produced."

        primary_horizon = self.config.horizons[0] if self.config.horizons else 1
        hit_rate = self.hit_rate(primary_horizon)
        return (
            f"{primary_horizon}d hit rate {hit_rate:.1%}; "
            f"long {self.overall_avg_return_long:+.2%} ({self.overall_n_long}); "
            f"short {self.overall_avg_return_short:+.2%} ({self.overall_n_short})"
        )
