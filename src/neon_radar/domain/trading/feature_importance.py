"""Domain models for Feature Importance (Ablation Analysis)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.trading.backtest import BacktestReport


@dataclass(slots=True, frozen=True)
class FeatureImportanceMetrics:
    """Metrics representing the contribution of a single feature (rule).
    
    Deltas are calculated such that a POSITIVE value indicates the rule is helpful
    (i.e., removing the rule made the metric worse).
    """

    rule_name: str
    delta_profit_factor: float
    delta_expectancy: float
    delta_sharpe_ratio: float
    delta_win_rate: float
    delta_probability_of_loss: float
    delta_p_value: float

    feature_score: float  # Composite integral score

    @property
    def rating_symbols(self) -> str:
        """Return a string of '+' or '-' based on the feature score."""
        if self.feature_score < -0.1:
            return "-"
        elif self.feature_score < 0.1:
            return ""
        elif self.feature_score < 0.3:
            return "+"
        elif self.feature_score < 0.6:
            return "++"
        elif self.feature_score < 1.0:
            return "+++"
        elif self.feature_score < 1.5:
            return "++++"
        else:
            return "+++++"


@dataclass(slots=True, frozen=True)
class FeatureImportanceReport:
    """The result of a full Ablation Analysis."""

    baseline: BacktestReport
    features: tuple[FeatureImportanceMetrics, ...]
