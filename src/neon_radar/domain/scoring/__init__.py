"""Scoring engine — domain layer.

This package contains:

* :mod:`value_objects`  — :class:`Signal`, :class:`Score`, :class:`AnalysisResult`, :class:`FactorBreakdown`
* :mod:`factor_rule`    — abstract :class:`FactorRule`
* :mod:`registry`       — :class:`RuleRegistry`
* :mod:`aggregator`     — :func:`aggregate`
* :mod:`engine`         — :class:`ScoringEngine`, :class:`RuleBasedEngine`
* :mod:`backtest`       — backtest value objects
* :mod:`rules`          — built-in scoring rules

Adding a new rule = one new file under :mod:`rules`, decorated with
``@RuleRegistry.register("name")``. The orchestrator discovers it
automatically.
"""

from neon_radar.domain.scoring.aggregator import aggregate
from neon_radar.domain.scoring.backtest import (
    BacktestConfig,
    BacktestResult,
    ConfidenceCalibration,
    CorrelationMatrix,
    EvaluationResult,
    RuleMetrics,
    SymbolBacktestResult,
)
from neon_radar.domain.scoring.engine import RuleBasedEngine, ScoringEngine
from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry

# Import built-in rules for side-effect registration.
from neon_radar.domain.scoring.rules import (
    BollingerBandsRule,
    CandleBreakoutRule,
    EMATrendRule,
    FundingRateRule,
    MACDMomentumRule,
    RSIMomentumRule,
    VolatilityFilterRule,
    VolumeConfirmationRule,
)
from neon_radar.domain.scoring.value_objects import (
    AnalysisResult,
    EvidenceItem,
    FactorBreakdown,
    Score,
    Signal,
)

__all__ = [
    "AnalysisResult",
    "BacktestConfig",
    "BollingerBandsRule",
    "CandleBreakoutRule",
    "BacktestResult",
    "ConfidenceCalibration",
    "CorrelationMatrix",
    "EMATrendRule",
    "FundingRateRule",
    "EvaluationResult",
    "MACDMomentumRule",
    "EvidenceItem",
    "FactorBreakdown",
    "FactorRule",
    "RSIMomentumRule",
    "RuleBasedEngine",
    "RuleDescription",
    "RuleMetrics",
    "RuleRegistry",
    "Score",
    "ScoringEngine",
    "Signal",
    "SymbolBacktestResult",
    "VolatilityFilterRule",
    "VolumeConfirmationRule",
    "aggregate",
]
