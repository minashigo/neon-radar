"""Market Regime Classification and Filtering models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from neon_radar.application.services.indicator_pipeline import IndicatorSpec
    from neon_radar.domain.market_state import MarketState


class MarketRegime(str, Enum):
    """Broad classification of the market environment."""

    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    CHOP = "CHOP"
    VOLATILE_CRASH = "VOLATILE_CRASH"
    UNKNOWN = "UNKNOWN"


class RegimeFilterConfig(BaseModel):
    """Configuration for rule-based regime classification."""

    enabled: bool = Field(
        default=True, 
        description="Whether to evaluate the regime filter at all."
    )
    
    adx_period: int = Field(default=14)
    adx_chop_threshold: float = Field(
        default=20.0,
        description="ADX values below this are classified as CHOP.",
    )
    
    ema_fast_period: int = Field(default=9)
    ema_slow_period: int = Field(default=21)
    
    atr_period: int = Field(default=14)
    atr_crash_threshold_pct: float = Field(
        default=0.08,  # 8% of price
        description="If ATR/Price > threshold, we classify as VOLATILE_CRASH.",
    )
    
    allowed_long_regimes: set[MarketRegime] = Field(
        default_factory=lambda: {MarketRegime.BULL_TREND, MarketRegime.UNKNOWN},
        description="Regimes where LONG trades are permitted.",
    )
    
    allowed_short_regimes: set[MarketRegime] = Field(
        default_factory=lambda: {MarketRegime.BEAR_TREND, MarketRegime.UNKNOWN},
        description="Regimes where SHORT trades are permitted.",
    )


@dataclass(slots=True, frozen=True)
class RegimeClassification:
    """Result of classifying the market regime."""

    regime: MarketRegime
    reason: str


class RegimeClassifier(Protocol):
    """Protocol for components that can classify market state into regimes."""

    def required_indicators(self) -> tuple[IndicatorSpec, ...]:
        """Indicators required by this classifier to evaluate the state."""
        ...

    def classify(self, state: MarketState) -> RegimeClassification:
        """Evaluate the current market state and classify the regime."""
        ...
