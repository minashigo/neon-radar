"""Built-in scoring rules.

Each rule is registered with :class:`RuleRegistry` on import. The
order of import determines display order in ``neon-radar --list-rules``.

To add a new rule:

1. Create ``<name>.py`` in this directory.
2. Subclass :class:`FactorRule`.
3. Decorate with :meth:`RuleRegistry.register`.
4. Add the import to this file.
"""

from neon_radar.domain.scoring.rules.bollinger_bands import BollingerBandsRule
from neon_radar.domain.scoring.rules.candle_breakout import CandleBreakoutRule
from neon_radar.domain.scoring.rules.context_flow import (
    LiquidationCascadeRule,
    LongShortCrowdedRule,
    TakerFlowImbalanceRule,
)
from neon_radar.domain.scoring.rules.context_funding import FundingExtremeRule, FundingTrendRule
from neon_radar.domain.scoring.rules.context_oi import (
    OpenInterestDivergenceRule,
    OpenInterestExpansionRule,
)
from neon_radar.domain.scoring.rules.ema_trend import EMATrendRule
from neon_radar.domain.scoring.rules.funding_rate import FundingRateRule
from neon_radar.domain.scoring.rules.higher_tf_trend import HigherTimeframeTrendRule
from neon_radar.domain.scoring.rules.macd_momentum import MACDMomentumRule
from neon_radar.domain.scoring.rules.rsi_momentum import RSIMomentumRule
from neon_radar.domain.scoring.rules.sma_trend import SMATrendRule
from neon_radar.domain.scoring.rules.volatility_filter import VolatilityFilterRule
from neon_radar.domain.scoring.rules.volume_confirmation import VolumeConfirmationRule

__all__ = [
    "BollingerBandsRule",
    "CandleBreakoutRule",
    "EMATrendRule",
    "FundingExtremeRule",
    "FundingRateRule",
    "FundingTrendRule",
    "HigherTimeframeTrendRule",
    "LiquidationCascadeRule",
    "LongShortCrowdedRule",
    "MACDMomentumRule",
    "OpenInterestDivergenceRule",
    "OpenInterestExpansionRule",
    "RSIMomentumRule",
    "SMATrendRule",
    "TakerFlowImbalanceRule",
    "VolatilityFilterRule",
    "VolumeConfirmationRule",
]
