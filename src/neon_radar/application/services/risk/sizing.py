"""Position Sizing Engine.

Calculates the optimal position size based on the RiskDecision and the TradeSetup.
Architected using the Strategy pattern to easily swap in Kelly Criterion or Risk Parity later.
"""

import abc
import math
from dataclasses import dataclass

from neon_radar.domain.risk import RiskDecision
from neon_radar.domain.trading.setup import TradeSetup


@dataclass(slots=True, frozen=True)
class SizedTradeSetup:
    """An actionable trade recommendation with final calculated sizes."""

    setup: TradeSetup
    quote_size: float
    base_size: float

    def __post_init__(self) -> None:
        if self.quote_size <= 0 or self.base_size <= 0:
            raise ValueError("Position sizes must be positive")


class PositionSizingStrategy(abc.ABC):
    """Abstract base strategy for position sizing."""

    @abc.abstractmethod
    def calculate_size(self, setup: TradeSetup, decision: RiskDecision) -> float:
        """Calculate the position size in QUOTE asset (e.g., USDT)."""
        pass


class FixedSizeStrategy(PositionSizingStrategy):
    """Allocates a fixed amount of quote asset to every trade, capped by risk budget."""

    def __init__(self, fixed_quote_amount: float) -> None:
        self.fixed_quote_amount = fixed_quote_amount

    def calculate_size(self, setup: TradeSetup, decision: RiskDecision) -> float:
        # Respect the risk manager's max absolute position size
        if decision.max_position_size > 0:
            return min(self.fixed_quote_amount, decision.max_position_size)
        return self.fixed_quote_amount


class FixedRiskStrategy(PositionSizingStrategy):
    """Allocates size based on the distance to stop loss.

    Size = RiskBudget / (Entry - SL)
    """

    def calculate_size(self, setup: TradeSetup, decision: RiskDecision) -> float:
        if not decision.max_risk_budget or decision.max_risk_budget <= 0:
            return 0.0

        sl_distance = abs(setup.entry_price - setup.stop_loss)
        if sl_distance == 0:
            return 0.0

        # We need the size in BASE asset to risk the max_risk_budget:
        # base_size = max_risk_budget / sl_distance
        base_size = decision.max_risk_budget / sl_distance

        # Convert to quote size
        quote_size = base_size * setup.entry_price

        # Cap by the risk manager's maximum absolute position exposure
        if decision.max_position_size > 0:
            quote_size = min(quote_size, decision.max_position_size)

        return quote_size


class ATRBasedStrategy(PositionSizingStrategy):
    """Allocates size using ATR to normalize volatility.

    Similar to FixedRisk, but strictly uses the ATR value from the setup's diagnostics
    to define the 'at-risk' distance, ignoring the explicit Stop Loss if needed.
    """

    def __init__(self, atr_multiplier: float = 1.0) -> None:
        self.atr_multiplier = atr_multiplier

    def calculate_size(self, setup: TradeSetup, decision: RiskDecision) -> float:
        if not decision.max_risk_budget or decision.max_risk_budget <= 0:
            return 0.0

        atr = setup.diagnostics.atr if setup.diagnostics else None
        if not atr or math.isnan(atr) or atr <= 0:
            return 0.0

        risk_distance = atr * self.atr_multiplier
        base_size = decision.max_risk_budget / risk_distance

        quote_size = base_size * setup.entry_price

        if decision.max_position_size > 0:
            quote_size = min(quote_size, decision.max_position_size)

        return quote_size


class PositionSizingEngine:
    def __init__(self, strategy: PositionSizingStrategy) -> None:
        self.strategy = strategy

    def build_sized_setup(
        self, setup: TradeSetup, decision: RiskDecision
    ) -> SizedTradeSetup | None:
        """Applies position sizing logic and returns a SizedTradeSetup.

        Returns None if the sizing calculation results in 0 (e.g., limits reached).
        """
        if not decision.is_allowed:
            return None

        quote_size = self.strategy.calculate_size(setup, decision)

        if quote_size <= 0:
            return None

        base_size = quote_size / setup.entry_price

        return SizedTradeSetup(setup=setup, quote_size=quote_size, base_size=base_size)
