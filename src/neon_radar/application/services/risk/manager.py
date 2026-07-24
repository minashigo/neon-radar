"""Risk Manager service.

Evaluates an AnalysisResult against the current PortfolioState and DrawdownState
to determine if a trade is permitted and what the maximum risk parameters should be.
"""

from dataclasses import dataclass

from neon_radar.domain.risk import DrawdownState, PortfolioState, RiskDecision
from neon_radar.domain.scoring.value_objects import AnalysisResult


@dataclass(slots=True, frozen=True)
class RiskManagerConfig:
    max_risk_per_trade_pct: float = 0.02  # 2% max risk per trade
    max_open_positions: int = 3
    max_portfolio_exposure_pct: float = 1.0  # 100% of equity (no leverage by default)
    # Drawdown scaling: if drawdown exceeds 10%, we start penalizing risk.
    drawdown_penalty_threshold_pct: float = 10.0
    drawdown_penalty_factor: float = 0.5  # Halve the risk if above threshold


class RiskManager:
    def __init__(self, config: RiskManagerConfig | None = None) -> None:
        self.config = config or RiskManagerConfig()

    def evaluate(
        self,
        analysis: AnalysisResult,
        portfolio: PortfolioState,
        drawdown: DrawdownState | None = None,
    ) -> RiskDecision:
        """Evaluates a potential trade signal against portfolio constraints."""

        # 1. Max Open Positions Check
        if len(portfolio.positions) >= self.config.max_open_positions:
            return RiskDecision(
                is_allowed=False,
                rejection_reason=f"Max open positions reached ({self.config.max_open_positions})",
            )

        # 2. Prevent identical symbol positions (optional, but standard for now)
        symbol = None
        if analysis.market_state and analysis.market_state.primary_series:
            symbol = analysis.market_state.primary_series.symbol

        if symbol:
            for pos in portfolio.positions:
                if pos.symbol == symbol:
                    return RiskDecision(
                        is_allowed=False,
                        rejection_reason=f"Position already open for {symbol}",
                    )

        # 3. Max Portfolio Exposure Check
        total_equity = portfolio.account.total_capital
        max_exposure = total_equity * self.config.max_portfolio_exposure_pct
        current_exposure = portfolio.total_exposure

        if current_exposure >= max_exposure:
            return RiskDecision(
                is_allowed=False,
                rejection_reason="Max portfolio exposure reached",
            )

        # Calculate how much more exposure is allowed
        allowed_exposure = max_exposure - current_exposure

        # Calculate base max risk per trade
        max_risk_budget = total_equity * self.config.max_risk_per_trade_pct

        # Apply Drawdown Penalty if needed
        penalty_factor = 1.0
        if drawdown and drawdown.current_drawdown_pct >= self.config.drawdown_penalty_threshold_pct:
            penalty_factor = self.config.drawdown_penalty_factor

        # Scale down budget based on penalty
        scaled_risk_budget = max_risk_budget * penalty_factor

        return RiskDecision(
            is_allowed=True,
            max_risk_budget=scaled_risk_budget,
            max_position_size=allowed_exposure,
            risk_penalty_factor=penalty_factor,
        )
