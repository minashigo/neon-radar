"""Risk Manager service.

Evaluates an AnalysisResult against the current PortfolioState and DrawdownState
to determine if a trade is permitted and what the maximum risk parameters should be.
"""

from dataclasses import dataclass

from neon_radar.domain.enums import Bias
from neon_radar.domain.portfolio import PortfolioState
from neon_radar.domain.risk import DrawdownState, PortfolioRiskPolicy, RiskDecision
from neon_radar.domain.scoring.value_objects import AnalysisResult


@dataclass(slots=True, frozen=True)
class RiskManagerConfig:
    """Legacy configuration object for backward compatibility."""

    max_risk_per_trade_pct: float = 0.02  # 2% max risk per trade
    max_open_positions: int = 3
    max_portfolio_exposure_pct: float = 1.0  # 100% of equity (no leverage by default)
    drawdown_penalty_threshold_pct: float = 10.0
    drawdown_penalty_factor: float = 0.5  # Halve the risk if above threshold


class RiskManager:
    """Evaluates trade signals against portfolio constraints, heat limits, and circuit breakers."""

    def __init__(
        self, policy_or_config: PortfolioRiskPolicy | RiskManagerConfig | None = None
    ) -> None:
        if policy_or_config is None:
            self.policy = PortfolioRiskPolicy()
        elif isinstance(policy_or_config, RiskManagerConfig):
            self.policy = PortfolioRiskPolicy(
                base_risk_per_trade_pct=policy_or_config.max_risk_per_trade_pct,
                max_concurrent_trades=policy_or_config.max_open_positions,
                max_portfolio_exposure_pct=policy_or_config.max_portfolio_exposure_pct,
                legacy_drawdown_threshold_pct=policy_or_config.drawdown_penalty_threshold_pct,
                legacy_drawdown_penalty_factor=policy_or_config.drawdown_penalty_factor,
            )
        else:
            self.policy = policy_or_config

    @property
    def config(self) -> PortfolioRiskPolicy:
        """Alias for backward compatibility with existing code accessing rm.config."""
        return self.policy

    @config.setter
    def config(self, value: PortfolioRiskPolicy | RiskManagerConfig) -> None:
        if isinstance(value, RiskManagerConfig):
            self.policy = PortfolioRiskPolicy(
                base_risk_per_trade_pct=value.max_risk_per_trade_pct,
                max_concurrent_trades=value.max_open_positions,
                max_portfolio_exposure_pct=value.max_portfolio_exposure_pct,
                legacy_drawdown_threshold_pct=value.drawdown_penalty_threshold_pct,
                legacy_drawdown_penalty_factor=value.drawdown_penalty_factor,
            )
        else:
            self.policy = value

    def evaluate(
        self,
        analysis: AnalysisResult,
        portfolio: PortfolioState | None,
        drawdown: DrawdownState | None = None,
    ) -> RiskDecision:
        """Evaluates a potential trade signal against portfolio constraints."""

        # 1. Portfolio State Validity Check
        if portfolio is None:
            return RiskDecision(
                is_allowed=False,
                rejection_reason="Missing portfolio state",
            )

        if not hasattr(portfolio, "account") or portfolio.account is None or portfolio.account.total_capital <= 0:
            return RiskDecision(
                is_allowed=False,
                rejection_reason="Invalid account capital",
            )

        if not hasattr(portfolio, "positions") or portfolio.positions is None:
            return RiskDecision(
                is_allowed=False,
                rejection_reason="Invalid portfolio positions",
            )

        # 2. Prevent identical symbol positions
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

        # 3. Max Concurrent Trades Check
        if len(portfolio.positions) >= self.policy.max_concurrent_trades:
            return RiskDecision(
                is_allowed=False,
                rejection_reason=f"Max open positions reached ({self.policy.max_concurrent_trades})",
            )

        # 4. Optional Short Constraints
        is_short = False
        if analysis.score is not None:
            is_short = (analysis.score.bias == Bias.BEARISH)

        short_cfg = self.policy.short_constraints
        if is_short and short_cfg.enabled:
            # Check maximum concurrent shorts
            if short_cfg.max_concurrent_shorts is not None:
                current_shorts = sum(1 for p in portfolio.positions if p.direction == Bias.BEARISH)
                if current_shorts >= short_cfg.max_concurrent_shorts:
                    return RiskDecision(
                        is_allowed=False,
                        rejection_reason=f"Max concurrent shorts reached ({short_cfg.max_concurrent_shorts})",
                    )

            # Check drawdown restriction for shorts
            if not short_cfg.allow_in_drawdown and drawdown is not None:
                is_in_dd = False
                if self.policy.enable_circuit_breaker and self.policy.drawdown_tiers:
                    min_thresh = min(t.threshold_pct for t in self.policy.drawdown_tiers)
                    is_in_dd = (drawdown.current_drawdown_pct >= min_thresh)
                else:
                    is_in_dd = (drawdown.current_drawdown_pct >= self.policy.legacy_drawdown_threshold_pct)

                if is_in_dd:
                    return RiskDecision(
                        is_allowed=False,
                        rejection_reason="Short positions disabled during drawdown",
                    )

        # 5. Max Portfolio Exposure Check (Position ceiling)
        total_equity = portfolio.account.total_capital
        max_exposure = total_equity * self.policy.max_portfolio_exposure_pct
        current_exposure = portfolio.total_exposure

        if current_exposure >= max_exposure:
            return RiskDecision(
                is_allowed=False,
                rejection_reason="Max portfolio exposure reached",
            )

        allowed_exposure = max_exposure - current_exposure

        # 6. Drawdown Circuit Breaker Evaluation
        penalty_factor = 1.0
        if self.policy.enable_circuit_breaker and self.policy.drawdown_tiers:
            if drawdown is not None:
                current_dd = drawdown.current_drawdown_pct
                sorted_tiers = sorted(
                    self.policy.drawdown_tiers, key=lambda t: t.threshold_pct, reverse=True
                )
                for tier in sorted_tiers:
                    if current_dd >= tier.threshold_pct:
                        if tier.is_trading_halt or tier.risk_multiplier <= 0.0:
                            return RiskDecision(
                                is_allowed=False,
                                rejection_reason=(
                                    f"Trading halted by drawdown circuit breaker "
                                    f"({current_dd:.2f}% >= {tier.threshold_pct:.2f}%)"
                                ),
                                risk_penalty_factor=0.0,
                            )
                        penalty_factor = tier.risk_multiplier
                        break
        else:
            # Legacy drawdown penalty (preserves exact baseline RC9.5 behavior)
            if drawdown and drawdown.current_drawdown_pct >= self.policy.legacy_drawdown_threshold_pct:
                penalty_factor = self.policy.legacy_drawdown_penalty_factor

        # 7. Calculate Risk Budget
        risk_budget = total_equity * self.policy.base_risk_per_trade_pct * penalty_factor
        if is_short and short_cfg.enabled:
            risk_budget *= short_cfg.risk_multiplier

        # 8. Portfolio Heat Guard Check
        if self.policy.enable_portfolio_heat:
            max_heat = total_equity * self.policy.max_portfolio_heat_pct
            current_heat = portfolio.total_risk
            remaining_heat = max_heat - current_heat

            if remaining_heat <= 0:
                return RiskDecision(
                    is_allowed=False,
                    rejection_reason="Max portfolio heat reached",
                )

            if remaining_heat < risk_budget:
                if remaining_heat >= self.policy.min_trade_risk_budget:
                    risk_budget = remaining_heat
                else:
                    return RiskDecision(
                        is_allowed=False,
                        rejection_reason=(
                            f"Insufficient portfolio heat budget "
                            f"({remaining_heat:.2f} < {self.policy.min_trade_risk_budget:.2f})"
                        ),
                    )

        return RiskDecision(
            is_allowed=True,
            max_risk_budget=risk_budget,
            max_position_size=allowed_exposure,
            risk_penalty_factor=penalty_factor,
        )

