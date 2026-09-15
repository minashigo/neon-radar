"""Pydantic models for portfolio risk management configuration.

Design notes
------------
* Models use frozen ConfigDict to prevent runtime mutation.
* Defaults are backward-compatible with the RC9.5 baseline:
  - Base risk per trade: 2% (0.02)
  - Max concurrent trades: 3
  - Max portfolio exposure: 100% (1.0)
  - Circuit breaker: disabled by default (falls back to legacy 10% DD / 0.5 factor)
  - Portfolio heat: disabled by default (max 6% when enabled)
  - Short constraints: disabled by default (shorts treated identically to longs)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from neon_radar.domain.risk import (
    DrawdownTier,
    PortfolioRiskPolicy,
    ShortRiskConstraints,
)


class DrawdownTierConfig(BaseModel):
    """Configuration for a single drawdown tier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Drawdown percentage from ATH (e.g. 15.0 for 15%).",
    )
    risk_multiplier: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Multiplier for risk budget (0.0 to 1.0).",
    )
    is_trading_halt: bool = Field(
        default=False,
        description="If True, halts all new trade entries when this tier is reached.",
    )

    def to_domain(self) -> DrawdownTier:
        return DrawdownTier(
            threshold_pct=self.threshold_pct,
            risk_multiplier=self.risk_multiplier,
            is_trading_halt=self.is_trading_halt,
        )


class ShortRiskConfig(BaseModel):
    """Configuration for short-specific risk controls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Whether short-specific constraints are active.",
    )
    risk_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Multiplier for short position risk budget.",
    )
    max_concurrent_shorts: int | None = Field(
        default=None,
        ge=1,
        description="Maximum simultaneous short positions across portfolio.",
    )
    allow_in_drawdown: bool = Field(
        default=True,
        description="Whether to permit short positions when portfolio is in drawdown.",
    )

    def to_domain(self) -> ShortRiskConstraints:
        return ShortRiskConstraints(
            enabled=self.enabled,
            risk_multiplier=self.risk_multiplier,
            max_concurrent_shorts=self.max_concurrent_shorts,
            allow_in_drawdown=self.allow_in_drawdown,
        )


class RiskPolicyConfig(BaseModel):
    """Configuration for portfolio-level risk limits, heat, and circuit breakers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_risk_per_trade_pct: float = Field(
        default=0.02,
        ge=0.0001,
        le=0.5,
        description="Fraction of portfolio equity risked per trade (default 2%).",
    )
    max_concurrent_trades: int = Field(
        default=3,
        ge=1,
        description="Maximum simultaneously open positions across portfolio.",
    )
    max_portfolio_exposure_pct: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum aggregate position notional as a fraction of equity.",
    )

    # Drawdown Circuit Breaker
    enable_circuit_breaker: bool = Field(
        default=False,
        description="Enable multi-tier drawdown circuit breaking.",
    )
    drawdown_tiers: list[DrawdownTierConfig] = Field(
        default_factory=list,
        description="Ordered tiers of drawdown thresholds and risk multipliers.",
    )
    legacy_drawdown_threshold_pct: float = Field(
        default=10.0,
        ge=0.0,
        description="Fallback single threshold for backward compatibility.",
    )
    legacy_drawdown_penalty_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fallback single penalty for backward compatibility.",
    )

    # Portfolio Heat
    enable_portfolio_heat: bool = Field(
        default=False,
        description="Enable maximum aggregate portfolio heat capping.",
    )
    max_portfolio_heat_pct: float = Field(
        default=0.06,
        ge=0.001,
        le=1.0,
        description="Maximum portfolio heat limit (sum of risk as fraction of equity).",
    )
    min_trade_risk_budget: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum quote risk budget required to execute a trade.",
    )

    # Short Constraints
    short_constraints: ShortRiskConfig = Field(
        default_factory=ShortRiskConfig,
        description="Optional constraints for short positions.",
    )

    def to_domain(self) -> PortfolioRiskPolicy:
        return PortfolioRiskPolicy(
            base_risk_per_trade_pct=self.base_risk_per_trade_pct,
            max_concurrent_trades=self.max_concurrent_trades,
            max_portfolio_exposure_pct=self.max_portfolio_exposure_pct,
            enable_circuit_breaker=self.enable_circuit_breaker,
            drawdown_tiers=tuple(t.to_domain() for t in self.drawdown_tiers),
            legacy_drawdown_threshold_pct=self.legacy_drawdown_threshold_pct,
            legacy_drawdown_penalty_factor=self.legacy_drawdown_penalty_factor,
            enable_portfolio_heat=self.enable_portfolio_heat,
            max_portfolio_heat_pct=self.max_portfolio_heat_pct,
            min_trade_risk_budget=self.min_trade_risk_budget,
            short_constraints=self.short_constraints.to_domain(),
        )
