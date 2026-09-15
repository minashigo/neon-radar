"""Domain models for Risk Engine and Portfolio Management.

Design notes
------------
* All models use ``@dataclass(slots=True, frozen=True)`` to maintain immutability.
* Time-based properties use Unix milliseconds for consistency with OHLCV data.
* Position size and capital amounts are represented in the quote asset (e.g. USDT)
  or base asset depending on context.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class DrawdownState:
    """Tracks account equity health over time to monitor drawdowns."""

    current_equity: float
    ath_equity: float
    max_drawdown_pct: float
    timestamp: int = 0

    def __post_init__(self) -> None:
        if self.current_equity < 0:
            raise ValueError("current_equity cannot be negative")
        if self.ath_equity < self.current_equity:
            raise ValueError("ath_equity cannot be less than current_equity")
        if self.max_drawdown_pct < 0:
            raise ValueError("max_drawdown_pct cannot be negative")

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown from the All-Time High, represented as a positive percentage (0-100)."""
        if self.ath_equity == 0:
            return 0.0
        return (self.ath_equity - self.current_equity) / self.ath_equity * 100.0


@dataclass(slots=True, frozen=True)
class DrawdownTier:
    """A drawdown threshold and associated risk penalty multiplier.

    Attributes:
        threshold_pct: Drawdown percentage threshold (e.g. 15.0 for 15%).
        risk_multiplier: Multiplier applied to base risk budget (0.0 to 1.0).
        is_trading_halt: If True, blocks opening any new positions.
    """

    threshold_pct: float
    risk_multiplier: float
    is_trading_halt: bool = False

    def __post_init__(self) -> None:
        if self.threshold_pct < 0:
            raise ValueError("threshold_pct cannot be negative")
        if not (0.0 <= self.risk_multiplier <= 1.0):
            raise ValueError("risk_multiplier must be between 0.0 and 1.0")


@dataclass(slots=True, frozen=True)
class ShortRiskConstraints:
    """Optional constraints specific to Short positions.

    Attributes:
        enabled: Whether short-specific constraints are active.
        risk_multiplier: Additional risk budget multiplier for shorts (e.g. 0.5).
        max_concurrent_shorts: Maximum number of simultaneous short positions.
        allow_in_drawdown: If False, blocks new short trades when in a drawdown tier.
    """

    enabled: bool = False
    risk_multiplier: float = 1.0
    max_concurrent_shorts: int | None = None
    allow_in_drawdown: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.risk_multiplier <= 1.0):
            raise ValueError("risk_multiplier must be between 0.0 and 1.0")
        if self.max_concurrent_shorts is not None and self.max_concurrent_shorts < 1:
            raise ValueError("max_concurrent_shorts must be >= 1")


@dataclass(slots=True, frozen=True)
class PortfolioRiskPolicy:
    """Policy for portfolio-level risk limits, heat, and circuit breakers.

    Attributes:
        base_risk_per_trade_pct: Fraction of equity to risk per trade (e.g. 0.02 = 2%).
        max_concurrent_trades: Maximum number of simultaneously open positions.
        max_portfolio_exposure_pct: Maximum aggregate position notional as a fraction of equity.
        enable_circuit_breaker: Whether multi-tier drawdown circuit breaking is enabled.
        drawdown_tiers: Tuple of DrawdownTier thresholds and multipliers.
        legacy_drawdown_threshold_pct: Fallback single DD threshold for backward compatibility.
        legacy_drawdown_penalty_factor: Fallback single DD penalty for backward compatibility.
        enable_portfolio_heat: Whether total portfolio risk budget capping is enabled.
        max_portfolio_heat_pct: Maximum allowed portfolio heat (aggregate risk as % of equity).
        min_trade_risk_budget: Minimum quote risk budget to allow opening a position.
        short_constraints: Optional direction-specific constraints for short positions.
    """

    base_risk_per_trade_pct: float = 0.02
    max_concurrent_trades: int = 3
    max_portfolio_exposure_pct: float = 1.0

    # Drawdown Circuit Breaker
    enable_circuit_breaker: bool = False
    drawdown_tiers: tuple[DrawdownTier, ...] = field(default_factory=tuple)

    # Legacy drawdown penalty (active when enable_circuit_breaker is False, preserves RC9.5 baseline)
    legacy_drawdown_threshold_pct: float = 10.0
    legacy_drawdown_penalty_factor: float = 0.5

    # Portfolio Heat
    enable_portfolio_heat: bool = False
    max_portfolio_heat_pct: float = 0.06
    min_trade_risk_budget: float = 1.0

    # Short constraints
    short_constraints: ShortRiskConstraints = field(default_factory=ShortRiskConstraints)

    def __post_init__(self) -> None:
        if self.base_risk_per_trade_pct <= 0:
            raise ValueError("base_risk_per_trade_pct must be positive")
        if self.max_concurrent_trades < 1:
            raise ValueError("max_concurrent_trades must be >= 1")
        if self.max_portfolio_exposure_pct <= 0:
            raise ValueError("max_portfolio_exposure_pct must be positive")
        if self.max_portfolio_heat_pct <= 0:
            raise ValueError("max_portfolio_heat_pct must be positive")
        if self.min_trade_risk_budget < 0:
            raise ValueError("min_trade_risk_budget cannot be negative")

    @property
    def max_risk_per_trade_pct(self) -> float:
        """Alias for backward compatibility with RiskManagerConfig."""
        return self.base_risk_per_trade_pct

    @property
    def max_open_positions(self) -> int:
        """Alias for backward compatibility with RiskManagerConfig."""
        return self.max_concurrent_trades

    @property
    def drawdown_penalty_threshold_pct(self) -> float:
        """Alias for backward compatibility with RiskManagerConfig."""
        return self.legacy_drawdown_threshold_pct

    @property
    def drawdown_penalty_factor(self) -> float:
        """Alias for backward compatibility with RiskManagerConfig."""
        return self.legacy_drawdown_penalty_factor


@dataclass(slots=True, frozen=True)
class RiskDecision:
    """Output from the RiskManager, dictating allowable risk constraints."""

    is_allowed: bool
    rejection_reason: str = ""
    # The maximum allowable risk budget (in quote asset) for a trade.
    max_risk_budget: float = 0.0
    # The maximum permitted absolute position size (in quote asset).
    max_position_size: float = 0.0
    # Penalty modifier applied to scale down sizing (e.g., during drawdowns). 1.0 = normal.
    risk_penalty_factor: float = 1.0

