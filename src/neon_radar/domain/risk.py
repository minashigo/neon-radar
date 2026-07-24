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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.enums import Bias
    from neon_radar.domain.models import Symbol


@dataclass(slots=True, frozen=True)
class AccountState:
    """Represents the basic capital state of a trading account."""

    total_capital: float
    free_capital: float
    currency: str = "USDT"

    def __post_init__(self) -> None:
        if self.total_capital < 0:
            raise ValueError("total_capital cannot be negative")
        if self.free_capital < 0:
            raise ValueError("free_capital cannot be negative")
        if self.free_capital > self.total_capital:
            raise ValueError("free_capital cannot exceed total_capital")


@dataclass(slots=True, frozen=True)
class PositionState:
    """Represents a single open position."""

    symbol: Symbol
    side: Bias
    entry_price: float
    size: float  # Base asset amount (e.g., amount of BTC)
    stop_loss: float | None = None
    unrealized_pnl: float = 0.0

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError("entry_price must be > 0")
        if self.size <= 0:
            raise ValueError("size must be > 0")

    @property
    def quote_size(self) -> float:
        """The total position value in quote asset (Entry Price * Base Size)."""
        return self.entry_price * self.size

    @property
    def max_risk(self) -> float | None:
        """Calculates the max risk in quote asset if stop loss is hit.

        Returns None if no stop loss is defined.
        """
        if self.stop_loss is None:
            return None
        # Max risk is the absolute distance * size
        return abs(self.entry_price - self.stop_loss) * self.size


@dataclass(slots=True, frozen=True)
class PortfolioState:
    """Aggregates the AccountState and active positions at a point in time."""

    account: AccountState
    positions: tuple[PositionState, ...] = field(default_factory=tuple)
    timestamp: int = 0

    @property
    def total_exposure(self) -> float:
        """Total active exposure in quote asset."""
        return sum(pos.quote_size for pos in self.positions)

    @property
    def total_risk(self) -> float | None:
        """Total capital at risk (if all stop losses are hit).

        Returns None if any position is missing a stop loss.
        """
        total = 0.0
        for pos in self.positions:
            r = pos.max_risk
            if r is None:
                return None
            total += r
        return total


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
