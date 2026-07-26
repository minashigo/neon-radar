"""Domain models for Risk Engine and Portfolio Management.

Design notes
------------
* All models use ``@dataclass(slots=True, frozen=True)`` to maintain immutability.
* Time-based properties use Unix milliseconds for consistency with OHLCV data.
* Position size and capital amounts are represented in the quote asset (e.g. USDT)
  or base asset depending on context.
"""

from __future__ import annotations

from dataclasses import dataclass


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
