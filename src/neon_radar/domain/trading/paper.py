"""Domain models for Forward Paper Trading.

Captures complete point-in-time telemetry, simulated order lifecycle,
and persistent state snapshots for forward paper execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.enums import Bias
    from neon_radar.domain.models import Symbol
    from neon_radar.domain.portfolio import AccountState, ClosedPosition, OpenPosition
    from neon_radar.domain.risk import DrawdownState, RiskDecision
    from neon_radar.domain.trading.backtest import TradeDiagnostics
    from neon_radar.domain.trading.setup import FinalTradeSetup


@dataclass(slots=True, frozen=True)
class PaperTradeRecord:
    """Full point-in-time telemetry record for a paper trade.

    Fulfills Requirement 5: Stores all signal, risk, execution, and cost attributes.
    """

    signal_timestamp: int
    symbol: Symbol
    timeframe: str
    direction: Bias
    entry_price: float
    simulated_fill_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    score: float
    position_size: float
    quote_size: float
    entry_time: int
    regime: str | None = None
    htf_regime: str | None = None
    risk_decision: RiskDecision | None = None
    exit_timestamp: int | None = None
    exit_price: float | None = None
    exit_reason: str = "OPEN"
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    net_pnl: float = 0.0
    profit_r: float = 0.0
    capital_at_entry: float = 0.0
    diagnostics: TradeDiagnostics | None = None


@dataclass(slots=True, frozen=True)
class PaperTradingState:
    """Immutable snapshot of the complete forward paper trading engine state.

    Enables crash-safe persistence and restart recovery (Requirement 11).
    """

    account: AccountState
    drawdown: DrawdownState
    open_positions: tuple[OpenPosition, ...]
    pending_setups: dict[str, FinalTradeSetup]
    last_processed_candles: dict[str, int]
    completed_trades: tuple[ClosedPosition, ...]
    updated_at: int
