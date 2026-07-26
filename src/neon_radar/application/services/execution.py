"""Execution engine.

Responsible for taking a FinalTradeSetup and executing it (Paper or Live).
It manages the lifecycle of a trade by processing market ticks.
"""
from abc import ABC, abstractmethod

from neon_radar.application.services.portfolio_engine import PortfolioEngine
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import OHLCV, Symbol
from neon_radar.domain.portfolio import OpenPosition
from neon_radar.domain.trading.setup import FinalTradeSetup


class ExecutionEngine(ABC):
    """Abstract Base Class for trade execution."""

    @abstractmethod
    def execute_setup(self, setup: FinalTradeSetup, timestamp: int) -> bool:
        """Attempt to execute a trade setup. Returns True if successfully opened."""
        pass

    @abstractmethod
    def process_market_tick(self, symbol: Symbol, candle: OHLCV) -> None:
        """Process a market tick (candle) to check for exits (TP/SL)."""
        pass


class PaperExecutionEngine(ExecutionEngine):
    """Simulates trade execution using paper trading logic."""

    def __init__(self, portfolio_engine: PortfolioEngine) -> None:
        self.portfolio_engine = portfolio_engine
        # Dictionary of pending setups that are waiting for an entry trigger
        self.pending_setups: dict[str, FinalTradeSetup] = {}

    def execute_setup(self, setup: FinalTradeSetup, timestamp: int) -> bool:
        """Register a setup to be triggered when the price hits the entry level.
        In a simple paper trader, we assume market orders execute immediately,
        or limit orders wait for price. We'll simulate immediate execution here
        if it's a market order, or store it as pending if limit.
        For Neon Radar, setups are usually hit on the *next* candle. We will just
        store it as pending.
        """
        if not setup.risk_decision.is_allowed:
            return False

        self.pending_setups[str(setup.symbol)] = setup
        return True

    def process_market_tick(self, symbol: Symbol, candle: OHLCV) -> None:
        """Update portfolio state based on market movement and process exits."""
        symbol_str = str(symbol)

        # 1. Update Portfolio with latest market price (using close price)
        self.portfolio_engine.update_market_prices({symbol_str: candle.close})

        # 2. Check Pending Setups for entry
        setup = self.pending_setups.get(symbol_str)
        if setup is not None:
            if candle.low <= setup.entry <= candle.high:
                # Trigger entry
                try:
                    pos = OpenPosition(
                        symbol=symbol,
                        direction=setup.direction,
                        entry_price=setup.entry,
                        quantity=setup.position_size / setup.entry, # Approximation since quantity usually is base asset
                        position_size=setup.position_size, # Margin
                        stop_loss=setup.stop_loss,
                        take_profit=setup.take_profit,
                        opened_at=candle.open_time
                    )
                    self.portfolio_engine.open_position(pos)
                    del self.pending_setups[symbol_str]
                except ValueError:
                    # E.g. insufficient funds
                    del self.pending_setups[symbol_str]

        # 3. Check Open Positions for exit (SL/TP)
        # Note: we fetch the latest state from PortfolioEngine
        state = self.portfolio_engine.state
        for pos in list(state.positions):
            if str(pos.symbol) != symbol_str:
                continue

            exit_price = None
            reason = None

            if pos.direction == Bias.BULLISH:
                if candle.low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    reason = "STOP_LOSS"
                elif candle.high >= pos.take_profit:
                    exit_price = pos.take_profit
                    reason = "TAKE_PROFIT"
            else:
                if candle.high >= pos.stop_loss:
                    exit_price = pos.stop_loss
                    reason = "STOP_LOSS"
                elif candle.low <= pos.take_profit:
                    exit_price = pos.take_profit
                    reason = "TAKE_PROFIT"

            if exit_price is not None:
                self.portfolio_engine.close_position(
                    symbol=symbol_str,
                    exit_price=exit_price,
                    closed_at=candle.open_time,
                    reason=reason
                )
