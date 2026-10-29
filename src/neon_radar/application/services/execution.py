"""Execution engine.

Responsible for taking a FinalTradeSetup and executing it (Paper or Live).
It manages the lifecycle of a trade by processing market ticks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING

from neon_radar.domain.enums import Bias
from neon_radar.domain.execution_costs import (
    BinanceFundingModel,
    BinanceFuturesFeeModel,
    ExecutionCostSummary,
    FeeModel,
    FixedSlippageModel,
    FundingModel,
    SlippageModel,
)
from neon_radar.domain.portfolio import OpenPosition
from neon_radar.domain.trading.execution import ExecutionType
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from neon_radar.application.services.portfolio_engine import PortfolioEngine
    from neon_radar.application.services.risk.manager import RiskManager
    from neon_radar.application.services.trade_backtester import HistoricalFundingProvider
    from neon_radar.domain.models import OHLCV, Symbol
    from neon_radar.domain.portfolio import PortfolioState
    from neon_radar.domain.risk import DrawdownState
    from neon_radar.domain.trading.setup import FinalTradeSetup


logger = get_logger(__name__)

# Stable allocation priority, independent of API responses or mapping iteration.
PAPER_SYMBOL_PRIORITY = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")


def ordered_symbols(symbols: Iterable[Symbol]) -> tuple[Symbol, ...]:
    """Keep allocation priority fixed; unknown symbols follow in lexical order."""
    priority = {symbol: index for index, symbol in enumerate(PAPER_SYMBOL_PRIORITY)}
    return tuple(
        sorted(symbols, key=lambda symbol: (priority.get(str(symbol), len(priority)), str(symbol)))
    )


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

    def __init__(
        self,
        portfolio_engine: PortfolioEngine,
        fee_model: FeeModel | None = None,
        slippage_model: SlippageModel | None = None,
        funding_model: FundingModel | None = None,
        funding_provider: HistoricalFundingProvider | None = None,
        risk_manager: RiskManager | None = None,
        drawdown_provider: Callable[[], DrawdownState] | None = None,
    ) -> None:
        self.portfolio_engine = portfolio_engine
        self.fee_model = fee_model or BinanceFuturesFeeModel()
        self.slippage_model = slippage_model or FixedSlippageModel()
        self.funding_model = funding_model or BinanceFundingModel()
        self.funding_provider = funding_provider
        self.risk_manager = risk_manager
        self.drawdown_provider = drawdown_provider
        # Dictionary of pending setups that are waiting for an entry trigger
        self.pending_setups: dict[str, FinalTradeSetup] = {}
        self.pending_setup_timestamps: dict[str, int] = {}

    @staticmethod
    def _order_position(setup: FinalTradeSetup) -> OpenPosition:
        """Represent committed order size for risk accounting, without booking a fill."""
        quote = setup.quote_size if setup.quote_size > 0 else setup.position_size
        quantity = setup.position_size if setup.quote_size > 0 else quote / setup.entry
        return OpenPosition(
            symbol=setup.symbol,
            direction=setup.direction,
            entry_price=setup.entry,
            quantity=quantity,
            position_size=quote,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            opened_at=0,
        )

    def portfolio_for_risk(self, exclude_symbol: str | None = None) -> PortfolioState:
        """Risk-only view: active positions plus pending reservations; equity is unchanged.

        Reservations are derived from the pending dictionary so recovery and cancellation
        cannot leave a separate reservation ledger out of sync. Never persist this view
        as the actual portfolio or publish its placeholder positions as fills.
        """
        state = self.portfolio_engine.state
        reserved = tuple(
            self._order_position(setup)
            for symbol, setup in self.pending_setups.items()
            if symbol != exclude_symbol
        )
        return replace(state, positions=state.positions + reserved)

    def _risk_allows(self, setup: FinalTradeSetup, exclude_symbol: str | None = None) -> bool:
        if not setup.risk_decision.is_allowed:
            return False
        position = self._order_position(setup)
        if not all(
            isfinite(v)
            for v in (
                position.position_size,
                position.quantity,
                position.entry_price,
                position.stop_loss,
                position.max_risk,
            )
        ):
            return False
        view = self.portfolio_for_risk(exclude_symbol)
        if any(p.symbol == setup.symbol for p in view.positions):
            return False
        reserved_quote = view.total_exposure - self.portfolio_engine.state.total_exposure
        if position.position_size > view.account.free_capital - reserved_quote + 1e-8:
            return False
        if self.risk_manager is None:
            return True  # Legacy standalone/backtest callers retain their policy ownership.
        drawdown = self.drawdown_provider() if self.drawdown_provider else None
        decision = self.risk_manager.evaluate_trade(setup.symbol, setup.direction, view, drawdown)
        return (
            decision.is_allowed
            and position.position_size <= decision.max_position_size + 1e-8
            and position.max_risk <= decision.max_risk_budget + 1e-8
        )

    def cancel_setup(self, symbol: str) -> None:
        """Release an order and its metadata together."""
        self.pending_setups.pop(symbol, None)
        self.pending_setup_timestamps.pop(symbol, None)

    def execute_setup(self, setup: FinalTradeSetup, timestamp: int) -> bool:
        """Register a setup to be triggered when the price hits the entry level.
        In a simple paper trader, we assume market orders execute immediately,
        or limit orders wait for price. We'll simulate immediate execution here
        if it's a market order, or store it as pending if limit.
        For Neon Radar, setups are usually hit on the *next* candle. We will just
        store it as pending.
        """
        symbol_str = str(setup.symbol)
        if symbol_str in self.pending_setups or not self._risk_allows(setup):
            logger.info("[PAPER ORDER REJECTED] %s: risk budget or duplicate order", symbol_str)
            return False
        self.pending_setups[symbol_str] = setup
        self.pending_setup_timestamps[symbol_str] = timestamp
        return True

    def _fill_pending(self, symbol: Symbol, candle: OHLCV) -> tuple[OpenPosition, bool] | None:
        """Convert one reservation into a fill without consulting candle close PnL."""
        symbol_str = str(symbol)
        setup = self.pending_setups.get(symbol_str)
        if setup is None:
            return None
        if setup.direction == Bias.BULLISH and candle.low <= setup.entry:
            fill_price = min(setup.entry, candle.open)
            intrabar = candle.open > setup.entry
        elif setup.direction == Bias.BEARISH and candle.high >= setup.entry:
            fill_price = max(setup.entry, candle.open)
            intrabar = candle.open < setup.entry
        else:
            return None

        reserved = self._order_position(setup)
        setup = replace(
            setup,
            entry=fill_price,
            position_size=reserved.quantity,
            quote_size=reserved.quantity * fill_price,
            risk_amount=abs(fill_price - setup.stop_loss) * reserved.quantity,
            expected_reward=abs(setup.take_profit - fill_price) * reserved.quantity,
        )
        if not self._risk_allows(setup, exclude_symbol=symbol_str):
            logger.info("[PAPER ORDER CANCELLED] %s: risk changed before fill", symbol_str)
            self.cancel_setup(symbol_str)
            return None

        entry_type = ExecutionType.MAKER
        htf_regime = None
        if setup.diagnostics and setup.diagnostics.htf_trend is not None:
            htf_regime = "BEAR" if setup.diagnostics.htf_trend < 0 else "BULL"
        position = OpenPosition(
            symbol=symbol,
            direction=setup.direction,
            entry_price=setup.entry,
            quantity=setup.position_size,
            position_size=setup.quote_size,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            opened_at=candle.open_time,
            capital_at_entry=self.portfolio_engine.state.account.total_capital,
            entry_fee=self.fee_model.calculate_entry_fee(setup.quote_size, entry_type),
            entry_slippage=self.slippage_model.calculate_slippage(
                setup.quote_size,
                symbol,
                entry_type,
                setup.direction,
            ),
            entry_execution_type=entry_type.name,
            diagnostics=setup.diagnostics,
            timeframe=getattr(setup, "timeframe", "1d"),
            signal_time=self.pending_setup_timestamps.get(symbol_str, candle.open_time),
            htf_regime=htf_regime,
        )
        try:
            self.portfolio_engine.open_position(position)
        except ValueError:
            self.cancel_setup(symbol_str)
            return None
        self.cancel_setup(symbol_str)
        return position, intrabar

    @staticmethod
    def _opening_exit(position: OpenPosition, candle: OHLCV) -> tuple[float, str] | None:
        if position.direction == Bias.BULLISH:
            if candle.open <= position.stop_loss:
                return candle.open, "STOP_LOSS"
            if candle.open >= position.take_profit:
                return candle.open, "TAKE_PROFIT"
        else:
            if candle.open >= position.stop_loss:
                return candle.open, "STOP_LOSS"
            if candle.open <= position.take_profit:
                return candle.open, "TAKE_PROFIT"
        return None

    @classmethod
    def _intrabar_exit(
        cls,
        position: OpenPosition,
        candle: OHLCV,
        entered_intrabar: bool,
    ) -> tuple[float, str] | None:
        if not entered_intrabar:
            # Also covers newly filled orders whose protective stop is already
            # breached at the opening gap. Existing gap exits ran before fills.
            opening = cls._opening_exit(position, candle)
            if opening is not None:
                return opening
        if position.direction == Bias.BULLISH:
            if candle.low <= position.stop_loss:
                return position.stop_loss, "STOP_LOSS"
            if candle.high >= position.take_profit and (
                not entered_intrabar or candle.close >= position.take_profit
            ):
                return position.take_profit, "TAKE_PROFIT"
        else:
            if candle.high >= position.stop_loss:
                return position.stop_loss, "STOP_LOSS"
            if candle.low <= position.take_profit and (
                not entered_intrabar or candle.close <= position.take_profit
            ):
                return position.take_profit, "TAKE_PROFIT"
        return None

    def _close_at(self, position: OpenPosition, price: float, reason: str, timestamp: int) -> None:
        exit_type = ExecutionType.TAKER if reason == "STOP_LOSS" else ExecutionType.MAKER
        notional = position.quantity * price
        funding = 0.0
        if self.funding_provider is not None:
            funding = self.funding_model.calculate_funding_cost(
                notional,
                position.symbol,
                position.direction,
                position.opened_at,
                timestamp,
                self.funding_provider,
                quantity=position.quantity,
            )
        direction = 1 if position.direction == Bias.BULLISH else -1
        summary = ExecutionCostSummary(
            entry_fee=position.entry_fee,
            exit_fee=self.fee_model.calculate_exit_fee(notional, exit_type),
            slippage_cost=position.entry_slippage
            + self.slippage_model.calculate_slippage(
                notional,
                position.symbol,
                exit_type,
                position.direction,
            ),
            funding_cost=funding,
            gross_pnl=(price - position.entry_price) * position.quantity * direction,
        )
        self.portfolio_engine.close_position(
            symbol=str(position.symbol),
            exit_price=price,
            closed_at=timestamp,
            execution_summary=summary,
            reason=reason,
        )

    def process_market_batch(
        self,
        candles: Mapping[Symbol, OHLCV],
        *,
        on_equity_update: Callable[[float, int], object] | None = None,
    ) -> tuple[OpenPosition, ...]:
        """Process one synchronized interval across the shared portfolio.

        Common O -> existing gap exits -> opening fills -> intrabar fills ->
        intrabar exits -> common C. Intrabar proceeds never finance earlier fills.
        Ambiguous intrabar exits retain the conservative OHLC rules. The fixed
        symbol priority is an allocation policy, not a reconstruction of tick order.
        Returns all fills, including positions closed within the same batch.
        """
        if not candles:
            return ()
        if len({c.open_time for c in candles.values()}) != 1:
            raise ValueError("Synchronized candles must have the same open_time")
        if len({c.close_time for c in candles.values()}) > 1:
            raise ValueError("Synchronized candles must have the same close_time")
        symbols = ordered_symbols(candles)
        first = candles[symbols[0]]
        closing_time = first.close_time if first.close_time is not None else first.open_time

        def observe(timestamp: int) -> None:
            if on_equity_update is not None:
                on_equity_update(self.portfolio_engine.state.account.total_capital, timestamp)

        # All prices in each snapshot belong to one common point in time.
        self.portfolio_engine.update_market_prices({str(s): candles[s].open for s in symbols})
        observe(first.open_time)
        for symbol in symbols:
            for position in tuple(self.portfolio_engine.state.positions):
                if position.symbol == symbol:
                    exit_info = self._opening_exit(position, candles[symbol])
                    if exit_info is not None:
                        self._close_at(position, *exit_info, first.open_time)
        observe(first.open_time)

        fills: list[OpenPosition] = []
        intrabar_fills: set[Symbol] = set()
        # Opening fills precede uncertain intrabar touches. Reservations for the
        # other symbols remain visible to the risk manager throughout both phases.
        for at_open in (True, False):
            for symbol in symbols:
                setup = self.pending_setups.get(str(symbol))
                if setup is None:
                    continue
                candle = candles[symbol]
                fills_at_open = (
                    candle.open <= setup.entry
                    if setup.direction == Bias.BULLISH
                    else candle.open >= setup.entry
                )
                if fills_at_open != at_open:
                    continue
                filled = self._fill_pending(symbol, candle)
                if filled is not None:
                    position, intrabar = filled
                    fills.append(position)
                    if intrabar:
                        intrabar_fills.add(symbol)
        observe(first.open_time)

        for symbol in symbols:
            for position in tuple(self.portfolio_engine.state.positions):
                if position.symbol == symbol:
                    exit_info = self._intrabar_exit(
                        position, candles[symbol], symbol in intrabar_fills
                    )
                    if exit_info is not None:
                        self._close_at(position, *exit_info, first.open_time)
        self.portfolio_engine.update_market_prices({str(s): candles[s].close for s in symbols})
        observe(closing_time)
        return tuple(fills)

    def process_market_tick(self, symbol: Symbol, candle: OHLCV) -> None:
        """Single-symbol compatibility adapter; multi-asset callers must submit a batch."""
        self.process_market_batch({symbol: candle})
