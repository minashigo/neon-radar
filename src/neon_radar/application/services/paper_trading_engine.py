"""Forward Paper Trading Orchestration Engine.

Manages real-time paper trading against live Binance Futures market data:
- Strictly executes signal pipeline only on closed candles (point-in-time integrity).
- Maintains a single, shared multi-asset PortfolioState across all canonical symbols.
- Simulates realistic order fills, stop-loss, take-profit, fees, slippage, and funding costs.
- Idempotently persists and recovers portfolio, orders, and processed candle timestamps.
- Zero real order placement API calls under all conditions.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neon_radar.application.services.execution import PaperExecutionEngine
from neon_radar.application.services.portfolio_engine import PortfolioEngine
from neon_radar.application.services.risk.drawdown import DrawdownMonitor
from neon_radar.config.models import TimeFrame
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.risk import DrawdownState
from neon_radar.domain.trading.paper import PaperTradingState
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from neon_radar.application.services.trading_pipeline import TradingPipeline
    from neon_radar.domain.portfolio import ClosedPosition, OpenPosition
    from neon_radar.domain.trading.setup import FinalTradeSetup
    from neon_radar.infrastructure.exchanges.base import ExchangeClient
    from neon_radar.infrastructure.storage.paper_state_store import JsonPaperStateStore

logger = get_logger(__name__)

CANONICAL_SYMBOLS = (
    Symbol("BTCUSDT"),
    Symbol("ETHUSDT"),
    Symbol("SOLUSDT"),
    Symbol("BNBUSDT"),
    Symbol("XRPUSDT"),
    Symbol("ADAUSDT"),
)


@dataclass(slots=True, frozen=True)
class PaperCycleReport:
    """Telemetry report produced by each polling cycle of the paper trading engine."""

    timestamp: int
    symbols_evaluated: tuple[Symbol, ...]
    new_setups: tuple[FinalTradeSetup, ...]
    new_fills: tuple[OpenPosition, ...]
    new_exits: tuple[ClosedPosition, ...]
    equity: float
    free_capital: float
    drawdown_pct: float
    warnings: tuple[str, ...]


class ForwardPaperTradingEngine:
    """Forward paper trading orchestrator operating across multiple assets."""

    def __init__(
        self,
        exchange: ExchangeClient,
        pipeline: TradingPipeline,
        portfolio_engine: PortfolioEngine | None = None,
        drawdown_monitor: DrawdownMonitor | None = None,
        execution_engine: PaperExecutionEngine | None = None,
        state_store: JsonPaperStateStore | None = None,
        symbols: tuple[Symbol, ...] = CANONICAL_SYMBOLS,
        timeframe: TimeFrame = TimeFrame.D1,
        higher_timeframe: TimeFrame | None = None,
        max_stale_tolerance_seconds: int = 86400 * 2,
        initial_capital: float = 10000.0,
        on_signal: Callable[[FinalTradeSetup], None] | None = None,
        on_trade_open: Callable[[OpenPosition], None] | None = None,
        on_trade_close: Callable[[ClosedPosition], None] | None = None,
        on_portfolio_update: Callable[[Any], None] | None = None,
    ) -> None:
        self.exchange = exchange
        self.pipeline = pipeline
        self.symbols = tuple(symbols)
        self.timeframe = timeframe
        self.higher_timeframe = higher_timeframe or timeframe.higher_timeframe
        self.max_stale_tolerance_ms = max_stale_tolerance_seconds * 1000
        self.state_store = state_store

        # Single shared multi-asset portfolio and risk monitors
        self.portfolio_engine = portfolio_engine or PortfolioEngine(initial_capital=initial_capital)
        self.drawdown_monitor = drawdown_monitor or DrawdownMonitor(initial_capital=initial_capital)
        self.execution_engine = execution_engine or PaperExecutionEngine(
            portfolio_engine=self.portfolio_engine
        )

        # Idempotency tracking: last processed candle open_time per symbol
        self.last_processed_candles: dict[str, int] = {}

        # Observers / Callbacks
        self.on_signal = on_signal
        self.on_trade_open = on_trade_open
        self.on_trade_close = on_trade_close
        self.on_portfolio_update = on_portfolio_update

        # Wire PortfolioEngine event publishers
        self.portfolio_engine.subscribe(self._on_portfolio_event)

        # Auto-restore if state store is provided and state exists
        if self.state_store is not None:
            self.load_state()

    # ------------------------------------------------------------------
    # Closed-candle & Stale Data Logic (Requirements 2, 7, 8)
    # ------------------------------------------------------------------

    def is_candle_closed(self, candle: OHLCV, server_time: int) -> bool:
        """Determines if a candle is completely closed using exchange server time."""
        if candle.close_time is not None:
            return server_time > candle.close_time
        return server_time >= candle.open_time + (self.timeframe.seconds * 1000)

    def is_candle_stale(self, candle: OHLCV, server_time: int) -> bool:
        """Checks if a candle is older than the configured stale tolerance."""
        close_t = candle.close_time or (candle.open_time + self.timeframe.seconds * 1000)
        return (server_time - close_t) > self.max_stale_tolerance_ms

    # ------------------------------------------------------------------
    # Polling & Execution Lifecycle (Requirements 2, 3, 4, 9, 10, 11)
    # ------------------------------------------------------------------

    async def poll_cycle(self) -> PaperCycleReport:
        """Executes one polling cycle across all configured assets."""
        warnings: list[str] = []
        new_setups: list[FinalTradeSetup] = []
        evaluated_symbols: list[Symbol] = []

        # 1. Authoritative exchange server time
        try:
            server_time = await self.exchange.get_server_time()
        except Exception as exc:
            msg = f"Failed to retrieve exchange server time: {exc}"
            logger.warning(msg)
            warnings.append(msg)
            import time

            server_time = int(time.time() * 1000)

        initial_open_count = len(self.portfolio_engine.state.positions)
        initial_closed_count = len(self.portfolio_engine.history)

        for symbol in self.symbols:
            symbol_str = str(symbol)
            try:
                # 2. Fetch base series
                series = await self.exchange.get_klines(symbol, self.timeframe, limit=200)
                if not series or len(series.candles) == 0:
                    warnings.append(f"{symbol_str}: Empty klines returned")
                    continue

                # 3. Filter only strictly closed candles
                closed_candles = [c for c in series.candles if self.is_candle_closed(c, server_time)]
                if not closed_candles:
                    # All candles in batch are forming / unclosed
                    continue

                latest_closed_candle = closed_candles[-1]

                # 4. Check for stale data (Requirement 7)
                if self.is_candle_stale(latest_closed_candle, server_time):
                    msg = f"{symbol_str}: Stale market data (candle closed at {latest_closed_candle.close_time}, server time {server_time})"
                    logger.warning(msg)
                    warnings.append(msg)
                    continue

                # 5. Check idempotency: avoid reprocessing the same closed candle
                last_processed = self.last_processed_candles.get(symbol_str, 0)
                is_new_candle = latest_closed_candle.open_time > last_processed

                if not is_new_candle:
                    # Already processed this candle for signals and ticks
                    continue

                evaluated_symbols.append(symbol)

                # 6. Process tick on execution engine with the newly closed candle:
                #    Checks pending setups for entry fills & open positions for SL/TP exits.
                self.execution_engine.process_market_tick(symbol, latest_closed_candle)

                # 7. Update Drawdown Monitor with updated equity
                current_equity = self.portfolio_engine.state.account.total_capital
                drawdown_state = self.drawdown_monitor.update(current_equity, latest_closed_candle.open_time)

                # 8. Point-in-Time History for Signal Pipeline:
                #    Slice strictly up to the latest closed candle.
                closed_history = KlineSeries(
                    symbol=symbol,
                    timeframe=self.timeframe.value,
                    candles=tuple(closed_candles),
                )

                # 9. Higher-Timeframe Data (if configured)
                htf_series: KlineSeries | None = None
                if self.higher_timeframe is not None:
                    try:
                        raw_htf = await self.exchange.get_klines(symbol, self.higher_timeframe, limit=100)
                        if raw_htf and len(raw_htf.candles) > 0:
                            candle_close_time = latest_closed_candle.close_time or (
                                latest_closed_candle.open_time + self.timeframe.seconds * 1000
                            )
                            # Slice HTF strictly to point-in-time
                            htf_closed = [
                                c
                                for c in raw_htf.candles
                                if (c.close_time or (c.open_time + self.higher_timeframe.seconds * 1000))
                                <= candle_close_time
                            ]
                            if htf_closed:
                                htf_series = KlineSeries(
                                    symbol=symbol,
                                    timeframe=self.higher_timeframe.value,
                                    candles=tuple(htf_closed),
                                )
                    except Exception as htf_exc:
                        logger.warning(f"{symbol_str}: Failed to fetch HTF klines: {htf_exc}")

                # 10. Optional Funding Rate
                funding_rate = None
                with contextlib.suppress(Exception):
                    funding_rate = await self.exchange.get_funding_rate(symbol)

                # 11. Run Trading Pipeline
                setup = self.pipeline.evaluate(
                    series=closed_history,
                    portfolio=self.portfolio_engine.state,
                    drawdown=drawdown_state,
                    timestamp=latest_closed_candle.open_time,
                    higher_tf_series=htf_series,
                    funding_rate=funding_rate,
                )

                if setup is not None:
                    # Register pending paper setup for next candle execution
                    registered = self.execution_engine.execute_setup(
                        setup, timestamp=latest_closed_candle.open_time
                    )
                    if registered:
                        new_setups.append(setup)
                        logger.info(
                            f"[PAPER ORDER REGISTERED] {setup.symbol} {setup.direction.value} "
                            f"Entry: {setup.entry} SL: {setup.stop_loss} TP: {setup.take_profit} "
                            f"Size: {setup.position_size:.4f} (${setup.quote_size:.2f})"
                        )
                        if self.on_signal:
                            self.on_signal(setup)

                # 12. Mark candle processed (idempotency key)
                self.last_processed_candles[symbol_str] = latest_closed_candle.open_time

            except Exception as sym_exc:
                msg = f"{symbol_str}: Cycle error: {sym_exc}"
                logger.error(msg, exc_info=True)
                warnings.append(msg)

        # 13. Detect newly filled or exited positions during this cycle
        new_fills = tuple(self.portfolio_engine.state.positions[initial_open_count:])
        new_exits = tuple(self.portfolio_engine.history[initial_closed_count:])

        # 14. Atomic Persistence
        if self.state_store is not None:
            try:
                self.save_state(server_time)
            except Exception as save_exc:
                msg = f"Failed to persist paper trading state: {save_exc}"
                logger.error(msg, exc_info=True)
                warnings.append(msg)

        # 15. Create Report
        state = self.portfolio_engine.state
        report = PaperCycleReport(
            timestamp=server_time,
            symbols_evaluated=tuple(evaluated_symbols),
            new_setups=tuple(new_setups),
            new_fills=new_fills,
            new_exits=new_exits,
            equity=state.account.total_capital,
            free_capital=state.account.free_capital,
            drawdown_pct=self.drawdown_monitor.max_drawdown_pct,
            warnings=tuple(warnings),
        )

        if self.on_portfolio_update:
            self.on_portfolio_update(report)

        return report

    # ------------------------------------------------------------------
    # Intraday Tick / Mark-to-Market (Requirement 3, 5)
    # ------------------------------------------------------------------

    def process_live_tick(self, symbol: Symbol, price: float, timestamp: int) -> None:
        """Processes real-time market ticks for mark-to-market updates and immediate SL/TP exits."""
        symbol_str = str(symbol)

        # Update mark-to-market price
        self.portfolio_engine.update_market_prices({symbol_str: price})

        # Check pending setups for entry trigger at tick price
        setup = self.execution_engine.pending_setups.get(symbol_str)
        if setup is not None:
            should_fill = False
            if (setup.direction == Bias.BULLISH and price <= setup.entry) or (setup.direction == Bias.BEARISH and price >= setup.entry):
                should_fill = True

            if should_fill:
                candle = OHLCV(
                    open_time=timestamp,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0.0,
                )
                self.execution_engine.process_market_tick(symbol, candle)

        # Check active positions for SL/TP breach at tick price
        for pos in list(self.portfolio_engine.state.positions):
            if str(pos.symbol) != symbol_str:
                continue

            reason = None
            if pos.direction == Bias.BULLISH:
                if price <= pos.stop_loss:
                    reason = "STOP_LOSS"
                elif price >= pos.take_profit:
                    reason = "TAKE_PROFIT"
            else:
                if price >= pos.stop_loss:
                    reason = "STOP_LOSS"
                elif price <= pos.take_profit:
                    reason = "TAKE_PROFIT"

            if reason is not None:
                candle = OHLCV(
                    open_time=timestamp,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0.0,
                )
                self.execution_engine.process_market_tick(symbol, candle)

        # Update drawdown
        self.drawdown_monitor.update(self.portfolio_engine.state.account.total_capital, timestamp)

    # ------------------------------------------------------------------
    # Persistence & Recovery (Requirements 6, 11)
    # ------------------------------------------------------------------

    def save_state(self, timestamp: int | None = None) -> None:
        """Persists current state snapshot atomically to disk."""
        if self.state_store is None:
            return

        import time

        ts = timestamp if timestamp is not None else int(time.time() * 1000)
        drawdown_state = DrawdownState(
            current_equity=self.portfolio_engine.state.account.total_capital,
            ath_equity=self.drawdown_monitor.ath_equity,
            max_drawdown_pct=self.drawdown_monitor.max_drawdown_pct,
            timestamp=ts,
        )

        state = PaperTradingState(
            account=self.portfolio_engine.state.account,
            drawdown=drawdown_state,
            open_positions=self.portfolio_engine.state.positions,
            pending_setups=dict(self.execution_engine.pending_setups),
            last_processed_candles=dict(self.last_processed_candles),
            completed_trades=self.portfolio_engine.history,
            updated_at=ts,
        )
        self.state_store.save(state)

    def load_state(self) -> bool:
        """Restores state from persistence store. Returns True if restored."""
        if self.state_store is None:
            return False

        saved = self.state_store.load()
        if saved is None:
            return False

        # Restore PortfolioEngine
        self.portfolio_engine.restore_state(
            account=saved.account,
            positions=saved.open_positions,
            history=saved.completed_trades,
            timestamp=saved.updated_at,
        )

        # Restore DrawdownMonitor
        self.drawdown_monitor.restore_state(
            ath_equity=saved.drawdown.ath_equity,
            max_drawdown_pct=saved.drawdown.max_drawdown_pct,
        )

        # Restore Pending Setups
        self.execution_engine.pending_setups = dict(saved.pending_setups)

        # Restore Processed Candle Timestamps
        self.last_processed_candles = dict(saved.last_processed_candles)

        logger.info(
            f"Forward paper trading state restored: {len(saved.open_positions)} open positions, "
            f"{len(saved.pending_setups)} pending setups, equity: ${saved.account.total_capital:.2f}"
        )
        return True

    # ------------------------------------------------------------------
    # Internal Event Handlers
    # ------------------------------------------------------------------

    def _on_portfolio_event(self, event: Any) -> None:
        from neon_radar.domain.events import PositionClosed, PositionOpened

        if isinstance(event, PositionOpened):
            logger.info(
                f"[PAPER FILL] {event.position.symbol} {event.position.direction.value} "
                f"Entry: {event.position.entry_price} Qty: {event.position.quantity:.4f} "
                f"Notional: ${event.position.position_size:.2f} Fee: ${event.position.entry_fee:.4f}"
            )
            if self.on_trade_open:
                self.on_trade_open(event.position)
        elif isinstance(event, PositionClosed):
            p = event.position
            s = p.execution_summary
            logger.info(
                f"[PAPER EXIT] {p.symbol} {p.direction.value} Exit: {p.exit_price} "
                f"Reason: {p.close_reason.value if hasattr(p.close_reason, 'value') else p.close_reason} "
                f"Net PnL: ${s.net_pnl:.2f} (Fees: ${s.entry_fee + s.exit_fee:.2f}, "
                f"Slippage: ${s.slippage_cost:.2f}, Funding: ${s.funding_cost:.2f})"
            )
            if self.on_trade_close:
                self.on_trade_close(p)
