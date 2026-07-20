"""Paper Trading Engine Service.

Evaluates new kline data against the scoring engine to open virtual trades,
and monitors active virtual trades for exit conditions (SL/TP).
"""
from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from neon_radar.application.services.analysis import analyze_series
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.trading.paper import VirtualPortfolio, VirtualPosition
from neon_radar.domain.trading.setup import TradeSetupEngine
from neon_radar.utils.logging import get_logger

if TYPE_CHECKING:
    from neon_radar.config.models import ScoringRulesConfig
    from neon_radar.domain.models import KlineSeries, Symbol

logger = get_logger(__name__)

# Setup a dedicated logger for paper trading events
paper_logger = logging.getLogger("neon_radar.paper_events")
paper_logger.setLevel(logging.INFO)
# Don't propagate to root logger if we want it isolated, but root is fine for now
# We will attach a FileHandler in the CLI

class PaperTradingEngine:
    """Core logic for executing paper trades against incoming live data."""

    def __init__(
        self,
        portfolio: VirtualPortfolio,
        scoring_config: ScoringRulesConfig,
        trades_csv_path: Path | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.scoring_config = scoring_config
        self.trades_csv_path = trades_csv_path
        self._last_eval_time: dict[str, int] = {}

        # Build rules and setup engine
        self._rules = RuleRegistry.build_all(scoring_config)
        self._setup_engine = TradeSetupEngine(
            min_confidence=0.5,  # Could be pulled from config if available
            regime_classifier=None,  # Will configure below
            regime_config=None,
        )

        # Configure regime filter if enabled
        if scoring_config.regime_filter:
            from neon_radar.application.services.regime_classifier import RuleBasedRegimeClassifier
            from neon_radar.domain.trading.regime import RegimeFilterConfig

            regime_config = RegimeFilterConfig(**scoring_config.regime_filter)
            self._setup_engine.regime_config = regime_config
            self._setup_engine.regime_classifier = RuleBasedRegimeClassifier(regime_config)

        self._ensure_csv_headers()

    def _ensure_csv_headers(self) -> None:
        """Initialize the trades CSV if it doesn't exist."""
        if not self.trades_csv_path:
            return

        if not self.trades_csv_path.exists():
            self.trades_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trades_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Symbol", "Direction", "EntryTime", "ExitTime",
                    "EntryPrice", "ExitPrice", "Quantity", "ExitReason", "NetPnL", "NewBalance"
                ])

    def _log_trade_to_csv(self, position: VirtualPosition, exit_price: float, reason: str, exit_time: int, net_pnl: float) -> None:
        if not self.trades_csv_path:
            return

        entry_dt = datetime.fromtimestamp(position.entry_time / 1000, tz=UTC).isoformat()
        exit_dt = datetime.fromtimestamp(exit_time / 1000, tz=UTC).isoformat()

        with open(self.trades_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                position.symbol,
                position.direction.value,
                entry_dt,
                exit_dt,
                f"{position.entry_price:.6f}",
                f"{exit_price:.6f}",
                f"{position.quantity:.6f}",
                reason,
                f"{net_pnl:.4f}",
                f"{self.portfolio.balance:.4f}"
            ])

    def process_kline(self, symbol: Symbol, series: KlineSeries) -> None:
        """Process the latest market data for a symbol.
        
        1. Checks if an active position hits SL/TP against the current incomplete candle.
        2. If no position, evaluates the *closed* series for a new setup.
        """
        if series.is_empty or len(series.candles) < 2:
            return

        latest_kline = series.candles[-1] # This is the current, incomplete candle

        from neon_radar.domain.models import KlineSeries
        closed_series = KlineSeries(symbol=series.symbol, timeframe=series.timeframe, candles=series.candles[:-1])

        sym_str = str(symbol)

        # 1. Manage existing position
        if sym_str in self.portfolio.positions:
            position = self.portfolio.positions[sym_str]
            exit_reason = position.update(latest_kline)

            if exit_reason:
                # Close position
                exit_price = position.stop_loss if exit_reason == "SL" else position.take_profit

                net_pnl = self.portfolio.close_position(sym_str, exit_price, exit_reason, latest_kline.open_time)

                msg = f"CLOSED {position.direction.value} {sym_str} | Reason: {exit_reason} | PnL: {net_pnl:.2f} | Bal: {self.portfolio.balance:.2f}"
                paper_logger.info(msg)

                self._log_trade_to_csv(position, exit_price, exit_reason, latest_kline.open_time, net_pnl)

            return # Skip opening a new trade on the same candle we exited

        # 2. Evaluate for new entry (only if no active position)
        if not self.portfolio.can_open_position(sym_str):
            return

        last_eval_time = self._last_eval_time.get(sym_str, 0)
        last_closed_candle = closed_series.candles[-1]

        if last_closed_candle.open_time <= last_eval_time:
            return # Already evaluated this candle

        self._last_eval_time[sym_str] = last_closed_candle.open_time

        analysis = analyze_series(closed_series, self._rules)
        setup = self._setup_engine.evaluate(analysis)

        if setup:
            # We enter at market, which is effectively the latest_kline's current price (close)
            entry_price = latest_kline.close

            # Position Sizing
            qty = self.portfolio.calculate_position_size(entry_price, setup.stop_loss)

            if qty > 0:
                setup = type(setup)(
                    direction=setup.direction,
                    entry_price=entry_price,
                    stop_loss=setup.stop_loss,
                    take_profit_1=setup.take_profit_1,
                    take_profit_2=setup.take_profit_2,
                    risk_reward=setup.risk_reward,
                    diagnostics=setup.diagnostics
                )

                pos = VirtualPosition.from_setup(symbol, setup, qty, latest_kline.open_time)
                self.portfolio.open_position(pos)

                msg = f"OPENED {pos.direction.value} {sym_str} | Entry: {entry_price:.4f} | SL: {setup.stop_loss:.4f} | TP: {setup.take_profit_1:.4f} | Qty: {qty:.6f}"
                paper_logger.info(msg)
