"""Trade-based backtesting engine.

Runs the scoring engine against historical data, simulating actual trade execution.
Ensures zero look-ahead bias by strictly separating signal generation (using candles up to T)
from trade execution (using candle T+1).
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from neon_radar.application.services.analysis import analyze_series
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import KlineSeries, Symbol
from neon_radar.domain.trading.backtest import Trade, TradeExitReason, TradeStatus

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neon_radar.config.models import TimeFrame
    from neon_radar.config.scoring_models import ScoringRulesConfig
    from neon_radar.domain.scoring.factor_rule import FactorRule
    from neon_radar.domain.trading.setup import TradeSetup
    from neon_radar.infrastructure.exchanges.base import ExchangeClient

logger = logging.getLogger(__name__)


class TradeBacktester:
    """Walk-forward trade simulation engine."""

    def __init__(
        self,
        *,
        exchange: ExchangeClient,
        scoring_config: ScoringRulesConfig,
        rules: tuple[FactorRule, ...] | None = None,
    ) -> None:
        self._exchange = exchange
        self._scoring_config = scoring_config
        if rules is None:
            raise ValueError("Pre-built rules are required.")
        self._rules = rules
        self._series_cache: dict[tuple[str, str], KlineSeries] = {}

    async def run(
        self,
        start_date: date,
        end_date: date,
        symbols: Iterable[Symbol],
        timeframe: TimeFrame = "1d",
        min_history_candles: int = 50,
    ) -> tuple[Trade, ...]:
        """Run trade-based walk-forward backtest over the period."""
        symbols = tuple(symbols)
        if not symbols or end_date <= start_date:
            return ()

        await self._prefetch(symbols, timeframe, start_date, end_date)

        all_trades: list[Trade] = []
        for symbol in symbols:
            trades = self._simulate_symbol(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                min_history_candles=min_history_candles,
            )
            all_trades.extend(trades)

        return tuple(all_trades)

    async def _prefetch(
        self,
        symbols: tuple[Symbol, ...],
        timeframe: str,
        start_date: date,
        end_date: date,
    ) -> None:
        """Fetch enough history for every (symbol, timeframe) once."""
        from neon_radar.config.models import TimeFrame

        tf = TimeFrame(timeframe)
        fetch_end_dt = datetime.combine(
            end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        )
        fetch_end = int(fetch_end_dt.timestamp() * 1000)
        limit = 1500

        for symbol in symbols:
            key = (str(symbol), timeframe)
            if key in self._series_cache:
                continue
            try:
                series = await self._exchange.get_klines(
                    symbol,
                    tf,
                    end_time=fetch_end,
                    limit=limit,
                )
                self._series_cache[key] = series
            except Exception as exc:
                logger.warning(f"Failed to fetch klines for {symbol}: {exc}")
                self._series_cache[key] = KlineSeries(symbol=symbol, timeframe=tf, candles=())

    def _simulate_symbol(
        self,
        *,
        symbol: Symbol,
        timeframe: str,
        start_date: date,
        end_date: date,
        min_history_candles: int,
    ) -> list[Trade]:
        """Simulate trades for a single symbol."""
        series = self._series_cache.get((str(symbol), timeframe))
        if series is None or len(series.candles) == 0:
            return []

        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
        start_ms = int(start_dt.timestamp() * 1000)

        end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
        end_ms = int(end_dt.timestamp() * 1000)

        # Find the starting index
        start_idx = 0
        while start_idx < len(series.candles) and series.candles[start_idx].open_time < start_ms:
            start_idx += 1

        if start_idx < min_history_candles:
            start_idx = min_history_candles

        trades: list[Trade] = []
        active_trade: Trade | None = None
        pending_setup: TradeSetup | None = None

        for i in range(start_idx, len(series.candles)):
            candle = series.candles[i]
            if candle.open_time >= end_ms:
                break

            # 1. Execution Phase (process active trade or pending setup)
            if active_trade is not None:
                # Check pessimistic SL first
                closed = False
                if active_trade.direction == Bias.BULLISH:
                    if candle.low <= active_trade.stop_loss:
                        active_trade = self._close_trade(
                            active_trade,
                            active_trade.stop_loss,
                            candle.open_time,
                            TradeStatus.LOSS,
                            TradeExitReason.STOP_LOSS,
                        )
                        trades.append(active_trade)
                        active_trade = None
                        closed = True
                    elif candle.high >= active_trade.take_profit:
                        active_trade = self._close_trade(
                            active_trade,
                            active_trade.take_profit,
                            candle.open_time,
                            TradeStatus.WIN,
                            TradeExitReason.TAKE_PROFIT,
                        )
                        trades.append(active_trade)
                        active_trade = None
                        closed = True
                else:  # BEARISH
                    if candle.high >= active_trade.stop_loss:
                        active_trade = self._close_trade(
                            active_trade,
                            active_trade.stop_loss,
                            candle.open_time,
                            TradeStatus.LOSS,
                            TradeExitReason.STOP_LOSS,
                        )
                        trades.append(active_trade)
                        active_trade = None
                        closed = True
                    elif candle.low <= active_trade.take_profit:
                        active_trade = self._close_trade(
                            active_trade,
                            active_trade.take_profit,
                            candle.open_time,
                            TradeStatus.WIN,
                            TradeExitReason.TAKE_PROFIT,
                        )
                        trades.append(active_trade)
                        active_trade = None
                        closed = True

                if closed:
                    pending_setup = None

            elif pending_setup is not None:
                # Check entry trigger
                if candle.low <= pending_setup.entry_price <= candle.high:
                    active_trade = Trade(
                        symbol=symbol,
                        direction=pending_setup.direction,
                        entry_time=candle.open_time,
                        entry_price=pending_setup.entry_price,
                        stop_loss=pending_setup.stop_loss,
                        take_profit=pending_setup.take_profit_1,  # MVP: TP1 target
                        status=TradeStatus.OPEN,
                        exit_reason=TradeExitReason.NONE,
                    )
                # In this MVP, setup expires if not triggered on the very next candle, or it persists?
                # Usually we let it expire if a new setup overrides it.
                # To be strict, let's keep it until overridden or active trade created.

            # 2. Analysis Phase (generate signals for NEXT candle)
            # We provide history inclusive of current candle
            if active_trade is None:
                history = series.candles[: i + 1]
                history_series = KlineSeries(
                    symbol=series.symbol, timeframe=series.timeframe, candles=history
                )

                try:
                    result = analyze_series(
                        history_series,
                        self._rules,
                        min_confidence=self._scoring_config.min_confidence,
                        timestamp=int(history[-1].open_time),
                    )
                    pending_setup = result.trade_setup
                except Exception:
                    pending_setup = None

        # Clean up any open trade at the end
        if active_trade is not None:
            # Force close at the last close price
            last_candle = series.candles[-1]
            status = TradeStatus.WIN if active_trade.pnl_pct > 0 else TradeStatus.LOSS
            if active_trade.pnl_pct == 0:
                status = TradeStatus.BREAK_EVEN
            active_trade = self._close_trade(
                active_trade,
                last_candle.close,
                last_candle.open_time,
                status,
                TradeExitReason.FORCE_CLOSE,
            )
            trades.append(active_trade)

        return tuple(trades)

    @staticmethod
    def _close_trade(
        trade: Trade,
        exit_price: float,
        exit_time: int,
        status: TradeStatus,
        exit_reason: TradeExitReason,
    ) -> Trade:
        from dataclasses import replace

        return replace(
            trade,
            exit_price=exit_price,
            exit_time=exit_time,
            status=status,
            exit_reason=exit_reason,
        )
