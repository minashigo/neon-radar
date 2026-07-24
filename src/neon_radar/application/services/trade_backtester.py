"""Trade-based backtesting engine.

Runs the scoring engine against historical data, simulating actual trade execution.
Ensures zero look-ahead bias by strictly separating signal generation (using candles up to T)
from trade execution (using candle T+1).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from neon_radar.application.services.analysis import analyze_series
from neon_radar.domain.enums import Bias
from neon_radar.domain.models import KlineSeries, Symbol
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.trading.backtest import Trade, TradeExitReason, TradeStatus
from neon_radar.domain.trading.execution import CostModel, ExecutionType
from neon_radar.domain.trading.setup import TradeSetup
from neon_radar.infrastructure.exchanges.base import ExchangeClient
from neon_radar.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neon_radar.config.models import ScoringRulesConfig, TimeFrame
    from neon_radar.domain.funding import FundingRate
    from neon_radar.domain.market_context import HistoricalMarketContext
    from neon_radar.application.services.market_context.history_service import MarketContextHistoryService
    from neon_radar.domain.scoring.factor_rule import FactorRule


class HistoricalFundingProvider(Protocol):
    """Provides historical funding rates for backtesting without look-ahead bias."""

    async def prefetch(self, symbols: tuple[Symbol, ...], start_date: date, end_date: date) -> None:
        """Prefetch all necessary historical funding data."""
        ...

    def get_funding_rate_at(self, symbol: Symbol, timestamp: int) -> FundingRate | None:
        """Return the effective funding rate for the given timestamp."""
        ...


class TradeBacktester:
    """Historical backtester using the full Scoring Engine.

    Architecture
    ------------
    1. Pre-fetches klines for all requested symbols and timeframes.
    2. Sequentially steps through candles.
    3. Triggers `analyze_series()` to generate trading signals exactly
       as it would happen live.
    4. Evaluates executions and generates Trade objects.
    """

    def __init__(
        self,
        exchange: ExchangeClient,
        scoring_config: ScoringRulesConfig,
        rules: tuple[FactorRule, ...] | None = None,
        funding_provider: HistoricalFundingProvider | None = None,
        history_service: MarketContextHistoryService | None = None,
        preloaded_series: dict[tuple[str, str], KlineSeries] | None = None,
        preloaded_context: dict[str, HistoricalMarketContext] | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self._exchange = exchange
        self._scoring_config = scoring_config
        self._rules = rules if rules is not None else RuleRegistry.build_all(scoring_config)
        self._funding_provider = funding_provider
        self._history_service = history_service
        self._series_cache: dict[tuple[str, str], KlineSeries] = preloaded_series or {}
        self._context_cache: dict[str, HistoricalMarketContext] = preloaded_context or {}
        self._cost_model = cost_model or CostModel()

        self._regime_config = None
        self._regime_classifier = None

        if scoring_config.regime_filter:
            from neon_radar.application.services.regime_classifier import RuleBasedRegimeClassifier
            from neon_radar.domain.trading.regime import RegimeFilterConfig

            self._regime_config = RegimeFilterConfig(**scoring_config.regime_filter)
            self._regime_classifier = RuleBasedRegimeClassifier(self._regime_config)

    @property
    def cache(self) -> dict[tuple[str, str], KlineSeries]:
        """Expose the series cache for advanced re-use (e.g. ablation analysis)."""
        return self._series_cache

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
        higher_tf = tf.higher_timeframe

        tfs_to_fetch = [tf]
        if higher_tf is not None:
            tfs_to_fetch.append(higher_tf)

        fetch_end_dt = datetime.combine(
            end_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        )
        fetch_end = int(fetch_end_dt.timestamp() * 1000)
        limit = 1500

        if self._funding_provider is not None:
            try:
                await self._funding_provider.prefetch(symbols, start_date, end_date)
            except Exception as exc:
                logger.warning(f"Failed to prefetch historical funding rates: {exc}")
                
        if self._history_service is not None:
            for symbol in symbols:
                if str(symbol) not in self._context_cache:
                    # Fetch using ms timestamps
                    start_ms = int(datetime.combine(start_date, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
                    end_ms = fetch_end
                    try:
                        ctx = await self._history_service.get_historical_context(symbol, fetch_end, start_ms, end_ms, limit=1500)
                        self._context_cache[str(symbol)] = ctx
                    except Exception as exc:
                        logger.warning(f"Failed to fetch historical market context for {symbol}: {exc}")

        for symbol in symbols:
            for current_tf in tfs_to_fetch:
                key = (str(symbol), current_tf.value)
                if key in self._series_cache:
                    continue
                try:
                    series = await self._exchange.get_klines(
                        symbol,
                        current_tf,
                        end_time=fetch_end,
                        limit=limit,
                    )
                    self._series_cache[key] = series
                except Exception as exc:
                    logger.warning(f"Failed to fetch klines for {symbol} on {current_tf.value}: {exc}")
                    self._series_cache[key] = KlineSeries(symbol=symbol, timeframe=current_tf, candles=())

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
        from neon_radar.config.models import TimeFrame
        tf_enum = TimeFrame(timeframe)
        higher_tf = tf_enum.higher_timeframe

        higher_full_series = None
        if higher_tf is not None:
            higher_full_series = self._series_cache.get((str(symbol), higher_tf.value))

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
                closed = False
                if active_trade.direction == Bias.BULLISH:
                    if candle.low <= active_trade.stop_loss:
                        active_trade = self._close_trade(
                            active_trade,
                            active_trade.stop_loss,
                            candle.open_time,
                            TradeStatus.LOSS,
                            TradeExitReason.STOP_LOSS,
                            ExecutionType.TAKER,
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
                            ExecutionType.MAKER,
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
                            ExecutionType.TAKER,
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
                            ExecutionType.MAKER,
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
                        diagnostics=pending_setup.diagnostics,
                    )

            # 2. Analysis Phase (generate signals for NEXT candle)
            if active_trade is None:
                history = series.candles[: i + 1]
                history_series = KlineSeries(
                    symbol=series.symbol, timeframe=series.timeframe, candles=history
                )

                higher_history_series = None
                if higher_full_series is not None:
                    from neon_radar.config.models import TimeFrame
                    base_tf_enum = TimeFrame(series.timeframe)
                    htf_enum = TimeFrame(higher_full_series.timeframe)

                    current_close_time = candle.open_time + (base_tf_enum.seconds * 1000)
                    htf_history = []
                    for c in higher_full_series.candles:
                        htf_close_time = c.open_time + (htf_enum.seconds * 1000)
                        if htf_close_time <= current_close_time:
                            htf_history.append(c)

                    if htf_history:
                        higher_history_series = KlineSeries(
                            symbol=higher_full_series.symbol,
                            timeframe=higher_full_series.timeframe,
                            candles=tuple(htf_history)
                        )

                funding_val = None
                if self._funding_provider is not None:
                    funding_val = self._funding_provider.get_funding_rate_at(
                        symbol, int(history[-1].open_time)
                    )
                    
                context_val = None
                if str(symbol) in self._context_cache:
                    context_val = self._context_cache[str(symbol)].slice_at(int(history[-1].open_time))

                try:
                    result = analyze_series(
                        history_series,
                        self._rules,
                        min_confidence=self._scoring_config.min_confidence,
                        confluence_bonus=self._scoring_config.confluence_bonus,
                        confluence_penalty=self._scoring_config.confluence_penalty,
                        max_confidence_boost=self._scoring_config.max_confidence_boost,
                        timestamp=int(history[-1].open_time),
                        higher_tf_series=higher_history_series,
                        funding_rate=funding_val,
                        market_context=context_val,
                        regime_classifier=self._regime_classifier,
                        regime_config=self._regime_config,
                    )
                    pending_setup = result.trade_setup
                except Exception:
                    pending_setup = None

        # Clean up any open trade at the end
        if active_trade is not None:
            last_candle = series.candles[-1]
            status = TradeStatus.WIN if active_trade.gross_pnl_pct > 0 else TradeStatus.LOSS
            if active_trade.gross_pnl_pct == 0:
                status = TradeStatus.BREAK_EVEN
            active_trade = self._close_trade(
                active_trade,
                last_candle.close,
                last_candle.open_time,
                status,
                TradeExitReason.FORCE_CLOSE,
                ExecutionType.TAKER,
            )
            trades.append(active_trade)

        return trades

    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        exit_time: int,
        status: TradeStatus,
        exit_reason: TradeExitReason,
        exit_type: ExecutionType,
    ) -> Trade:
        from dataclasses import replace

        costs = self._cost_model.calculate_costs(
            symbol=trade.symbol,
            direction=trade.direction,
            entry_type=ExecutionType.MAKER,  # Entry is assumed MAKER per defaults
            exit_type=exit_type,
            entry_time=trade.entry_time,
            exit_time=exit_time,
            funding_provider=self._funding_provider,
        )

        return replace(
            trade,
            exit_price=exit_price,
            exit_time=exit_time,
            status=status,
            exit_reason=exit_reason,
            costs=costs,
        )
