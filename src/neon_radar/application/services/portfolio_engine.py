"""Portfolio Engine.

Stateful service responsible for managing the PortfolioState as the single source of truth.
It handles opening/closing positions, updating market prices, and applying domain events.
"""

from collections.abc import Callable
from dataclasses import replace
from math import isfinite

from neon_radar.domain.enums import Bias
from neon_radar.domain.events import FundingSettled, PositionClosed, PositionOpened
from neon_radar.domain.execution_costs import ExecutionCostSummary
from neon_radar.domain.funding import FundingSettlement
from neon_radar.domain.portfolio import (
    AccountState,
    ClosedPosition,
    OpenPosition,
    PortfolioState,
    PortfolioStatistics,
)


class PortfolioEngine:
    def __init__(self, initial_capital: float, currency: str = "USDT") -> None:
        account = AccountState(
            total_capital=initial_capital, free_capital=initial_capital, currency=currency
        )
        self._state = PortfolioState(account=account, positions=())
        self._history: list[ClosedPosition] = []
        self._subscribers: list[Callable] = []

    @property
    def state(self) -> PortfolioState:
        return self._state

    @property
    def history(self) -> tuple[ClosedPosition, ...]:
        """Closed positions history."""
        return tuple(self._history)

    def subscribe(self, callback: Callable) -> None:
        self._subscribers.append(callback)

    def restore_state(
        self,
        account: AccountState,
        positions: tuple[OpenPosition, ...] = (),
        history: tuple[ClosedPosition, ...] = (),
        statistics: PortfolioStatistics | None = None,
        timestamp: int | None = None,
    ) -> None:
        """Restore portfolio state and history from persistence."""
        self._state = PortfolioState(
            account=account,
            positions=positions,
            statistics=statistics or self._restored_statistics(positions, history),
            timestamp=timestamp,
        )
        self._history = list(history)

    @staticmethod
    def _restored_statistics(
        positions: tuple[OpenPosition, ...], history: tuple[ClosedPosition, ...]
    ) -> PortfolioStatistics:
        """Reconstruct legacy snapshots; funding on still-open positions is already cash."""
        return PortfolioStatistics(
            total_trades=len(history),
            winning_trades=sum(p.net_pnl > 0 for p in history),
            losing_trades=sum(p.net_pnl <= 0 for p in history),
            gross_profit=sum(max(0, p.gross_pnl) for p in history),
            gross_loss=sum(max(0, -p.gross_pnl) for p in history),
            total_fees=sum(
                p.execution_summary.entry_fee + p.execution_summary.exit_fee for p in history
            ),
            total_slippage=sum(p.execution_summary.slippage_cost for p in history),
            total_funding=sum(p.execution_summary.funding_cost for p in history)
            + sum(p.funding_paid for p in positions),
        )

    def apply_funding_settlements(self, events: tuple[FundingSettlement, ...]) -> float:
        """Atomically debit/credit one timestamp, netting simultaneous assets.

        No margin borrowing/liquidation is invented: an unrepresentable negative
        free balance or equity raises before mutation. Same-event retries are no-ops.
        """
        if not events:
            return 0.0
        timestamp = events[0].timestamp
        if any(e.timestamp != timestamp for e in events) or len({e.symbol for e in events}) != len(
            events
        ):
            raise ValueError("Funding group requires one timestamp and unique symbols")
        positions = {p.symbol: p for p in self._state.positions}
        updates = {}
        applied = []
        total = 0.0
        for event in events:
            pos = positions.get(event.symbol)
            if pos is None:
                raise ValueError("Cannot settle funding for a missing position")
            if event.timestamp <= pos.opened_at:
                raise ValueError("Settlement must follow position entry")
            if (
                pos.funding_processed_until is not None
                and event.timestamp <= pos.funding_processed_until
            ):
                continue
            if pos.direction not in (Bias.BULLISH, Bias.BEARISH):
                raise ValueError("Funding requires a directional position")
            cost = (
                pos.quantity
                * event.mark_price
                * event.rate
                * (1 if pos.direction == Bias.BULLISH else -1)
            )
            if not isfinite(cost):
                raise ValueError("Funding cash flow must be finite")
            total += cost
            updates[pos.symbol] = replace(
                pos, funding_paid=pos.funding_paid + cost, funding_processed_until=timestamp
            )
            applied.append(event)
        if not applied:
            return 0.0
        free = self._state.account.free_capital - total
        equity = self._state.account.total_capital - total
        if not isfinite(total) or free < 0 or equity < 0:
            raise ValueError(
                "Funding exceeds representable cash/equity; margin insolvency is not modeled"
            )
        account = replace(self._state.account, total_capital=equity, free_capital=free)
        self._state = replace(
            self._state,
            account=account,
            positions=tuple(updates.get(p.symbol, p) for p in self._state.positions),
            statistics=replace(
                self._state.statistics, total_funding=self._state.statistics.total_funding + total
            ),
            timestamp=timestamp,
        )
        self._publish(FundingSettled(timestamp, tuple(applied), total))
        return total

    def _publish(self, event) -> None:
        for cb in self._subscribers:
            cb(event)

    def update_market_prices(self, symbol_prices: dict[str, float]) -> None:
        """Update unrealized PnL based on latest market prices."""
        updated_positions = []
        total_unrealized = 0.0

        for pos in self._state.positions:
            price = symbol_prices.get(str(pos.symbol))
            if price is not None:
                if pos.direction == Bias.BULLISH:
                    unrealized = (price - pos.entry_price) * pos.quantity
                else:
                    unrealized = (pos.entry_price - price) * pos.quantity

                updated_pos = replace(pos, unrealized_pnl=unrealized)
                updated_positions.append(updated_pos)
                total_unrealized += unrealized
            else:
                updated_positions.append(pos)
                total_unrealized += pos.unrealized_pnl

        # We don't change free capital until position is closed.
        # But we do update total_capital (equity)
        # Note: In a real system, you might want a clearer separation of balance vs equity.
        account = AccountState(
            total_capital=self._state.account.free_capital
            + sum(p.position_size for p in updated_positions)
            + total_unrealized,
            free_capital=self._state.account.free_capital,
            currency=self._state.account.currency,
        )
        self._state = PortfolioState(
            account=account,
            positions=tuple(updated_positions),
            statistics=self._state.statistics,
            timestamp=self._state.timestamp,
        )

    def open_position(self, position: OpenPosition) -> None:
        if position.position_size > self._state.account.free_capital:
            raise ValueError("Insufficient free capital to open position.")

        new_positions = [*list(self._state.positions), position]

        account = AccountState(
            total_capital=self._state.account.total_capital,
            free_capital=self._state.account.free_capital - position.position_size,
            currency=self._state.account.currency,
        )

        self._state = PortfolioState(
            account=account,
            positions=tuple(new_positions),
            statistics=self._state.statistics,
            timestamp=position.opened_at,
        )

        event = PositionOpened(timestamp=position.opened_at, position=position)
        self._publish(event)

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        closed_at: int,
        execution_summary: ExecutionCostSummary,
        reason: str = "MANUAL",
    ) -> ClosedPosition | None:
        target_pos = None
        remaining_positions = []
        for pos in self._state.positions:
            if str(pos.symbol) == symbol:
                target_pos = pos
            else:
                remaining_positions.append(pos)

        if not target_pos:
            return None

        if (
            target_pos.funding_processed_until is not None
            and closed_at < target_pos.funding_processed_until
        ):
            raise ValueError("Cannot close before already booked funding")

        closed = ClosedPosition(
            symbol=target_pos.symbol,
            direction=target_pos.direction,
            entry_price=target_pos.entry_price,
            exit_price=exit_price,
            quantity=target_pos.quantity,
            execution_summary=execution_summary,
            close_reason=reason,
            entry_time=target_pos.opened_at,
            exit_time=closed_at,
            stop_loss=target_pos.stop_loss,
            take_profit=target_pos.take_profit,
            initial_risk=target_pos.max_risk,
            capital_at_entry=getattr(target_pos, "capital_at_entry", 0.0),
            diagnostics=getattr(target_pos, "diagnostics", None),
            timeframe=getattr(target_pos, "timeframe", "1d"),
            signal_time=getattr(target_pos, "signal_time", None),
            htf_regime=getattr(target_pos, "htf_regime", None),
        )
        self._history.append(closed)

        # Update statistics
        stats = self._state.statistics
        total_trades = stats.total_trades + 1

        is_win = execution_summary.net_pnl > 0
        winning_trades = stats.winning_trades + (1 if is_win else 0)
        losing_trades = stats.losing_trades + (1 if not is_win else 0)

        gross_pnl = execution_summary.gross_pnl
        gross_profit = stats.gross_profit + (gross_pnl if gross_pnl > 0 else 0)
        gross_loss = stats.gross_loss + (abs(gross_pnl) if gross_pnl <= 0 else 0)

        total_fees = stats.total_fees + execution_summary.entry_fee + execution_summary.exit_fee
        total_slippage = stats.total_slippage + execution_summary.slippage_cost
        total_funding = (
            stats.total_funding + execution_summary.funding_cost - target_pos.funding_paid
        )

        # Net PnL includes all funding; add back the amount already booked in cash.
        new_free = (
            self._state.account.free_capital
            + target_pos.position_size
            + execution_summary.net_pnl
            + target_pos.funding_paid
        )
        # Recompute total capital based on remaining positions
        total_unrealized = sum(p.unrealized_pnl for p in remaining_positions)
        new_total = new_free + sum(p.position_size for p in remaining_positions) + total_unrealized

        # We don't recalculate max_drawdown here deeply since DrawdownMonitor handles the actual peak/trough,
        # but we preserve existing stats. (Ideally PortfolioEngine integrates with DrawdownMonitor)

        new_stats = PortfolioStatistics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            total_fees=total_fees,
            total_slippage=total_slippage,
            total_funding=total_funding,
            max_drawdown=stats.max_drawdown,
            current_drawdown=stats.current_drawdown,
        )

        account = AccountState(
            total_capital=new_total, free_capital=new_free, currency=self._state.account.currency
        )

        self._state = PortfolioState(
            account=account,
            positions=tuple(remaining_positions),
            statistics=new_stats,
            timestamp=closed_at,
        )

        event = PositionClosed(timestamp=closed_at, position=closed)
        self._publish(event)

        return closed
