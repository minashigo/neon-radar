"""Portfolio Engine.

Stateful service responsible for managing the PortfolioState as the single source of truth.
It handles opening/closing positions, updating market prices, and applying domain events.
"""

from collections.abc import Callable

from neon_radar.domain.enums import Bias
from neon_radar.domain.events import PositionClosed, PositionOpened
from neon_radar.domain.execution_costs import ExecutionCostSummary
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

    def subscribe(self, callback: Callable) -> None:
        self._subscribers.append(callback)

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

                updated_pos = OpenPosition(
                    symbol=pos.symbol,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    quantity=pos.quantity,
                    position_size=pos.position_size,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    opened_at=pos.opened_at,
                    unrealized_pnl=unrealized,
                    fees=pos.fees,
                    funding_paid=pos.funding_paid,
                )
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
        total_funding = stats.total_funding + execution_summary.funding_cost

        # Balance = old free capital + position margin + net PnL
        new_free = (
            self._state.account.free_capital + target_pos.position_size + execution_summary.net_pnl
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
