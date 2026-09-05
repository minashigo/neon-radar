"""Domain models for realistic execution costs calculation.

Provides independent components (FeeModel, SlippageModel, FundingModel) and a unified ExecutionCostSummary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from neon_radar.domain.enums import Bias  # noqa: TC001

if TYPE_CHECKING:
    from neon_radar.application.services.trade_backtester import HistoricalFundingProvider
    from neon_radar.domain.models import Symbol
    from neon_radar.domain.trading.execution import ExecutionType


@dataclass(slots=True, frozen=True)
class ExecutionCostSummary:
    """Immutable summary of all execution costs for a closed position (in quote asset)."""

    entry_fee: float
    exit_fee: float
    slippage_cost: float
    funding_cost: float
    gross_pnl: float

    @property
    def total_cost(self) -> float:
        """Total costs incurred by this position."""
        return self.entry_fee + self.exit_fee + self.slippage_cost + self.funding_cost

    @property
    def net_pnl(self) -> float:
        """Net Profit and Loss after deducting all costs."""
        return self.gross_pnl - self.total_cost


class FeeModel(Protocol):
    """Calculates absolute entry and exit fees."""

    def calculate_entry_fee(self, notional_value: float, order_type: ExecutionType) -> float: ...

    def calculate_exit_fee(self, notional_value: float, order_type: ExecutionType) -> float: ...


class BinanceFuturesFeeModel:
    """Binance Futures standard fee tier implementation.

    Default Maker: 0.02%
    Default Taker: 0.05%
    """

    def __init__(self, maker_fee_pct: float = 0.0002, taker_fee_pct: float = 0.0005) -> None:
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct

    def calculate_entry_fee(self, notional_value: float, order_type: ExecutionType) -> float:
        rate = self.maker_fee_pct if order_type.name == "MAKER" else self.taker_fee_pct
        return notional_value * rate

    def calculate_exit_fee(self, notional_value: float, order_type: ExecutionType) -> float:
        rate = self.maker_fee_pct if order_type.name == "MAKER" else self.taker_fee_pct
        return notional_value * rate


class SlippageModel(Protocol):
    """Calculates absolute slippage costs."""

    def calculate_slippage(
        self, notional_value: float, symbol: Symbol, order_type: ExecutionType, direction: Bias
    ) -> float: ...


class FixedSlippageModel:
    """A deterministic fixed percentage slippage model."""

    def __init__(self, slippage_pct: float = 0.0005) -> None:
        self.slippage_pct = slippage_pct

    def calculate_slippage(
        self, notional_value: float, symbol: Symbol, order_type: ExecutionType, direction: Bias
    ) -> float:
        """Slippage usually only applies to TAKER orders, representing the cost of crossing the spread."""
        if order_type.name == "TAKER":
            return notional_value * self.slippage_pct
        return 0.0


class FundingModel(Protocol):
    """Calculates funding costs accrued over the trade holding period."""

    def calculate_funding_cost(
        self,
        notional_value: float,
        symbol: Symbol,
        direction: Bias,
        entry_time: int,
        exit_time: int,
        provider: HistoricalFundingProvider,
    ) -> float: ...


class BinanceFundingModel:
    """Calculates funding costs using the exact 8h funding rate intervals."""

    def calculate_funding_cost_pct(
        self,
        symbol: Symbol,
        direction: Bias,
        entry_time: int,
        exit_time: int,
        provider: HistoricalFundingProvider,
    ) -> float:
        """Accumulates fractional funding rates over 8h boundaries between entry and exit."""
        eight_hours_ms = 8 * 60 * 60 * 1000
        next_boundary = math.ceil(entry_time / eight_hours_ms) * eight_hours_ms

        cost_pct = 0.0
        current_time = next_boundary
        while current_time <= exit_time:
            rate_obj = provider.get_funding_rate_at(symbol, current_time)
            if rate_obj is not None:
                if direction.name == "BULLISH":
                    cost_pct += rate_obj.rate
                else:
                    cost_pct -= rate_obj.rate
            current_time += eight_hours_ms

        return cost_pct

    def calculate_funding_cost(
        self,
        notional_value: float,
        symbol: Symbol,
        direction: Bias,
        entry_time: int,
        exit_time: int,
        provider: HistoricalFundingProvider,
    ) -> float:
        """Accumulates funding rates as absolute quote value."""
        return notional_value * self.calculate_funding_cost_pct(
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            exit_time=exit_time,
            provider=provider,
        )
