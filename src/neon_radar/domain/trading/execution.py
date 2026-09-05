"""Domain models for execution costs calculation.

Provides independent components (FeeModel, SlippageModel, FundingModel) and a unified CostModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from neon_radar.domain.execution_costs import (
    BinanceFundingModel as _CanonicalFundingModel,
)
from neon_radar.domain.execution_costs import (
    BinanceFuturesFeeModel as _CanonicalFeeModel,
)
from neon_radar.domain.execution_costs import (
    FixedSlippageModel as _CanonicalSlippageModel,
)

if TYPE_CHECKING:
    from neon_radar.application.services.trade_backtester import HistoricalFundingProvider
    from neon_radar.domain.enums import Bias
    from neon_radar.domain.models import Symbol


class ExecutionType(StrEnum):
    """How the order was executed against the orderbook."""

    MAKER = "maker"
    TAKER = "taker"


@dataclass(slots=True, frozen=True)
class TradeCosts:
    """Breakdown of costs for a single trade execution (all in percentages or fractional form)."""

    fees_pct: float
    slippage_pct: float
    funding_pct: float

    @property
    def total_costs_pct(self) -> float:
        """The total execution cost as a percentage."""
        return self.fees_pct + self.slippage_pct + self.funding_pct


class FeeModel(Protocol):
    """Calculates entry and exit fees."""

    def calculate_entry_fee_pct(self, order_type: ExecutionType) -> float: ...

    def calculate_exit_fee_pct(self, order_type: ExecutionType) -> float: ...


class BinanceFuturesFeeModel:
    """Binance Futures standard fee tier implementation."""

    def __init__(self, maker_fee: float = 0.0002, taker_fee: float = 0.0005) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self._canonical = _CanonicalFeeModel(maker_fee_pct=maker_fee, taker_fee_pct=taker_fee)

    def calculate_entry_fee_pct(self, order_type: ExecutionType) -> float:
        return self._canonical.calculate_entry_fee(1.0, order_type)

    def calculate_exit_fee_pct(self, order_type: ExecutionType) -> float:
        return self._canonical.calculate_exit_fee(1.0, order_type)


class SlippageModel(Protocol):
    """Calculates slippage costs."""

    def calculate_slippage_pct(
        self, symbol: Symbol, order_type: ExecutionType, trade_direction: Bias
    ) -> float: ...


class FixedSlippageModel:
    """A deterministic fixed slippage model."""

    def __init__(self, slippage_pct: float = 0.0005) -> None:
        self.slippage_pct = slippage_pct
        self._canonical = _CanonicalSlippageModel(slippage_pct=slippage_pct)

    def calculate_slippage_pct(
        self, symbol: Symbol, order_type: ExecutionType, trade_direction: Bias
    ) -> float:
        """Slippage only applies to TAKER orders usually."""
        return self._canonical.calculate_slippage(1.0, symbol, order_type, trade_direction)


class FundingModel(Protocol):
    """Calculates funding costs accrued over the trade holding period."""

    def calculate_funding_cost_pct(
        self,
        symbol: Symbol,
        direction: Bias,
        entry_time: int,
        exit_time: int,
        provider: HistoricalFundingProvider,
    ) -> float: ...


class BinanceFundingModel:
    """Calculates funding costs using the exact 8h funding rate intervals."""

    def __init__(self) -> None:
        self._canonical = _CanonicalFundingModel()

    def calculate_funding_cost_pct(
        self,
        symbol: Symbol,
        direction: Bias,
        entry_time: int,
        exit_time: int,
        provider: HistoricalFundingProvider,
    ) -> float:
        """Accumulates funding rates via canonical execution costs model."""
        return self._canonical.calculate_funding_cost_pct(
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            exit_time=exit_time,
            provider=provider,
        )


class CostModel:
    """Unified service for calculating all trade costs.

    Acts as a facade so the Backtester does not have to deal with individual models.
    """

    def __init__(
        self,
        fee_model: FeeModel | None = None,
        slippage_model: SlippageModel | None = None,
        funding_model: FundingModel | None = None,
    ) -> None:
        self.fee_model = fee_model or BinanceFuturesFeeModel()
        self.slippage_model = slippage_model or FixedSlippageModel()
        self.funding_model = funding_model or BinanceFundingModel()

    def calculate_costs(
        self,
        symbol: Symbol,
        direction: Bias,
        entry_type: ExecutionType,
        exit_type: ExecutionType,
        entry_time: int,
        exit_time: int,
        funding_provider: HistoricalFundingProvider | None = None,
    ) -> TradeCosts:
        fees = self.fee_model.calculate_entry_fee_pct(
            entry_type
        ) + self.fee_model.calculate_exit_fee_pct(exit_type)

        slippage = self.slippage_model.calculate_slippage_pct(
            symbol, entry_type, direction
        ) + self.slippage_model.calculate_slippage_pct(symbol, exit_type, direction)

        funding = 0.0
        if funding_provider is not None:
            funding = self.funding_model.calculate_funding_cost_pct(
                symbol, direction, entry_time, exit_time, funding_provider
            )

        return TradeCosts(
            fees_pct=fees,
            slippage_pct=slippage,
            funding_pct=funding,
        )
