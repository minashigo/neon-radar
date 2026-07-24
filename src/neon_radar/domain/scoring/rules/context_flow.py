"""Market flow context rules (Long/Short ratio, Taker Flow, Liquidations)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("ls_crowded")
class LongShortCrowdedRule(FactorRule):
    """Contrarian signal when Global Long/Short ratio reaches extremes."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.15,
        description: str | None = None,
        extreme_long_ratio: float = 2.5,
        extreme_short_ratio: float = 0.6,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        self.extreme_long_ratio = extreme_long_ratio
        self.extreme_short_ratio = extreme_short_ratio

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="ls_crowded",
            display_name="L/S Ratio Crowded",
            summary="Fades retail extremes based on Global Long/Short Account Ratio",
            default_params={
                "extreme_long_ratio": 2.5,
                "extreme_short_ratio": 0.6,
            },
        )

    def required_indicators(self) -> tuple:
        return ()

    def evaluate(self, state: MarketState) -> Signal | None:
        hmc = state.historical_context
        if not hmc or not hmc.ls_ratio_history or hmc.ls_ratio_history.is_empty:
            return None

        latest = hmc.ls_ratio_history.latest()
        if not latest:
            return None

        ratio = latest.ls_ratio

        # If ratio > extreme_long_ratio, crowd is heavily long -> signal short
        if ratio > self.extreme_long_ratio:
            direction = -1.0
            # Scale linearly past the threshold up to max 4.0
            magnitude = min(1.0, (ratio - self.extreme_long_ratio) / 1.5)
        # If ratio < extreme_short_ratio, crowd is heavily short -> signal long
        elif ratio < self.extreme_short_ratio:
            direction = 1.0
            # Scale linearly past the threshold down to min 0.2
            magnitude = min(1.0, (self.extreme_short_ratio - ratio) / 0.4)
        else:
            return None

        # Add a small buffer to confidence so we don't return 0.0 confidence immediately
        confidence = min(1.0, magnitude + 0.2)

        return Signal(
            name=self.name,
            weight=self.weight,
            value=direction * magnitude,
            confidence=confidence,
            description=f"Extreme L/S Ratio: {ratio:.2f} (crowd is {'long' if direction < 0 else 'short'})",
            evidence=(
                EvidenceItem("ls_ratio", f"{ratio:.2f}"),
                EvidenceItem("long_pct", f"{latest.long_pct*100:.1f}%"),
                EvidenceItem("short_pct", f"{latest.short_pct*100:.1f}%"),
            ),
        )


@RuleRegistry.register("taker_flow_imbalance")
class TakerFlowImbalanceRule(FactorRule):
    """Detects sustained taker buying or selling pressure."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.20,
        description: str | None = None,
        window_size: int = 12, # 1 hour at 5m resolution
        imbalance_threshold: float = 0.15, # Net volume is 15% of total volume
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        self.window_size = window_size
        self.imbalance_threshold = imbalance_threshold

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="taker_flow_imbalance",
            display_name="Taker Flow Imbalance",
            summary="Detects sustained aggressive buying or selling",
            default_params={
                "window_size": 12,
                "imbalance_threshold": 0.15,
            },
        )

    def required_indicators(self) -> tuple:
        return ()

    def evaluate(self, state: MarketState) -> Signal | None:
        hmc = state.historical_context
        if not hmc or not hmc.taker_flow_history or hmc.taker_flow_history.is_empty:
            return None

        series = hmc.taker_flow_history.window(self.window_size)
        if len(series) < 2:
            return None

        total_buy = sum(ctx.buy_volume for ctx in series)
        total_sell = sum(ctx.sell_volume for ctx in series)
        total_vol = total_buy + total_sell

        if total_vol <= 0:
            return None

        net_buy = total_buy - total_sell
        imbalance = net_buy / total_vol

        if abs(imbalance) < self.imbalance_threshold:
            return None

        direction = 1.0 if imbalance > 0 else -1.0
        magnitude = min(1.0, abs(imbalance) / (self.imbalance_threshold * 2.5))

        return Signal(
            name=self.name,
            weight=self.weight,
            value=direction * magnitude,
            confidence=magnitude,
            description=f"Taker flow imbalance: {imbalance*100:+.1f}% over {len(series)} periods",
            evidence=(
                EvidenceItem("imbalance_pct", f"{imbalance*100:.1f}%"),
                EvidenceItem("net_buy_vol", f"{net_buy:.2f}"),
                EvidenceItem("total_vol", f"{total_vol:.2f}"),
            ),
        )


@RuleRegistry.register("liquidation_cascade")
class LiquidationCascadeRule(FactorRule):
    """Contrarian signal when mass liquidations occur (capitulation)."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.25,
        description: str | None = None,
        window_size: int = 3, # Recent 15 mins at 5m resolution
        cascade_threshold_usd: float = 5_000_000.0, # $5m threshold
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        self.window_size = window_size
        self.cascade_threshold_usd = cascade_threshold_usd

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="liquidation_cascade",
            display_name="Liquidation Cascade",
            summary="Detects capitulation via mass liquidations (strong contrarian signal)",
            default_params={
                "window_size": 3,
                "cascade_threshold_usd": 5000000.0,
            },
        )

    def required_indicators(self) -> tuple:
        return ()

    def evaluate(self, state: MarketState) -> Signal | None:
        hmc = state.historical_context
        # Depending on whether liquidations_history exists (currently not implemented in BinanceProvider)
        if not hmc or not hmc.liquidations_history or hmc.liquidations_history.is_empty:
            return None

        series = hmc.liquidations_history.window(self.window_size)
        if len(series) == 0:
            return None

        # In liquidation cascade, we look for massive liquidations on one side
        # `long_liquidations` means longs were forced to sell (price dropped)
        # `short_liquidations` means shorts were forced to buy (price surged)
        total_long_liq = sum(getattr(ctx, "long_liquidations_usd", 0.0) for ctx in series)
        total_short_liq = sum(getattr(ctx, "short_liquidations_usd", 0.0) for ctx in series)

        # If long liquidations > threshold, market flushed longs -> buy signal
        # If short liquidations > threshold, market flushed shorts -> sell signal
        if total_long_liq > self.cascade_threshold_usd and total_long_liq > total_short_liq * 3:
            direction = 1.0
            magnitude = min(1.0, total_long_liq / (self.cascade_threshold_usd * 3))
            liq_val = total_long_liq
            side = "longs"
        elif total_short_liq > self.cascade_threshold_usd and total_short_liq > total_long_liq * 3:
            direction = -1.0
            magnitude = min(1.0, total_short_liq / (self.cascade_threshold_usd * 3))
            liq_val = total_short_liq
            side = "shorts"
        else:
            return None

        # High confidence for liquidation cascades
        confidence = min(1.0, magnitude + 0.3)

        return Signal(
            name=self.name,
            weight=self.weight,
            value=direction * magnitude,
            confidence=confidence,
            description=f"Massive {side} liquidation cascade (${liq_val/1e6:.1f}M)",
            evidence=(
                EvidenceItem("long_liq_usd", f"${total_long_liq:,.0f}"),
                EvidenceItem("short_liq_usd", f"${total_short_liq:,.0f}"),
            ),
        )
