"""Open Interest context rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("oi_expansion")
class OpenInterestExpansionRule(FactorRule):
    """Detects strong trend confirmation via Open Interest expansion.
    
    If price is moving strongly (determined by close vs open over the window) 
    and OI is expanding significantly, it confirms the trend strength.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.20,
        description: str | None = None,
        window_size: int = 6,  # 30 mins at 5m
        oi_expansion_threshold: float = 0.02, # 2% expansion
        price_move_threshold: float = 0.005,  # 0.5% move
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        self.window_size = window_size
        self.oi_expansion_threshold = oi_expansion_threshold
        self.price_move_threshold = price_move_threshold

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="oi_expansion",
            display_name="OI Expansion",
            summary="Trend confirmation when OI increases alongside price movement",
            default_params={
                "window_size": 6,
                "oi_expansion_threshold": 0.02,
                "price_move_threshold": 0.005,
            },
        )

    def required_indicators(self) -> tuple:
        return ()

    def evaluate(self, state: MarketState) -> Signal | None:
        hmc = state.historical_context
        if not hmc or not hmc.open_interest_history or hmc.open_interest_history.is_empty:
            return None

        series = hmc.open_interest_history.window(self.window_size)
        if len(series) < 2:
            return None

        first_oi = series[0].oi_coin
        last_oi = series[-1].oi_coin
        if first_oi <= 0:
            return None

        oi_change = (last_oi - first_oi) / first_oi

        # We only care about expansion here
        if oi_change < self.oi_expansion_threshold:
            return None

        # Check price movement over the same window roughly
        # We use the primary series (candles)
        candles = state.primary_series.candles
        if len(candles) < 2:
            return None

        # If the window_size is for 5m intervals, and primary series is e.g. 5m, we can use it.
        # But we don't know the primary series timeframe here. So we just look at the latest close
        # vs the close N bars ago in the primary series that roughly matches our time window.
        # For simplicity, let's just check the last candle or two in the primary series.
        # Actually, a better way is to compare current close to close at series[0].time_context.publish_time
        # But `candles` doesn't provide easy time lookup.
        # Let's use the last N candles where N = window_size, assuming they roughly align,
        # OR just use the latest candle's momentum.
        n_candles = min(len(candles), self.window_size)
        first_price = candles[-n_candles].open
        last_price = candles[-1].close
        price_change = (last_price - first_price) / first_price

        if abs(price_change) < self.price_move_threshold:
            return None

        direction = 1.0 if price_change > 0 else -1.0
        magnitude = min(1.0, oi_change / (self.oi_expansion_threshold * 3))

        return Signal(
            name=self.name,
            weight=self.weight,
            value=direction * magnitude,
            confidence=magnitude,
            description=f"OI expanded by {oi_change*100:.2f}% confirming {'uptrend' if direction > 0 else 'downtrend'}",
            evidence=(
                EvidenceItem("oi_change", f"{oi_change*100:.2f}%"),
                EvidenceItem("price_change", f"{price_change*100:.2f}%"),
            ),
        )


@RuleRegistry.register("oi_divergence")
class OpenInterestDivergenceRule(FactorRule):
    """Detects trend weakness when price moves but OI drops (divergence)."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.15,
        description: str | None = None,
        window_size: int = 6,
        oi_drop_threshold: float = -0.015, # 1.5% drop
        price_move_threshold: float = 0.005,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)
        self.window_size = window_size
        self.oi_drop_threshold = oi_drop_threshold
        self.price_move_threshold = price_move_threshold

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="oi_divergence",
            display_name="OI Divergence",
            summary="Reversal signal when price moves but OI decreases (short covering or long taking profit)",
            default_params={
                "window_size": 6,
                "oi_drop_threshold": -0.015,
                "price_move_threshold": 0.005,
            },
        )

    def required_indicators(self) -> tuple:
        return ()

    def evaluate(self, state: MarketState) -> Signal | None:
        hmc = state.historical_context
        if not hmc or not hmc.open_interest_history or hmc.open_interest_history.is_empty:
            return None

        series = hmc.open_interest_history.window(self.window_size)
        if len(series) < 2:
            return None

        first_oi = series[0].oi_coin
        last_oi = series[-1].oi_coin
        if first_oi <= 0:
            return None

        oi_change = (last_oi - first_oi) / first_oi

        # We only care about OI dropping
        if oi_change > self.oi_drop_threshold:
            return None

        candles = state.primary_series.candles
        if len(candles) < 2:
            return None

        n_candles = min(len(candles), self.window_size)
        first_price = candles[-n_candles].open
        last_price = candles[-1].close
        price_change = (last_price - first_price) / first_price

        if abs(price_change) < self.price_move_threshold:
            return None

        # Reversal signal: price up + OI down = short covering (bearish/weak bullish) -> signal short
        # Price down + OI down = long liquidation (bullish/weak bearish) -> signal long
        direction = -1.0 if price_change > 0 else 1.0
        magnitude = min(1.0, abs(oi_change) / abs(self.oi_drop_threshold * 2))

        return Signal(
            name=self.name,
            weight=self.weight,
            value=direction * magnitude,
            confidence=magnitude,
            description=f"OI dropped by {abs(oi_change)*100:.2f}% during price move, signaling reversal",
            evidence=(
                EvidenceItem("oi_change", f"{oi_change*100:.2f}%"),
                EvidenceItem("price_change", f"{price_change*100:.2f}%"),
            ),
        )
