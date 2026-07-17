"""Trading Setup Engine models and logic.

Provides the foundational architecture for converting an AnalysisResult
and MarketState into an actionable TradeSetup recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neon_radar.application.services.indicator_pipeline import IndicatorSpec
from neon_radar.domain.enums import Bias

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState
    from neon_radar.domain.scoring.value_objects import Score


@dataclass(slots=True, frozen=True)
class TradeSetup:
    """An actionable trade recommendation based on score and market state."""

    direction: Bias
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward: tuple[float, float]

    def __post_init__(self) -> None:
        if self.direction == Bias.NEUTRAL:
            raise ValueError("TradeSetup direction cannot be NEUTRAL")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be > 0")
        if self.stop_loss <= 0:
            raise ValueError("stop_loss must be > 0")
        if self.take_profit_1 <= 0 or self.take_profit_2 <= 0:
            raise ValueError("take_profits must be > 0")

        # Verify ordering based on direction
        if self.direction == Bias.BULLISH:
            if not (self.stop_loss < self.entry_price < self.take_profit_1 < self.take_profit_2):
                raise ValueError("BULLISH setup must have SL < Entry < TP1 < TP2")
        else:
            if not (self.stop_loss > self.entry_price > self.take_profit_1 > self.take_profit_2):
                raise ValueError("BEARISH setup must have SL > Entry > TP1 > TP2")


class TradeSetupEngine:
    """Generates a TradeSetup recommendation from market state and score.

    This is an independent service that does not compute indicators itself.
    It expects the necessary indicators (e.g., ATR) to be present in the
    MarketState. The pipeline is responsible for querying `required_indicators()`
    before computing the MarketState.

    Confidence Integration
    ----------------------
    The engine requires the aggregated `Score.confidence` to be greater than
    or equal to `min_confidence` to generate a trade.
    
    * **Як формується confidence:** Кожне правило (FactorRule) повертає власний `confidence` [0.0, 1.0]. `ScoringAggregator` розраховує середньозважений `confidence` усього скорингу, використовуючи вагу правил (weight).
    * **Діапазон:** [0.0, 1.0]. Де 0.0 — повна невпевненість, 1.0 — абсолютна впевненість.
    * **Чому цей поріг (наприклад, 0.5):** Базовий поріг 0.5 є логічним рубіконом — якщо зважена впевненість системи падає нижче 50%, торговий сигнал вважається шумовим і відхиляється. Це захищає від торгівлі в періоди хаосу (що детектує `volatility_filter`).
    * **Оптимізація:** У майбутніх спринтах `min_confidence` може оптимізуватися за допомогою Walk-Forward Validation під конкретний ринковий режим або актив.
    """

    def __init__(
        self,
        atr_period: int = 14,
        sl_multiplier: float = 1.5,
        tp1_rr: float = 1.5,
        tp2_rr: float = 3.0,
        min_confidence: float = 0.5,
    ) -> None:
        self.atr_period = atr_period
        self.sl_multiplier = sl_multiplier
        self.tp1_rr = tp1_rr
        self.tp2_rr = tp2_rr
        self.min_confidence = min_confidence

    def required_indicators(self) -> tuple[IndicatorSpec, ...]:
        """Indicators required by this engine to formulate a setup."""
        return (
            IndicatorSpec(
                name="atr",
                params={"period": self.atr_period},
                tag=str(self.atr_period),
            ),
        )

    def build_setup(self, state: MarketState, score: Score) -> TradeSetup | None:
        """Formulate a TradeSetup based on the given score and state."""
        if score.bias == Bias.NEUTRAL:
            return None

        if score.confidence < self.min_confidence:
            return None

        latest = state.primary_series.latest()
        if latest is None:
            return None

        atr_series = state.get_indicator(f"atr_{self.atr_period}")
        if not atr_series or len(atr_series.snapshots) == 0:
            return None

        import math

        atr_val = atr_series.snapshots[-1].get("atr")
        if atr_val is None or math.isnan(atr_val) or atr_val <= 0:
            return None

        entry = latest.close
        risk = self.sl_multiplier * atr_val

        # Safety check: avoid negative prices
        if risk >= entry:
            return None

        if score.bias == Bias.BULLISH:
            sl = entry - risk
            tp1 = entry + (risk * self.tp1_rr)
            tp2 = entry + (risk * self.tp2_rr)
        else:
            sl = entry + risk
            tp1 = entry - (risk * self.tp1_rr)
            tp2 = entry - (risk * self.tp2_rr)

        # Safety check: avoid negative prices for TPs in bearish
        if tp2 <= 0:
            return None

        return TradeSetup(
            direction=score.bias,
            entry_price=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward=(self.tp1_rr, self.tp2_rr),
        )
