"""MACD momentum rule.

This is a deliberately simple first version. It uses the latest available
MACD, signal, and histogram values only.

* MACD line crosses above signal line -> bullish (+1, high confidence)
* MACD line crosses below signal line -> bearish (-1, high confidence)
* If the signal is unclear (e.g. no data, value missing, or line is flat)
  -> return ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.registry import RuleRegistry
from neon_radar.domain.scoring.value_objects import EvidenceItem, Signal

if TYPE_CHECKING:
    from neon_radar.domain.market_state import MarketState


@RuleRegistry.register("macd_momentum")
class MACDMomentumRule(FactorRule):
    """Simple MACD cross rule."""

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 0.25,
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, weight=weight, description=description)

    @classmethod
    def description(cls) -> RuleDescription:
        return RuleDescription(
            name="macd_momentum",
            display_name="MACD Momentum",
            summary="Bullish when MACD crosses above signal; bearish when it crosses below",
            default_params={},
        )

    def required_indicators(self) -> tuple:
        # Lazy import to avoid circular dependency at module load.
        from neon_radar.application.services.indicator_pipeline import IndicatorSpec

        return (
            IndicatorSpec(name="macd", params={}, tag=""),
        )

    def evaluate(self, state: MarketState) -> Signal | None:
        macd_series = state.get_indicator("macd")
        if macd_series is None:
            return None

        latest = macd_series.latest()
        if latest is None:
            return None

        macd = latest.get("macd")
        signal = latest.get("signal")
        histogram = latest.get("histogram")

        if macd is None or signal is None or histogram is None:
            return None
        if macd != macd or signal != signal or histogram != histogram:
            return None

        prev = macd_series.snapshots[-2] if len(macd_series.snapshots) >= 2 else None
        if prev is None:
            return None

        prev_macd = prev.get("macd")
        prev_signal = prev.get("signal")
        prev_histogram = prev.get("histogram")
        if prev_macd is None or prev_signal is None or prev_histogram is None:
            return None
        if prev_macd != prev_macd or prev_signal != prev_signal or prev_histogram != prev_histogram:
            return None

        if histogram == 0.0 or prev_histogram == 0.0:
            return None

        if prev_macd <= prev_signal and macd > signal:
            return Signal(
                name=self.name,
                weight=self.weight,
                value=1.0,
                confidence=0.8,
                description="MACD crossed above signal line",
                evidence=(
                    EvidenceItem("macd", f"{macd:.4f}"),
                    EvidenceItem("signal", f"{signal:.4f}"),
                    EvidenceItem("histogram", f"{histogram:.4f}"),
                ),
            )

        if prev_macd >= prev_signal and macd < signal:
            return Signal(
                name=self.name,
                weight=self.weight,
                value=-1.0,
                confidence=0.8,
                description="MACD crossed below signal line",
                evidence=(
                    EvidenceItem("macd", f"{macd:.4f}"),
                    EvidenceItem("signal", f"{signal:.4f}"),
                    EvidenceItem("histogram", f"{histogram:.4f}"),
                ),
            )

        return None
