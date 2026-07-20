"""Regime Classifier implementation."""

from __future__ import annotations

from neon_radar.application.services.indicator_pipeline import IndicatorSpec
from neon_radar.domain.market_state import MarketState
from neon_radar.domain.trading.regime import MarketRegime, RegimeClassification, RegimeFilterConfig


class RuleBasedRegimeClassifier:
    """Classifies market regime based on configurable rules."""

    def __init__(self, config: RegimeFilterConfig) -> None:
        self._config = config

    def required_indicators(self) -> tuple[IndicatorSpec, ...]:
        """Indicators required to classify the regime."""
        if not self._config.enabled:
            return ()

        return (
            IndicatorSpec(
                name="adx",
                params={"period": self._config.adx_period},
                tag=str(self._config.adx_period),
            ),
            IndicatorSpec(
                name="ema",
                params={"period": self._config.ema_fast_period},
                tag=str(self._config.ema_fast_period),
            ),
            IndicatorSpec(
                name="ema",
                params={"period": self._config.ema_slow_period},
                tag=str(self._config.ema_slow_period),
            ),
            IndicatorSpec(
                name="atr",
                params={"period": self._config.atr_period},
                tag=str(self._config.atr_period),
            ),
        )

    def classify(self, state: MarketState) -> RegimeClassification:
        """Evaluate the current market state and classify the regime."""
        if not self._config.enabled:
            return RegimeClassification(regime=MarketRegime.UNKNOWN, reason="Regime filter disabled")

        # 1. Volatility Crash Check (Priority 1)
        atr_val = state.get_indicator_value(f"atr_{self._config.atr_period}")
        latest = state.primary_series.latest()

        if atr_val is not None and latest is not None and latest.close > 0:
            atr_pct = atr_val / latest.close
            if atr_pct > self._config.atr_crash_threshold_pct:
                return RegimeClassification(
                    regime=MarketRegime.VOLATILE_CRASH,
                    reason=f"ATR pct {atr_pct*100:.2f}% > {self._config.atr_crash_threshold_pct*100:.2f}%"
                )

        # 2. ADX Chop Check (Priority 2)
        adx_val = state.get_indicator_value(f"adx_{self._config.adx_period}", "adx")
        if adx_val is not None:
            if adx_val < self._config.adx_chop_threshold:
                return RegimeClassification(
                    regime=MarketRegime.CHOP,
                    reason=f"ADX {adx_val:.2f} < {self._config.adx_chop_threshold:.2f}"
                )

        # 3. EMA Trend Check (Priority 3)
        ema_fast = state.get_indicator_value(f"ema_{self._config.ema_fast_period}")
        ema_slow = state.get_indicator_value(f"ema_{self._config.ema_slow_period}")

        if ema_fast is not None and ema_slow is not None:
            if ema_fast > ema_slow:
                reason = f"EMA({self._config.ema_fast_period}) > EMA({self._config.ema_slow_period})"
                if adx_val is not None:
                    reason += f" and ADX {adx_val:.2f} >= {self._config.adx_chop_threshold:.2f}"
                return RegimeClassification(
                    regime=MarketRegime.BULL_TREND,
                    reason=reason
                )
            else:
                reason = f"EMA({self._config.ema_fast_period}) < EMA({self._config.ema_slow_period})"
                if adx_val is not None:
                    reason += f" and ADX {adx_val:.2f} >= {self._config.adx_chop_threshold:.2f}"
                return RegimeClassification(
                    regime=MarketRegime.BEAR_TREND,
                    reason=reason
                )

        # Fallback
        return RegimeClassification(
            regime=MarketRegime.UNKNOWN,
            reason="Missing indicator data"
        )
