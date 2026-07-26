"""Trading Pipeline Orchestrator.

Combines the Analysis Layer (Rule Engine, Trade Setup Engine) with the Risk Layer 
(Risk Manager, Position Sizing Engine) into a single, cohesive decision-making pipeline.
"""

from collections.abc import Iterable

from neon_radar.application.services.analysis import analyze_series
from neon_radar.application.services.indicator_pipeline import IndicatorSpec
from neon_radar.application.services.risk.manager import RiskManager
from neon_radar.application.services.risk.sizing import PositionSizingEngine
from neon_radar.domain.funding import FundingRate, OpenInterest
from neon_radar.domain.market_context import MarketContext
from neon_radar.domain.models import KlineSeries, TickerStats
from neon_radar.domain.portfolio import PortfolioState
from neon_radar.domain.risk import DrawdownState
from neon_radar.domain.scoring.factor_rule import FactorRule
from neon_radar.domain.trading.regime import RegimeClassifier, RegimeFilterConfig
from neon_radar.domain.trading.setup import FinalTradeSetup, TradeSetupEngine


class TradingPipeline:
    """Orchestrates the entire trade decision cycle from raw data to a FinalTradeSetup."""

    def __init__(
        self,
        rules: Iterable[FactorRule],
        setup_engine: TradeSetupEngine,
        risk_manager: RiskManager,
        sizing_engine: PositionSizingEngine,
        min_confidence: float = 0.0,
        confluence_bonus: float = 0.20,
        confluence_penalty: float = 0.15,
        max_confidence_boost: float = 0.40,
        regime_classifier: RegimeClassifier | None = None,
        regime_config: RegimeFilterConfig | None = None,
    ) -> None:
        self.rules = tuple(rules)
        self.setup_engine = setup_engine
        self.risk_manager = risk_manager
        self.sizing_engine = sizing_engine
        self.min_confidence = min_confidence
        self.confluence_bonus = confluence_bonus
        self.confluence_penalty = confluence_penalty
        self.max_confidence_boost = max_confidence_boost
        self.regime_classifier = regime_classifier
        self.regime_config = regime_config

    def required_indicators(self) -> tuple[IndicatorSpec, ...]:
        """Aggregate all indicators required by rules and the setup engine."""
        spec_map: dict[str, IndicatorSpec] = {}
        for rule in self.rules:
            for spec in rule.required_indicators():
                spec_map.setdefault(spec.series_name, spec)
        for spec in self.setup_engine.required_indicators():
            spec_map.setdefault(spec.series_name, spec)
        return tuple(spec_map.values())

    def evaluate(
        self,
        series: KlineSeries,
        portfolio: PortfolioState,
        drawdown: DrawdownState | None = None,
        timestamp: int | None = None,
        higher_tf_series: KlineSeries | None = None,
        ticker: TickerStats | None = None,
        funding_rate: FundingRate | None = None,
        open_interest: OpenInterest | None = None,
        market_context: MarketContext | None = None,
    ) -> FinalTradeSetup | None:
        """Run the full pipeline on a series and portfolio state."""

        # 1. Analyze Market Data -> AnalysisResult
        # We pass setup_engine indicators via extra_indicators so they are available in MarketState
        extra_indicators = self.setup_engine.required_indicators()

        analysis_result = analyze_series(
            series=series,
            rules=self.rules,
            min_confidence=self.min_confidence,
            confluence_bonus=self.confluence_bonus,
            confluence_penalty=self.confluence_penalty,
            max_confidence_boost=self.max_confidence_boost,
            timestamp=timestamp,
            higher_tf_series=higher_tf_series,
            ticker=ticker,
            funding_rate=funding_rate,
            open_interest=open_interest,
            market_context=market_context,
            extra_indicators=extra_indicators,
            regime_classifier=self.regime_classifier,
            regime_config=self.regime_config,
        )

        # 2. Build TradeSetup (Analysis Layer output)
        # Ensure we have a market state and a valid setup before proceeding
        if analysis_result.market_state is None:
            return None

        trade_setup = self.setup_engine.build_setup(
            analysis_result.market_state, analysis_result
        )
        if trade_setup is None:
            return None

        # 3. Evaluate Risk Limits -> RiskDecision
        risk_decision = self.risk_manager.evaluate(
            analysis_result, portfolio, drawdown
        )
        if not risk_decision.is_allowed:
            return None

        # 4. Position Sizing
        sized_setup = self.sizing_engine.build_sized_setup(trade_setup, risk_decision)
        if sized_setup is None:
            return None

        # 5. Compile FinalTradeSetup
        expected_reward = (
            abs(trade_setup.entry_price - trade_setup.take_profit_1) * sized_setup.base_size
        )
        risk_amount = (
            abs(trade_setup.entry_price - trade_setup.stop_loss) * sized_setup.base_size
        )

        return FinalTradeSetup(
            symbol=analysis_result.market_state.primary_series.symbol,
            direction=trade_setup.direction,
            entry=trade_setup.entry_price,
            stop_loss=trade_setup.stop_loss,
            take_profit=trade_setup.take_profit_1,  # MVP: target TP1
            confidence=analysis_result.score.confidence,
            score=analysis_result.score.value,
            risk_decision=risk_decision,
            position_size=sized_setup.base_size,
            quote_size=sized_setup.quote_size,
            risk_amount=risk_amount,
            expected_reward=expected_reward,
            diagnostics=trade_setup.diagnostics,
        )
