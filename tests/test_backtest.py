"""Tests for backtest value objects and the WalkForwardBacktester."""

from __future__ import annotations

from datetime import date

import pytest

from neon_radar.application.services.backtester import WalkForwardBacktester
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.models import KlineSeries, Symbol
from neon_radar.domain.scoring.backtest import (
    BacktestConfig,
    BacktestResult,
    ConfidenceCalibration,
    EvaluationResult,
)
from neon_radar.domain.scoring.factor_rule import FactorRule, RuleDescription
from neon_radar.domain.scoring.value_objects import Signal
from tests.conftest import make_series

# ---------------------------------------------------------------------------
# Fake exchange that returns pre-canned series.
# ---------------------------------------------------------------------------


class FakeExchange:
    """Minimal exchange stub for backtesting — returns the same series every time."""

    def __init__(self, series: KlineSeries) -> None:
        self._series = series

    async def get_klines(self, symbol, timeframe, *, end_time=None, limit=500):
        # Return the full series; the backtester slices in memory.
        return self._series


# ---------------------------------------------------------------------------
# Rules with deterministic behaviour
# ---------------------------------------------------------------------------


class _AlwaysUpRule(FactorRule):
    """Always returns +1.0 with full confidence — a perfect oracle."""

    NAME = "always_up"

    @classmethod
    def description(cls):  # type: ignore[override]
        return RuleDescription(
            name="always_up", display_name="AU", summary="always up", default_params={}
        )

    def evaluate(self, state):  # type: ignore[override]
        return Signal(
            name=self.name,
            weight=self.weight,
            value=1.0,
            confidence=self.weight,
            description="always up",
        )


class _RandomRule(FactorRule):
    """Returns alternating +1 / -1 each day — should hit ~50%."""

    NAME = "random"
    _counter = 0

    @classmethod
    def description(cls):  # type: ignore[override]
        return RuleDescription(
            name="random", display_name="R", summary="alternates", default_params={}
        )

    def evaluate(self, state):  # type: ignore[override]
        _RandomRule._counter += 1
        v = 1.0 if _RandomRule._counter % 2 == 0 else -1.0
        return Signal(
            name=self.name,
            weight=self.weight,
            value=v,
            confidence=1.0,
            description="alternating",
        )


def _scoring_config_with(rules) -> ScoringRulesConfig:
    """Build a minimal ScoringRulesConfig — for type-level compatibility."""
    return ScoringRulesConfig.model_validate(
        {
            "rules": [
                {
                    "name": rule.NAME,
                    "enabled": True,
                    "weight": rule.weight,
                    "params": {},
                }
                for rule in rules
            ],
            "min_confidence": 0.0,
        }
    )


def _trending_series(
    symbol: str, n: int = 100, start_price: float = 100.0, daily_step: float = 1.0
) -> KlineSeries:
    """Build a strictly monotonically rising series (always_up test)."""
    return make_series(
        [start_price + i * daily_step for i in range(n)],
        symbol=symbol,
    )


def _alternating_series(symbol: str, n: int = 100, base: float = 100.0) -> KlineSeries:
    """Build an oscillating series (alternating_random test)."""
    closes = [base + (1.0 if i % 2 == 0 else -1.0) for i in range(n)]
    return make_series(closes, symbol=symbol)


@pytest.fixture(autouse=True)
def _reset_random_counter():
    _RandomRule._counter = 0


class TestWalkForwardBacktester:
    @pytest.mark.asyncio
    async def test_perfect_oracle_has_high_hit_rate(self) -> None:
        """``AlwaysUpRule`` on a strictly rising series should be near 100%."""
        symbol = Symbol("BTCUSDT")
        series = _trending_series("BTCUSDT", n=100, start_price=100.0, daily_step=1.0)
        exchange = FakeExchange(series)
        rules = (_AlwaysUpRule(weight=0.5),)
        scoring_cfg = _scoring_config_with(rules)
        backtester = WalkForwardBacktester(
            exchange=exchange, scoring_config=scoring_cfg, rules=rules
        )

        start = date(2024, 1, 10)
        end = date(2024, 1, 20)  # 11 days
        result = await backtester.run(
            start_date=start,
            end_date=end,
            symbols=(symbol,),
            timeframe="1d",
            horizons=(1,),
            min_history_candles=50,
        )
        assert result.n_evaluations > 0
        assert result.hit_rate(1) > 0.9

    @pytest.mark.asyncio
    async def test_random_rule_hits_around_50_percent(self) -> None:
        """Alternating signals on strictly rising prices → ~0% hit rate.

        The rule alternates +1 / -1 starting from -1 on the first call.
        On a strictly rising series this should produce poor results
        because every other prediction is bearish but the price goes up.
        """
        symbol = Symbol("BTCUSDT")
        # Strictly rising — oracle would hit 100%, alternating rule should hit 0%.
        series = _trending_series("BTCUSDT", n=100, start_price=100.0, daily_step=1.0)
        exchange = FakeExchange(series)
        rules = (_RandomRule(weight=0.5),)
        scoring_cfg = _scoring_config_with(rules)
        backtester = WalkForwardBacktester(
            exchange=exchange, scoring_config=scoring_cfg, rules=rules
        )

        start = date(2024, 1, 10)
        end = date(2024, 1, 30)  # 21 days
        result = await backtester.run(
            start_date=start,
            end_date=end,
            symbols=(symbol,),
            timeframe="1d",
            horizons=(1,),
            min_history_candles=50,
        )
        # Random rule on rising series — most predictions wrong.
        hr = result.hit_rate(1)
        assert hr < 0.5  # significantly worse than oracle (0.9+)

    @pytest.mark.asyncio
    async def test_empty_window_returns_empty_result(self) -> None:
        symbol = Symbol("BTCUSDT")
        series = _trending_series("BTCUSDT", n=100)
        exchange = FakeExchange(series)
        rules = (_AlwaysUpRule(weight=0.5),)
        scoring_cfg = _scoring_config_with(rules)
        backtester = WalkForwardBacktester(
            exchange=exchange, scoring_config=scoring_cfg, rules=rules
        )

        result = await backtester.run(
            start_date=date(2024, 1, 1),
            end_date=date(2023, 12, 31),  # end < start
            symbols=(symbol,),
            timeframe="1d",
            horizons=(1,),
            min_history_candles=50,
        )
        assert result.n_evaluations == 0

    @pytest.mark.asyncio
    async def test_correlation_matrix_is_symmetric(self) -> None:
        symbol = Symbol("BTCUSDT")
        series = _trending_series("BTCUSDT", n=100)
        exchange = FakeExchange(series)
        rules = (_AlwaysUpRule(weight=0.5), _RandomRule(weight=0.5))
        scoring_cfg = _scoring_config_with(rules)
        backtester = WalkForwardBacktester(
            exchange=exchange, scoring_config=scoring_cfg, rules=rules
        )
        result = await backtester.run(
            start_date=date(2024, 1, 10),
            end_date=date(2024, 1, 20),
            symbols=(symbol,),
            timeframe="1d",
            horizons=(1,),
            min_history_candles=50,
        )
        if result.correlation is None:
            pytest.skip("no correlation data")
        m = result.correlation.matrix
        n = len(m)
        for i in range(n):
            for j in range(n):
                assert m[i][j] == pytest.approx(m[j][i], abs=1e-9)
            assert m[i][i] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_requires_rules_explicitly(self) -> None:
        """Passing only config without pre-built rules is an error."""
        exchange = FakeExchange(_trending_series("BTCUSDT", n=10))
        with pytest.raises(ValueError, match="pre-built rule"):
            WalkForwardBacktester(
                exchange=exchange,
                scoring_config=ScoringRulesConfig.model_validate(
                    {"rules": [], "min_confidence": 0.0}
                ),
            )


# ---------------------------------------------------------------------------
# Backtest value objects
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def test_direction(self) -> None:
        e = _mk_eval(score=0.5, actual_return=0.01)
        assert e.direction == 1
        assert e.hit is True

        e = _mk_eval(score=-0.5, actual_return=-0.01)
        assert e.direction == -1
        assert e.hit is True

        e = _mk_eval(score=-0.5, actual_return=0.01)
        assert e.direction == -1
        assert e.hit is False

    def test_neutral_direction(self) -> None:
        e = _mk_eval(score=0.0, actual_return=0.05)
        assert e.direction == 0
        assert e.hit is None

    def test_actual_return(self) -> None:
        e = _mk_eval(score=0.1, price_at=100, price_after=110)
        assert e.actual_return_pct == pytest.approx(0.1)


def _mk_eval(
    score: float,
    actual_return: float = 0.01,
    price_at: float = 100.0,
    price_after: float | None = None,
) -> EvaluationResult:
    if price_after is None:
        price_after = price_at * (1 + actual_return)
    return EvaluationResult(
        symbol=Symbol("BTCUSDT"),
        day=date(2024, 1, 1),
        score_value=score,
        confidence=0.5,
        price_at_signal=price_at,
        price_after_horizon=price_after,
        horizon_days=1,
        rule_values=(("a", score),),
    )


class TestConfidenceCalibration:
    def test_from_pairs(self) -> None:
        cal = ConfidenceCalibration.from_pairs(
            [
                (0.0, 0.3, 5, 10),
                (0.3, 0.5, 12, 20),
                (0.5, 0.7, 14, 20),
                (0.7, 1.0, 18, 20),
            ]
        )
        rates = [b[2] for b in cal.buckets]
        assert rates == [0.5, 0.6, 0.7, 0.9]

    def test_empty_bucket(self) -> None:
        cal = ConfidenceCalibration.from_pairs([(0.0, 0.3, 0, 0)])
        assert cal.buckets[0][2] == 0.0


class TestBacktestSummary:
    def test_summary_mentions_hit_rate_and_returns(self) -> None:
        result = BacktestResult(
            config=BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 10),
                timeframe="1d",
                symbols=("BTCUSDT",),
                horizons=(1,),
            ),
            n_evaluations=10,
            overall_hit_rate={1: 0.75},
            overall_avg_return_long=0.03,
            overall_avg_return_short=-0.02,
            overall_n_long=4,
            overall_n_short=3,
        )

        assert "1d hit rate" in result.summary
        assert "75.0%" in result.summary
        assert "long" in result.summary.lower()
        assert "short" in result.summary.lower()

    def test_summary_for_empty_result(self) -> None:
        result = BacktestResult(
            config=BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 10),
                timeframe="1d",
                symbols=("BTCUSDT",),
                horizons=(1,),
            ),
            n_evaluations=0,
            overall_hit_rate={},
        )

        assert result.summary == "No evaluations produced."
