"""Walk-forward backtester.

Runs the scoring engine against historical data, then computes
objective metrics that answer:

* Did the signals predict direction correctly?
* Are certain rules redundant (correlated with others)?
* Is confidence calibrated (high confidence → high hit rate)?
* What weights would the metrics suggest?

Design notes
------------
* The backtester **pre-fetches** all klines for each symbol once, then
  slices them in memory for each evaluation day. This avoids the
  N+1 API request problem.
* For each evaluation day T (one per day in the backtest window):
    - Take the slice of klines up to T (excluding T itself).
    - Compute indicators + score.
    - Record the result.
* Outcomes are computed from the same pre-fetched klines: the close
  at T + horizon days (or the last available close if the backtest
  window ends earlier).
* The rule set is loaded from ``ScoringRulesConfig`` exactly as
  ``neon-radar scan`` does — so what you backtest is what you ship.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from neon_radar.application.services.analysis import analyze_series
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.scoring import (
    AnalysisResult,
    EvaluationResult,
    RuleBasedEngine,
)
from neon_radar.domain.scoring.backtest import (
    BacktestConfig,
    BacktestResult,
    ConfidenceCalibration,
    CorrelationMatrix,
    RuleMetrics,
    SymbolBacktestResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neon_radar.config.scoring_models import ScoringRulesConfig
    from neon_radar.domain.scoring.factor_rule import FactorRule
    from neon_radar.infrastructure.exchanges.base import ExchangeClient


class WalkForwardBacktester:
    """Walk-forward historical evaluation of the scoring engine.

    Parameters
    ----------
    exchange
        Source of historical klines. Typically :class:`BinanceClient`.
    scoring_config
        Rules and weights to evaluate. The engine is built from this.
    """

    def __init__(
        self,
        *,
        exchange: ExchangeClient,
        scoring_config: ScoringRulesConfig,
        rules: tuple[FactorRule, ...] | None = None,
    ) -> None:
        """Construct a backtester.

        Parameters
        ----------
        exchange
            Source of historical klines.
        scoring_config
            Rules and weights to evaluate. Used for ``min_confidence``
            and recommendations.
        rules
            Pre-built rule instances. If ``None``, they are taken from
            ``scoring_config.enabled_rules()`` (the loader's
            ``RuleSpec``s, not actual :class:`FactorRule` instances —
            use the loader to instantiate them, or pass pre-built rules
            in tests).
        """
        self._exchange = exchange
        self._scoring_config = scoring_config
        if rules is None:
            raise ValueError(
                "Pass pre-built rule instances via `rules=`. "
                "ScoringRulesConfig holds specs, not instantiated rules. "
                "Use `neon_radar.config.scoring_loader.load_rules` to "
                "instantiate them, or build them in tests."
            )
        self._rules = rules
        self._engine = RuleBasedEngine(
            rules=self._rules,
            min_confidence=scoring_config.min_confidence,
        )
        # Cache of (symbol, timeframe) -> full KlineSeries fetched once.
        self._series_cache: dict[tuple[str, str], KlineSeries] = {}

    @property
    def engine(self) -> RuleBasedEngine:
        return self._engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        start_date: date,
        end_date: date,
        symbols: Iterable[Symbol],
        timeframe: str,
        horizons: Iterable[int] = (1, 3, 7),
        min_history_candles: int = 100,
    ) -> BacktestResult:
        """Run walk-forward backtest over the period.

        One evaluation per day per symbol. Outcomes are computed at
        each forward horizon (1d, 3d, 7d by default).
        """
        symbols = tuple(symbols)
        horizons = tuple(sorted(set(horizons)))
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            symbols=tuple(str(s) for s in symbols),
            horizons=horizons,
            min_confidence=self._scoring_config.min_confidence,
        )

        if not symbols:
            return BacktestResult(config=config, n_evaluations=0)
        if end_date <= start_date:
            return BacktestResult(config=config, n_evaluations=0)

        # Pre-fetch klines for every symbol with a generous buffer.
        await self._prefetch(symbols, timeframe, start_date, end_date, horizons)

        evaluations = self._evaluate_all(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            horizons=horizons,
            min_history_candles=min_history_candles,
        )

        return self._aggregate(config=config, evaluations=evaluations, horizons=horizons)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _prefetch(
        self,
        symbols: tuple[Symbol, ...],
        timeframe: str,
        start_date: date,
        end_date: date,
        horizons: tuple[int, ...],
    ) -> None:
        """Fetch enough history for every (symbol, timeframe) once."""
        max_horizon = max(horizons) if horizons else 0
        # We need klines from before start_date (for warm-up) up to
        # end_date + max_horizon (for outcome computation).
        # Use a generous limit — 1500 is Binance's max per call.
        from neon_radar.config.models import TimeFrame

        tf = TimeFrame(timeframe)
        fetch_end_dt = datetime.combine(
            end_date + timedelta(days=max_horizon + 1), datetime.min.time()
        )
        fetch_end = int(fetch_end_dt.timestamp() * 1000)
        # We fetch 1500 candles which is plenty for any reasonable window.
        limit = 1500
        for symbol in symbols:
            key = (str(symbol), timeframe)
            if key in self._series_cache:
                continue
            try:
                series = await self._exchange.get_klines(
                    symbol,
                    tf,
                    end_time=fetch_end,
                    limit=limit,
                )
                self._series_cache[key] = series
            except Exception:
                # Don't abort the whole backtest on one symbol failure.
                self._series_cache[key] = KlineSeries(
                    symbol=symbol,
                    timeframe=tf,
                    candles=(),
                )

    def _evaluate_all(
        self,
        *,
        symbols: tuple[Symbol, ...],
        timeframe: str,
        start_date: date,
        end_date: date,
        horizons: tuple[int, ...],
        min_history_candles: int,
    ) -> list[EvaluationResult]:
        """Evaluate the score for every (symbol, day) pair and compute outcomes."""
        out: list[EvaluationResult] = []
        day = start_date
        while day <= end_date:
            day_start_dt = datetime.combine(day, datetime.min.time())
            day_start_ms = int(day_start_dt.timestamp() * 1000)
            for symbol in symbols:
                series = self._series_cache.get((str(symbol), timeframe))
                if series is None or len(series) == 0:
                    continue
                # Slice: keep candles with open_time < day_start_ms.
                # This means "we are scoring at the open of day".
                history = tuple(c for c in series if c.open_time < day_start_ms)
                if len(history) < min_history_candles:
                    continue
                history_series = KlineSeries(
                    symbol=series.symbol,
                    timeframe=series.timeframe,
                    candles=history,
                )
                result = self._score_at(history_series)
                if result is None:
                    continue

                # Find price at signal (close of last candle in history).
                price_at_signal = history[-1].close

                # For each horizon, find the outcome price.
                for horizon in horizons:
                    target_ms = day_start_ms + horizon * 86_400_000
                    outcome_candle = self._find_candle_at(series, target_ms)
                    if outcome_candle is None:
                        continue
                    price_after = outcome_candle.close

                    rule_values = tuple((s.name, s.value) for s in result.signals if s.value != 0)
                    out.append(
                        EvaluationResult(
                            symbol=symbol,
                            day=day,
                            score_value=result.score.value,
                            confidence=result.score.confidence,
                            price_at_signal=price_at_signal,
                            price_after_horizon=price_after,
                            horizon_days=horizon,
                            rule_values=rule_values,
                        )
                    )
            day = day + timedelta(days=1)
        return out

    def _score_at(self, series: KlineSeries) -> AnalysisResult | None:
        """Run the engine on a synthetic slice."""
        try:
            return analyze_series(
                series,
                self._rules,
                min_confidence=self._scoring_config.min_confidence,
                timestamp=int(series.candles[-1].open_time),
            )
        except Exception:
            return None

    @staticmethod
    def _find_candle_at(series: KlineSeries, target_ms: int) -> OHLCV | None:
        """First candle whose open_time >= target_ms. ``None`` if past series end."""
        for c in series:
            if c.open_time >= target_ms:
                return c
        return series.candles[-1] if series.candles else None

    # ------------------------------------------------------------------
    # Metric aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        *,
        config: BacktestConfig,
        evaluations: list[EvaluationResult],
        horizons: tuple[int, ...],
    ) -> BacktestResult:
        """Compute all metrics from raw evaluations."""
        # Per-symbol metrics.
        per_symbol: dict[str, list[EvaluationResult]] = defaultdict(list)
        for e in evaluations:
            per_symbol[str(e.symbol)].append(e)

        symbol_results: dict[str, SymbolBacktestResult] = {}
        for sym_str, evals in per_symbol.items():
            symbol_results[sym_str] = self._symbol_metrics(sym_str, evals, horizons)

        # Overall hit rate per horizon.
        overall_hit_rate: dict[int, float] = {}
        for h in horizons:
            hits = [e for e in evaluations if e.horizon_days == h and e.hit is True]
            misses = [e for e in evaluations if e.horizon_days == h and e.hit is False]
            total = len(hits) + len(misses)
            overall_hit_rate[h] = len(hits) / total if total else 0.0

        # Overall Long / Short returns (using 1d horizon for headline).
        primary = [e for e in evaluations if e.horizon_days == 1]
        long_evals = [e for e in primary if e.direction == 1]
        short_evals = [e for e in primary if e.direction == -1]
        avg_long = (
            sum(e.actual_return_pct for e in long_evals) / len(long_evals) if long_evals else 0.0
        )
        avg_short = (
            sum(e.actual_return_pct for e in short_evals) / len(short_evals) if short_evals else 0.0
        )

        # Per-rule metrics (using 1d horizon for the headline).
        rule_metrics = self._rule_metrics(evaluations, horizons)

        # Rule correlation.
        correlation = self._correlation(evaluations)

        # Confidence calibration (using 1d horizon).
        calibration = self._calibration(primary)

        recommendations = self._build_recommendations(
            overall_hit_rate=overall_hit_rate,
            correlation=correlation,
            calibration=calibration,
            rule_metrics=rule_metrics,
        )

        return BacktestResult(
            config=config,
            n_evaluations=len(evaluations),
            symbol_results=symbol_results,
            overall_hit_rate=overall_hit_rate,
            overall_avg_return_long=avg_long,
            overall_avg_return_short=avg_short,
            overall_n_long=len(long_evals),
            overall_n_short=len(short_evals),
            rule_metrics=rule_metrics,
            correlation=correlation,
            calibration=calibration,
            recommendations=recommendations,
        )

    @staticmethod
    def _symbol_metrics(
        sym: str,
        evals: list[EvaluationResult],
        horizons: tuple[int, ...],
    ) -> SymbolBacktestResult:
        hit_rate: dict[int, float] = {}
        for h in horizons:
            relevant = [e for e in evals if e.horizon_days == h]
            hits = [e for e in relevant if e.hit is True]
            misses = [e for e in relevant if e.hit is False]
            total = len(hits) + len(misses)
            hit_rate[h] = len(hits) / total if total else 0.0

        primary = [e for e in evals if e.horizon_days == 1]
        longs = [e for e in primary if e.direction == 1]
        shorts = [e for e in primary if e.direction == -1]
        neutrals = [e for e in primary if e.direction == 0]
        avg_long = sum(e.actual_return_pct for e in longs) / len(longs) if longs else 0.0
        avg_short = sum(e.actual_return_pct for e in shorts) / len(shorts) if shorts else 0.0
        avg_neutral = (
            sum(e.actual_return_pct for e in neutrals) / len(neutrals) if neutrals else 0.0
        )
        return SymbolBacktestResult(
            symbol=Symbol(sym),
            n_evaluations=len(evals),
            hit_rate=hit_rate,
            avg_return_long=avg_long,
            avg_return_short=avg_short,
            avg_return_neutral=avg_neutral,
            n_long=len(longs),
            n_short=len(shorts),
            n_neutral=len(neutrals),
        )

    @staticmethod
    def _rule_metrics(
        evaluations: list[EvaluationResult],
        horizons: tuple[int, ...],
    ) -> dict[str, RuleMetrics]:
        rule_values: dict[str, list[float]] = defaultdict(list)
        # Per-day per-rule value (use any horizon — same signal at the
        # evaluation point).
        for e in evaluations:
            for name, value in e.rule_values:
                rule_values[name].append(value)

        result: dict[str, RuleMetrics] = {}
        for rule_name, values in rule_values.items():
            votes = [v for v in values if v != 0]
            n_votes = len(votes)
            hit_rate: dict[int, float] = {}
            for h in horizons:
                relevant = [
                    e
                    for e in evaluations
                    if e.horizon_days == h
                    and any(n == rule_name and v != 0 for n, v in e.rule_values)
                ]
                hits = [e for e in relevant if e.direction != 0 and e.hit is True]
                misses = [e for e in relevant if e.direction != 0 and e.hit is False]
                total = len(hits) + len(misses)
                hit_rate[h] = len(hits) / total if total else 0.0
            avg_abs = sum(abs(v) for v in values) / len(values) if values else 0.0
            result[rule_name] = RuleMetrics(
                rule_name=rule_name,
                n_votes=n_votes,
                hit_rate_by_horizon=hit_rate,
                avg_abs_value=avg_abs,
            )
        return result

    @staticmethod
    def _correlation(evaluations: list[EvaluationResult]) -> CorrelationMatrix | None:
        """Compute pairwise Pearson correlation of per-day rule values."""
        # Group by day: rule_name -> value (taking last value per day).
        daily: dict[tuple[str, date], dict[str, float]] = defaultdict(dict)
        rule_set: set[str] = set()
        for e in evaluations:
            for name, value in e.rule_values:
                daily[(str(e.symbol), e.day)][name] = value
                rule_set.add(name)
        if not rule_set:
            return None
        names = tuple(sorted(rule_set))
        # Build aligned vectors per rule.
        days = sorted(set(daily))
        vectors = {n: [daily[d].get(n, 0.0) for d in days] for n in names}
        n = len(names)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    matrix[i][j] = 1.0
                elif j > i:
                    corr = _pearson(vectors[a], vectors[b])
                    matrix[i][j] = corr
                    matrix[j][i] = corr
        return CorrelationMatrix(
            rule_names=names,
            matrix=tuple(tuple(row) for row in matrix),
        )

    @staticmethod
    def _calibration(primary_evals: list[EvaluationResult]) -> ConfidenceCalibration | None:
        """Bucket evaluations by confidence, compute hit rate per bucket."""
        buckets_def = [
            (0.0, 0.3),
            (0.3, 0.5),
            (0.5, 0.7),
            (0.7, 1.0),
        ]
        pairs: list[tuple[float, float, int, int]] = []
        for low, high in buckets_def:
            in_bucket = [
                e for e in primary_evals if low <= e.confidence < high and e.hit is not None
            ]
            hits = sum(1 for e in in_bucket if e.hit)
            pairs.append((low, high, hits, len(in_bucket)))
        if all(total == 0 for _, _, _, total in pairs):
            return None
        return ConfidenceCalibration.from_pairs(pairs)

    @staticmethod
    def _build_recommendations(
        *,
        overall_hit_rate: dict[int, float],
        correlation: CorrelationMatrix | None,
        calibration: ConfidenceCalibration | None,
        rule_metrics: dict[str, RuleMetrics],
    ) -> tuple[str, ...]:
        """Heuristic suggestions based on the metrics."""
        recs: list[str] = []

        # 1. Check hit rate — is the engine actually predictive?
        hr_1d = overall_hit_rate.get(1, 0.0)
        if hr_1d < 0.52:
            recs.append(
                f"1d hit rate is {hr_1d:.1%} — close to random. "
                f"Current rules are not strongly predictive on this asset."
            )

        # 2. Check correlation — factor crowding?
        if correlation is not None:
            for i, a in enumerate(correlation.rule_names):
                for j, b in enumerate(correlation.rule_names):
                    if j <= i:
                        continue
                    c = correlation.matrix[i][j]
                    if abs(c) > 0.7:
                        recs.append(
                            f"{a} and {b} correlate at {c:.2f} — "
                            f"consider combining into a single factor."
                        )

        # 3. Check calibration — is confidence meaningful?
        if calibration is not None and len(calibration.buckets) >= 2:
            first_hit = calibration.buckets[0][2]
            last_hit = calibration.buckets[-1][2]
            if last_hit <= first_hit + 0.03:
                recs.append(
                    "Confidence does not correlate with hit rate — "
                    "the confidence axis is not informative for this market."
                )
            else:
                recs.append(
                    f"Confidence is calibrated: {first_hit:.0%} hit rate at low conf "
                    f"vs {last_hit:.0%} at high conf."
                )

        # 4. Per-rule hit rate — any rule below random?
        for name, m in rule_metrics.items():
            hr = m.hit_rate_by_horizon.get(1, 0.0)
            if m.n_votes >= 20 and hr < 0.48:
                recs.append(
                    f"{name} has hit rate {hr:.0%} ({m.n_votes} votes) — "
                    f"below random. Consider reducing its weight or removing it."
                )

        if not recs:
            recs.append("No specific recommendations — metrics look healthy.")
        return tuple(recs)


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient. Returns 0.0 if undefined."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    if dx2 == 0 or dy2 == 0:
        return 0.0
    corr = num / math.sqrt(dx2 * dy2)
    # Clamp to [-1, 1] to absorb floating-point noise.
    return max(-1.0, min(1.0, corr))
