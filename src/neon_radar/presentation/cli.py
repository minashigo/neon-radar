"""Command-line interface for Neon Radar.

This is the **first usable form** of the product. No Qt, no chart —
just a table of scoring results for every configured symbol.

Subcommands
-----------
``scan``
    Fetch klines for every enabled symbol, compute the configured
    indicators, run the scoring engine, print the results table.
    ``--explain`` adds a per-factor breakdown below each row.

``list-rules``
    Print all registered scoring rules and their default parameters.

``backtest``
    Walk-forward historical simulation. Computes hit rates, per-rule
    metrics, rule correlations and confidence calibration over a
    date range. Outputs recommendations based on the metrics.

Both subcommands share common options (``--config``, ``--scoring``).

Usage examples::

    neon-radar scan --explain
    neon-radar scan --timeframe 4h --limit 500
    neon-radar backtest --start 2024-01-01 --end 2024-12-31
    neon-radar list-rules
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from neon_radar.application.services.analysis import analyze_series
from neon_radar.config.loader import ConfigLoader
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.enums import Bias
from neon_radar.domain.exceptions import NeonRadarError
from neon_radar.domain.indicators import IndicatorRegistry
from neon_radar.domain.models import Symbol
from neon_radar.domain.scoring import (
    AnalysisResult,
    RuleRegistry,
)
from neon_radar.infrastructure.exchanges.binance import BinanceClient
from neon_radar.utils.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from neon_radar.domain.funding import FundingRate
    from neon_radar.domain.market_state import MarketState
    from neon_radar.domain.scoring.backtest import BacktestResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neon-radar",
        description="Neon Radar — Binance Futures scoring CLI",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the application config (default: config.json)",
    )
    parser.add_argument(
        "--scoring",
        type=Path,
        default=Path("scoring_rules.json"),
        help="Path to the scoring rules config (default: scoring_rules.json)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: WARNING)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable coloured output",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- scan ----
    scan = sub.add_parser("scan", help="Score every enabled symbol and print a table")
    scan.add_argument(
        "--timeframe",
        default=None,
        help="Timeframe to analyse (default: first configured)",
    )
    scan.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of candles to fetch per symbol (default: 300)",
    )
    scan.add_argument(
        "--explain",
        action="store_true",
        help="Show per-factor score breakdown below each row",
    )

    # ---- list-rules ----
    sub.add_parser(
        "list-rules",
        help="List all registered scoring rules and their defaults",
    )

    # ---- signals-backtest ----
    signals_backtest = sub.add_parser(
        "signals-backtest",
        help="Walk-forward historical simulation of the scoring engine (signals only)",
    )
    signals_backtest.add_argument(
        "--start",
        type=_parse_date,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    signals_backtest.add_argument(
        "--end",
        type=_parse_date,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    signals_backtest.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (default: all enabled in config.json)",
    )
    signals_backtest.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe (default: 1d)",
    )
    signals_backtest.add_argument(
        "--horizons",
        default="1,3,7",
        help="Comma-separated forward horizons in days (default: 1,3,7)",
    )
    signals_backtest.add_argument(
        "--min-history",
        type=int,
        default=100,
        help="Minimum candles of history before scoring (default: 100)",
    )
    signals_backtest.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # ---- backtest ----
    backtest = sub.add_parser(
        "backtest",
        help="Walk-forward historical simulation of the complete trading system",
    )
    backtest.add_argument(
        "--start",
        type=_parse_date,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    backtest.add_argument(
        "--end",
        type=_parse_date,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    backtest.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (default: all enabled in config.json)",
    )
    backtest.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe (default: 1d)",
    )
    backtest.add_argument(
        "--min-history",
        type=int,
        default=100,
        help="Minimum candles of history before trading (default: 100)",
    )
    backtest.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    backtest.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Path to export the trade history to CSV",
    )
    backtest.add_argument(
        "--feature-analysis",
        action="store_true",
        help="Run Ablation Analysis to determine feature importance",
    )
    backtest.add_argument(
        "--bootstrap",
        action="store_true",
        help="Run Block Bootstrap statistical validation on the backtest results",
    )
    backtest.add_argument(
        "--bootstrap-block-size",
        type=int,
        default=20,
        help="Block size for bootstrap resampling (default: 20)",
    )
    backtest.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000)",
    )
    backtest.add_argument(
        "--export-bootstrap-json",
        type=Path,
        default=None,
        help="Path to export the full bootstrap report to JSON",
    )
    backtest.add_argument(
        "--export-bootstrap-csv",
        type=Path,
        default=None,
        help="Path to export the bootstrap summary metrics to CSV",
    )

    return parser


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}, expected YYYY-MM-DD") from exc


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_list_rules(args: argparse.Namespace) -> int:
    """Print all registered scoring rules."""
    print("Registered scoring rules:\n")
    for name in RuleRegistry.names():
        cls = RuleRegistry.get(name)
        desc = cls.description()
        print(f"  {desc.name}")
        print(f"    display:  {desc.display_name}")
        print(f"    summary:  {desc.summary}")
        print(f"    defaults: {desc.default_params}")
        print()
    print("Registered indicators:")
    for name in IndicatorRegistry.names():
        cls = IndicatorRegistry.get(name)
        print(f"  {name:20s}  kind={cls.KIND.value}")
    return 0


async def _score_one_symbol(
    client: BinanceClient,
    symbol: Symbol,
    timeframe,
    *,
    limit: int,
    rules: tuple,
    min_confidence: float,
) -> tuple[Symbol, MarketState, AnalysisResult]:
    """Fetch + compute + score one symbol. Returns (symbol, state, result)."""
    htf = timeframe.higher_timeframe
    coros = [client.get_klines(symbol, timeframe, limit=limit)]
    if htf:
        coros.append(client.get_klines(symbol, htf, limit=limit))

    results = await asyncio.gather(*coros)
    series = results[0]
    higher_tf_series = results[1] if htf else None

    funding_rate: FundingRate | None = None
    try:
        funding_rate = await client.get_funding_rate(symbol)
    except NeonRadarError:
        logger.debug("Funding rate unavailable for %s", symbol)

    result = analyze_series(
        series,
        rules,
        min_confidence=min_confidence,
        timestamp=int(_now_ms()),
        funding_rate=funding_rate,
        higher_tf_series=higher_tf_series,
    )
    assert result.market_state is not None
    return symbol, result.market_state, result


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


async def _run_scan(args: argparse.Namespace) -> int:
    """Fetch + score every enabled symbol; print a sorted table."""
    config = ConfigLoader(args.config).load()
    scoring_cfg = ScoringRulesConfig.model_validate(_strip_meta(_read_json(args.scoring)))
    scoring_cfg.enabled_rules()
    # Rebuild rules with config weights. ``load_rules`` already does this,
    # but we need the scoring_cfg for min_confidence.
    from neon_radar.config.scoring_loader import load_rules as _load

    rules = tuple(_load(args.scoring))
    if not rules:
        print(
            "No enabled scoring rules found. Check your scoring_rules.json.",
            file=sys.stderr,
        )
        return 2

    timeframe = args.timeframe or config.timeframes[0].value
    limit = args.limit

    logger.info(
        "Scanning %d symbols on %s, %d candles each",
        len(config.enabled_symbols()),
        timeframe,
        limit,
    )

    async with BinanceClient(config.api) as client:
        rows: list[tuple[Symbol, AnalysisResult]] = []
        for sym_cfg in config.enabled_symbols():
            symbol = Symbol(sym_cfg.symbol)
            logger.info("Fetching %s...", symbol)
            try:
                _, _, result = await _score_one_symbol(
                    client,
                    symbol,
                    _tf_from_str(timeframe),
                    limit=limit,
                    rules=rules,
                    min_confidence=scoring_cfg.min_confidence,
                )
                rows.append((symbol, result))
            except Exception as exc:
                logger.warning("Failed to score %s: %s", symbol, exc)

    rows.sort(key=lambda r: r[1].score.value, reverse=True)

    print_table(rows, use_color=_should_color(args), explain=args.explain)
    return 0


async def _run_backtest(args: argparse.Namespace) -> int:
    """Walk-forward backtest via ``WalkForwardBacktester``."""
    from neon_radar.application.services.backtester import WalkForwardBacktester

    config = ConfigLoader(args.config).load()
    scoring_cfg = ScoringRulesConfig.model_validate(_strip_meta(_read_json(args.scoring)))

    if args.symbols:
        symbols = tuple(Symbol(s.strip()) for s in args.symbols.split(",") if s.strip())
    else:
        symbols = tuple(Symbol(s.symbol) for s in config.enabled_symbols())

    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())

    logger.info(
        "Backtest: %s to %s, %d symbols, horizons=%s",
        args.start,
        args.end,
        len(symbols),
        horizons,
    )

    async with BinanceClient(config.api) as client:
        from neon_radar.config.scoring_loader import load_rules as _load

        rules = tuple(_load(args.scoring))
        backtester = WalkForwardBacktester(
            exchange=client,
            scoring_config=scoring_cfg,
            rules=rules,
        )
        result = await backtester.run(
            start_date=args.start,
            end_date=args.end,
            symbols=symbols,
            timeframe=args.timeframe,
            horizons=horizons,
            min_history_candles=args.min_history,
        )

    if args.output == "json":
        print_result_json(result)
    else:
        print_backtest_report(result, use_color=_should_color(args))
    return 0


async def _run_trade_backtest(args: argparse.Namespace) -> int:
    """Walk-forward trade backtest via ``TradeBacktester``."""
    from neon_radar.application.services.trade_backtester import TradeBacktester

    config = ConfigLoader(args.config).load()
    scoring_cfg = ScoringRulesConfig.model_validate(_strip_meta(_read_json(args.scoring)))

    if args.symbols:
        symbols = tuple(Symbol(s.strip()) for s in args.symbols.split(",") if s.strip())
    else:
        symbols = tuple(Symbol(s.symbol) for s in config.enabled_symbols())

    logger.info(
        "Trade Backtest: %s to %s, %d symbols",
        args.start,
        args.end,
        len(symbols),
    )

    async with BinanceClient(config.api) as client:
        from neon_radar.config.scoring_loader import load_rules as _load

        rules = tuple(_load(args.scoring))
        backtester = TradeBacktester(
            exchange=client,
            scoring_config=scoring_cfg,
            rules=rules,
        )

        from neon_radar.application.services.trade_analyzer import TradeAnalyzer
        analyzer = TradeAnalyzer()

        if args.feature_analysis:
            from neon_radar.application.services.feature_analyzer import FeatureImportanceAnalyzer
            feature_analyzer = FeatureImportanceAnalyzer(analyzer)
            feature_report = await feature_analyzer.analyze(
                baseline_tester=backtester,
                start_date=args.start,
                end_date=args.end,
                symbols=symbols,
                timeframe=args.timeframe,
                min_history_candles=args.min_history,
            )
            report = feature_report.baseline
            trades = report.trades

            if args.output == "json":
                print_result_json(feature_report)
            else:
                print_feature_importance_report(feature_report, use_color=_should_color(args))
        else:
            trades = await backtester.run(
                start_date=args.start,
                end_date=args.end,
                symbols=symbols,
                timeframe=args.timeframe,
                min_history_candles=args.min_history,
            )
            report = analyzer.analyze(trades)

            if args.output == "json":
                print_result_json(report)
            else:
                print_trade_backtest_report(report, use_color=_should_color(args))

            if args.bootstrap:
                from neon_radar.application.services.bootstrap_analyzer import BootstrapAnalyzer
                boot_analyzer = BootstrapAnalyzer(analyzer.analyze)
                boot_report = boot_analyzer.run(
                    trades,
                    block_size=args.bootstrap_block_size,
                    iterations=args.bootstrap_iterations,
                )
                if boot_report:
                    if args.output == "json":
                        print_result_json(boot_report)
                    else:
                        print_bootstrap_report(boot_report, use_color=_should_color(args))

                    if args.export_bootstrap_json:
                        _export_bootstrap_json(boot_report, args.export_bootstrap_json)
                        logger.info("Exported bootstrap JSON to %s", args.export_bootstrap_json)

                    if args.export_bootstrap_csv:
                        _export_bootstrap_csv(boot_report, args.export_bootstrap_csv)
                        logger.info("Exported bootstrap CSV to %s", args.export_bootstrap_csv)

    if args.export_csv:
        from neon_radar.infrastructure.exporters.trade_exporter import export_trades_to_csv
        export_trades_to_csv(trades, args.export_csv)
        logger.info("Exported trades to %s", args.export_csv)

    return 0


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _export_bootstrap_json(report, path: Path) -> None:
    import dataclasses
    import json

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, dict):
            return {str(k): _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_dict(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_dict(report), f, indent=2)


def _export_bootstrap_csv(report, path: Path) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Median", "StdDev", "Min", "Max", "95% CI Lower", "95% CI Upper"])
        for name, dist in report.metrics.items():
            writer.writerow([
                name,
                f"{dist.mean:.4f}",
                f"{dist.median:.4f}",
                f"{dist.std_dev:.4f}",
                f"{dist.min_val:.4f}",
                f"{dist.max_val:.4f}",
                f"{dist.ci_lower_95:.4f}",
                f"{dist.ci_upper_95:.4f}",
            ])


def _strip_meta(data):
    """Strip ``$schema`` and ``_*`` keys (delegates to config loader helper)."""
    from neon_radar.config.loader import _strip_meta as _impl

    return _impl(data)


def _tf_from_str(value: str):
    from neon_radar.config.models import TimeFrame

    try:
        return TimeFrame(value)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid timeframe: {value!r}. Valid options: {[t.value for t in TimeFrame]}"
        ) from exc


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


_ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "green": "\x1b[32m",
    "red": "\x1b[31m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
    "magenta": "\x1b[35m",
}


def _should_color(args: argparse.Namespace) -> bool:
    if args.no_color:
        return False
    return sys.stdout.isatty()


def _c(text: str, code: str, enabled: bool) -> str:
    return f"{_ANSI[code]}{text}{_ANSI['reset']}" if enabled else text


def _signed(value: float) -> str:
    return f"{value:+.2f}"


def _arrow(value: float) -> str:
    if value > 0:
        return "+"
    if value < 0:
        return "-"
    return "0"


def _format_score_row(
    symbol: Symbol,
    result: AnalysisResult,
    use_color: bool,
) -> str:
    score = result.score
    bias = score.bias
    if bias is Bias.BULLISH:
        bias_text = _c("BULLISH ", "green", use_color)
    elif bias is Bias.BEARISH:
        bias_text = _c("BEARISH ", "red", use_color)
    else:
        bias_text = _c("NEUTRAL ", "yellow", use_color)

    factors = (
        ", ".join(f"{s.name}:{_arrow(s.value)}" for s in result.signals if s.value != 0)
        or "(no directional signals)"
    )

    return (
        f"{symbol:<10}  "
        f"score {_c(_signed(score.value), 'bold', use_color)}  "
        f"conf {score.confidence:.2f}  "
        f"{bias_text} "
        f"{_c(factors, 'dim', use_color)}"
    )


def print_table(
    rows: list[tuple[Symbol, AnalysisResult]],
    *,
    use_color: bool,
    explain: bool = False,
) -> None:
    """Print the scoring results as a sorted table.

    When ``explain`` is true, each row is followed by a per-factor
    breakdown showing signed contribution, value, weight, confidence.
    """
    if not rows:
        print(_c("No symbols scored.", "yellow", use_color))
        return

    header = f"{'SYMBOL':<10}  {'SCORE':>7}  {'CONF':>5}  {'BIAS':<8}  FACTORS"
    print(_c(header, "bold", use_color))
    print("-" * len(header))
    for symbol, result in rows:
        print(_format_score_row(symbol, result, use_color))
        if explain:
            _print_breakdown(result, use_color)


def _print_breakdown(result: AnalysisResult, use_color: bool) -> None:
    """Indented per-factor breakdown for ``--explain``."""
    breakdown = result.breakdown()
    if not breakdown:
        print(_c("    (no signals)", "dim", use_color))
        return
    for b in breakdown:
        _arrow(b.contribution)
        contrib = f"{b.contribution:+.2f}"
        value = f"{b.value:+.2f}"
        weight = f"{b.weight:.2f}"
        conf = f"{b.confidence:.2f}"
        line = (
            f"    ├─ {b.factor:<22} contrib {contrib:<7} "
            f"value {value:<7} weight {weight}  conf {conf}"
        )
        if b.is_bullish:
            print(_c(line, "green", use_color))
        elif b.is_bearish:
            print(_c(line, "red", use_color))
        else:
            print(_c(line, "dim", use_color))


# ---------------------------------------------------------------------------
# Backtest report rendering
# ---------------------------------------------------------------------------


def print_backtest_report(result: BacktestResult, *, use_color: bool) -> None:
    """Human-readable backtest output."""
    if result.n_evaluations == 0:
        print(_c("No evaluations produced. Check date range and history.", "yellow", use_color))
        return

    cfg = result.config
    print(
        f"Walk-forward backtest: {cfg.start_date} → {cfg.end_date} "
        f"({result.n_evaluations} evaluations)"
    )
    print(f"Timeframe: {cfg.timeframe}  |  Horizons: {cfg.horizons}")
    print(f"Symbols: {', '.join(cfg.symbols)}\n")

    # Overall hit rate.
    print(_c("=== Overall hit rate ===", "bold", use_color))
    print(f"{'horizon':<10}  {'hit%':>7}")
    for h in cfg.horizons:
        hr = result.hit_rate(h)
        print(f"{h}d        {hr:>7.1%}")
    print()

    # Long/Short returns (using 1d).
    print(_c("=== Avg return when Long vs Short (1d horizon) ===", "bold", use_color))
    print(f"  Long:   {result.overall_avg_return_long:+.2%}  (n={result.overall_n_long})")
    print(f"  Short:  {result.overall_avg_return_short:+.2%}  (n={result.overall_n_short})")
    print()

    # Per-rule.
    print(_c("=== Per-rule hit rate (1d horizon, when rule voted) ===", "bold", use_color))
    print(f"{'rule':<25}  {'n_votes':>8}  {'hit_rate':>9}  {'avg|val|':>8}")
    for name, m in sorted(result.rule_metrics.items()):
        hr = m.hit_rate_by_horizon.get(1, 0.0)
        print(f"{name:<25}  {m.n_votes:>8}  {hr:>9.1%}  {m.avg_abs_value:>8.2f}")
    print()

    # Correlation.
    if result.correlation is not None:
        print(_c("=== Rule correlation (Pearson of daily signal values) ===", "bold", use_color))
        names = result.correlation.rule_names
        # Header.
        header = "                " + "  ".join(f"{n[:10]:<10}" for n in names)
        print(header)
        for i, row_name in enumerate(names):
            cells = "  ".join(
                f"{result.correlation.matrix[i][j]:>+10.2f}" for j in range(len(names))
            )
            print(f"  {row_name[:14]:<14}  {cells}")
        print()

    # Calibration.
    if result.calibration is not None:
        print(_c("=== Confidence calibration (1d horizon) ===", "bold", use_color))
        print(f"{'bucket':<10}  {'hit%':>7}  {'bar'}")
        for low, high, hr in result.calibration.buckets:
            bar_len = int(hr * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"{low:.1f}-{high:.1f}   {hr:>7.1%}  {bar}")
        print()

    # Recommendations.
    print(_c("=== Recommendations ===", "bold", use_color))
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    print()


def print_trade_backtest_report(result, *, use_color: bool) -> None:
    """Human-readable trade backtest output."""
    from neon_radar.domain.trading.backtest import BacktestReport

    if not isinstance(result, BacktestReport):
        return

    if result.total_trades == 0:
        print(_c("No trades executed. Check date range and history.", "yellow", use_color))
        return

    print(_c("=== Trade Backtest Summary ===", "bold", use_color))
    print(f"Total Trades:  {result.total_trades}")
    print(f"Win Rate:        {result.win_rate:>7.1%}")
    print(f"Wins:          {result.wins}")
    print(f"Losses:        {result.losses}")
    print(f"Avg Win:       {result.net_avg_win_pct:>+7.2%}")
    print(f"Avg Loss:      {result.net_avg_loss_pct:>+7.2%}")
    print(f"Profit Factor:   {result.net_profit_factor:>7.2f}")
    print(f"Expectancy:    {result.net_expectancy:>+7.2%}")
    print(f"Max Cons. Wins:  {result.max_consecutive_wins}")
    print(f"Max Cons. Loss:  {result.max_consecutive_losses}")

    # Format holding time (e.g. '4.5 hours', '1.2 days')
    ms = result.avg_holding_time_ms
    if ms < 1000 * 60 * 60:
        ht = f"{ms / (1000 * 60):.1f} mins"
    elif ms < 1000 * 60 * 60 * 24:
        ht = f"{ms / (1000 * 60 * 60):.1f} hours"
    else:
        ht = f"{ms / (1000 * 60 * 60 * 24):.1f} days"
    print(f"Avg Holding Time:{ht:>7s}")
    print()

    if result.validation and result.validation.is_valid:
        print(_c("=== Statistical Validation ===", "bold", use_color))

        pval_color = "green" if result.validation.p_value < 0.05 else "yellow"
        pval_msg = "<0.05 -> Edge Detected" if result.validation.p_value < 0.05 else ">=0.05 -> Insufficient Edge"

        print(f"P-Value (T-Test):       {_c(f'{result.validation.p_value:.4f} ({pval_msg})', pval_color, use_color)}")
        print(f"T-Statistic:            {result.validation.t_statistic:.3f}")

        ci_lower = result.validation.mc_expectancy_95_ci_lower
        ci_upper = result.validation.mc_expectancy_95_ci_upper
        ci_color = "green" if ci_lower > 0 else "yellow"
        print(f"MC Expectancy 95% CI:   {_c(f'[{ci_lower:>+5.2%}, {ci_upper:>+5.2%}]', ci_color, use_color)}")

        loss_prob = result.validation.mc_probability_of_loss
        loss_color = "green" if loss_prob < 0.1 else ("yellow" if loss_prob < 0.3 else "red")
        print(f"MC Probability of Loss: {_c(f'{loss_prob:.1%}', loss_color, use_color)}")
        print()

    print(_c("=== Executed Trades ===", "bold", use_color))
    print(
        f"{'SYMBOL':<10} {'DIR':<8} {'STATUS':<10} {'REASON':<12} {'ENTRY':<10} {'EXIT':<10} {'PNL%':>8}"
    )
    for t in result.trades:
        color = "green" if t.net_pnl_pct > 0 else ("red" if t.net_pnl_pct < 0 else "dim")
        status = t.status.value.upper()
        dir_name = t.direction.name
        reason = t.exit_reason.value.upper()

        # Format prices to match significant digits
        ep = f"{t.entry_price:.4f}"
        xp = f"{t.exit_price:.4f}" if t.exit_price is not None else "OPEN"
        pnl = f"{t.net_pnl_pct:>+8.2%}"

        row = f"{t.symbol!s:<10} {dir_name:<8} {status:<10} {reason:<12} {ep:<10} {xp:<10} {pnl}"
        print(_c(row, color, use_color))
    print()


def print_feature_importance_report(report, *, use_color: bool) -> None:
    """Print the feature importance table from Ablation Analysis."""
    from neon_radar.domain.trading.feature_importance import FeatureImportanceReport

    if not isinstance(report, FeatureImportanceReport):
        return

    print(_c("=== Baseline Performance ===", "bold", use_color))
    print(f"Total Trades:    {report.baseline.total_trades}")
    print(f"Win Rate:        {report.baseline.win_rate:>7.1%}")
    print(f"Profit Factor:   {report.baseline.profit_factor:>7.2f}")
    print(f"Expectancy:      {report.baseline.expectancy:>+7.2%}")
    print()

    print(_c("=== Feature Importance ===", "bold", use_color))
    if not report.features:
        print("No rules evaluated.")
        print()
        return

    max_len = max(len(f.rule_name) for f in report.features)
    for f in report.features:
        rating = f.rating_symbols
        # Add color based on rating
        if "-" in rating:
            color = "red"
        elif "+" in rating:
            color = "green"
        else:
            color = "dim"

        row = f"{f.rule_name:<{max_len}}   {rating}"
        print(_c(row, color, use_color))

    print()
    print(_c("Detailed Deltas:", "dim", use_color))
    header = f"{'RULE':<{max_len}} | {'dPF':>6} | {'dEXP':>7} | {'dSHARPE':>7} | {'dWR':>6} | {'dPROB_L':>7} | SCORE"
    print(_c(header, "dim", use_color))

    for f in report.features:
        row = (
            f"{f.rule_name:<{max_len}} | "
            f"{f.delta_profit_factor:>+6.2f} | "
            f"{f.delta_expectancy:>+7.2%} | "
            f"{f.delta_sharpe_ratio:>+7.2f} | "
            f"{f.delta_win_rate:>+6.1%} | "
            f"{f.delta_probability_of_loss:>+7.1%} | "
            f"{f.feature_score:+.2f}"
        )
        print(_c(row, "dim", use_color))
    print()


def print_bootstrap_report(report, *, use_color: bool) -> None:
    """Print the bootstrap validation summary."""
    from neon_radar.domain.trading.bootstrap import BootstrapReport

    if not isinstance(report, BootstrapReport):
        return

    print(_c("=== Bootstrap Summary ===", "bold", use_color))
    print(f"Iterations: {report.iterations}")
    print(f"Block Size: {report.block_size}")
    print()

    for name, dist in report.metrics.items():
        print(_c(f"{name}:", "bold", use_color))
        print(f"  mean:         {dist.mean:.4f}")
        print(f"  median:       {dist.median:.4f}")
        print(f"  95% CI:       [{dist.ci_lower_95:.4f}, {dist.ci_upper_95:.4f}]")
        print()


def print_result_json(result) -> None:
    """JSON dump of the backtest — for further analysis."""
    import dataclasses
    import json

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, dict):
            return {str(k): _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_dict(v) for v in obj]
        if isinstance(obj, date):
            return obj.isoformat()
        if hasattr(obj, "value"):  # StrEnum etc.
            return obj.value
        return obj

    print(json.dumps(_to_dict(result), indent=2, default=str))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(level=args.log_level, console=True)

    if args.command == "list-rules":
        return cmd_list_rules(args)
    if args.command == "scan":
        return asyncio.run(_run_scan(args))
    if args.command == "signals-backtest":
        return asyncio.run(_run_backtest(args))
    if args.command == "backtest":
        return asyncio.run(_run_trade_backtest(args))

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
