"""Paired historical replay of the actual forward engine, never the per-symbol backtester.

Run each revision in a separate process using --source-root; fixed input archives,
configuration, portfolio and execution assumptions are shared across revisions.
"""

# ruff: noqa: E402  # Select the tested revision before importing the application.
from __future__ import annotations

import argparse
import asyncio
import bisect
import hashlib
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--timeframe", choices=("1h", "4h"), required=True)
parser.add_argument("--start", default="2025-01-01")
parser.add_argument("--end", default="2025-04-01", help="Exclusive UTC boundary")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--symbols-order", choices=("canonical", "reverse"), default="canonical")
args = parser.parse_args()
sys.path.insert(0, str(args.source_root.resolve() / "src"))

import pandas as pd

from neon_radar.application.services.paper_trading_engine import ForwardPaperTradingEngine
from neon_radar.application.services.regime_classifier import RuleBasedRegimeClassifier
from neon_radar.application.services.risk.manager import RiskManager
from neon_radar.application.services.risk.sizing import FixedRiskStrategy, PositionSizingEngine
from neon_radar.application.services.trading_pipeline import TradingPipeline
from neon_radar.config.loader import _strip_meta
from neon_radar.config.models import TimeFrame
from neon_radar.config.scoring_loader import ScoringRulesConfig, load_rules
from neon_radar.domain.enums import Bias
from neon_radar.domain.execution_costs import ExecutionCostSummary
from neon_radar.domain.funding import FundingRate
from neon_radar.domain.models import OHLCV, KlineSeries, Symbol
from neon_radar.domain.risk import PortfolioRiskPolicy
from neon_radar.domain.trading.execution import ExecutionType
from neon_radar.domain.trading.regime import RegimeFilterConfig
from neon_radar.domain.trading.setup import TradeSetupEngine
from neon_radar.infrastructure.exchanges.base import ExchangeClient, ExchangeInfo

logging.disable(logging.CRITICAL)
DATA = Path(__file__).resolve().parent / "data"
CANONICAL_ORDER = tuple(
    Symbol(s) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
)
SYMBOLS = CANONICAL_ORDER if args.symbols_order == "canonical" else tuple(reversed(CANONICAL_ORDER))
HOUR = 3600000


def timestamp(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)


class ArchiveExchange(ExchangeClient):
    name = "verified-binance-archive"

    def __init__(self):
        for record in json.loads((DATA / "manifest.json").read_text()):
            if hashlib.sha256((DATA / record["file"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Archive changed: {record['file']}")
        self.now = timestamp(args.start)
        self.series = {}
        self.close_times = {}
        self.funding = {}
        self.funding_times = {}
        self.published_funding = {}
        for symbol in SYMBOLS:
            df = pd.concat([pd.read_csv(p) for p in sorted(DATA.glob(f"{symbol}-1h-*.zip"))])
            df = df.sort_values("open_time").reset_index(drop=True)
            times = [int(t) for t in df.open_time]
            if len(times) != len(set(times)) or any(b - a != HOUR for a, b in pairwise(times)):
                raise ValueError(f"Duplicated or missing hourly candles: {symbol}")
            if times[0] > timestamp(args.start) - 200 * 24 * HOUR or times[-1] + HOUR < timestamp(
                args.end
            ):
                raise ValueError(f"Insufficient warmup or test coverage: {symbol}")
            df.index = pd.to_datetime(df.open_time, unit="ms", utc=True)
            for tf, hours in ((TimeFrame.H1, 1), (TimeFrame.H4, 4), (TimeFrame.D1, 24)):
                grouped = df.resample(f"{hours}h", origin="epoch").agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                        "open_time": "count",
                    }
                )
                grouped = grouped[grouped.open_time == hours]
                candles = tuple(
                    OHLCV(
                        open_time=int(row.Index.timestamp() * 1000),
                        open=float(row.open),
                        high=float(row.high),
                        low=float(row.low),
                        close=float(row.close),
                        volume=float(row.volume),
                        close_time=int(row.Index.timestamp() * 1000) + hours * HOUR - 1,
                    )
                    for row in grouped.itertuples()
                )
                self.series[symbol, tf] = candles
                self.close_times[symbol, tf] = [c.close_time for c in candles]
            rates = pd.concat(
                [pd.read_csv(p) for p in sorted(DATA.glob(f"{symbol}-fundingRate-*.zip"))]
            )
            rates = rates.sort_values("calc_time")
            if not all(
                int(row.funding_interval_hours) == 8 and int(row.calc_time) % (8 * HOUR) < 1000
                for row in rates.itertuples()
            ):
                raise ValueError(f"Unsupported funding interval or timestamp jitter: {symbol}")
            self.published_funding[symbol] = {
                int(row.calc_time): float(row.last_funding_rate) for row in rates.itertuples()
            }
            self.funding[symbol] = {
                int(row.calc_time) // (8 * HOUR) * (8 * HOUR): float(row.last_funding_rate)
                for row in rates.itertuples()
            }
            self.funding_times[symbol] = sorted(self.published_funding[symbol])
            ft = [
                t
                for t in sorted(self.funding[symbol])
                if timestamp(args.start) <= t < timestamp(args.end)
            ]
            expected = list(range(timestamp(args.start), timestamp(args.end), 8 * HOUR))
            if ft != expected:
                raise ValueError(f"Funding coverage is not exactly 8-hourly: {symbol}")

    async def info(self):
        return ExchangeInfo(self.name, self.name, "https://data.binance.vision", True, False)

    async def get_server_time(self):
        return self.now

    async def get_ticker(self, symbol):
        raise NotImplementedError("Replay has no contemporary ticker")

    async def get_klines(self, symbol, timeframe, *, limit=500, end_time=None):
        end = bisect.bisect_left(self.close_times[symbol, timeframe], self.now)
        candles = self.series[symbol, timeframe][max(0, end - limit) : end]
        assert all(c.close_time < self.now for c in candles)
        return KlineSeries(symbol=symbol, timeframe=timeframe, candles=candles)

    def get_funding_rate_at(self, symbol, time):
        rate = self.funding[symbol].get(time)
        return FundingRate(symbol, rate, timestamp=time) if rate is not None else None

    async def get_funding_rate(self, symbol):
        # Last published settlement, not the unknowable historical premium-index snapshot.
        idx = bisect.bisect_right(self.funding_times[symbol], self.now) - 1
        if idx < 0:
            raise ValueError("No point-in-time funding snapshot")
        time = self.funding_times[symbol][idx]
        assert time <= self.now
        return FundingRate(symbol, self.published_funding[symbol][time], timestamp=time)


async def main():
    exchange = ArchiveExchange()
    tf = TimeFrame(args.timeframe)
    config_path = args.source_root / "scoring_rules.json"
    cfg = ScoringRulesConfig.model_validate(_strip_meta(json.loads(config_path.read_text())))
    regime = RegimeFilterConfig(**cfg.regime_filter)
    classifier = RuleBasedRegimeClassifier(regime)
    pipeline = TradingPipeline(
        rules=load_rules(config_path),
        setup_engine=TradeSetupEngine(
            min_confidence=cfg.min_confidence, regime_classifier=classifier, regime_config=regime
        ),
        risk_manager=RiskManager(PortfolioRiskPolicy(base_risk_per_trade_pct=0.01)),
        sizing_engine=PositionSizingEngine(FixedRiskStrategy()),
        min_confidence=cfg.min_confidence,
        confluence_bonus=cfg.confluence_bonus,
        confluence_penalty=cfg.confluence_penalty,
        max_confidence_boost=cfg.max_confidence_boost,
        regime_classifier=classifier,
        regime_config=regime,
    )
    engine = ForwardPaperTradingEngine(
        exchange=exchange, pipeline=pipeline, symbols=SYMBOLS, timeframe=tf
    )
    # Identical historical funding costs for all revisions (live paper omits these).
    engine.execution_engine.funding_provider = exchange
    peak = 10000.0
    max_dd = 0.0
    max_positions = 0
    open_limit_events = 0
    exposure_limit_events = 0
    max_exposure_pct = 0.0
    curve = []

    def inspect_state(_event=None):
        nonlocal \
            peak, \
            max_dd, \
            max_positions, \
            open_limit_events, \
            exposure_limit_events, \
            max_exposure_pct
        state = engine.portfolio_engine.state
        equity = state.account.total_capital
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak * 100)
        max_positions = max(max_positions, len(state.positions))
        max_exposure_pct = max(max_exposure_pct, state.total_exposure / equity * 100)
        open_limit_events += len(state.positions) > 3
        exposure_limit_events += state.total_exposure > equity + 1e-8

    engine.portfolio_engine.subscribe(inspect_state)
    for t in range(
        timestamp(args.start) + tf.seconds * 1000, timestamp(args.end) + 1, tf.seconds * 1000
    ):
        exchange.now = t
        report = await engine.poll_cycle()
        if report.warnings:
            raise RuntimeError(report.warnings)
        if len(report.symbols_evaluated) != len(SYMBOLS):
            raise RuntimeError("Incomplete synchronized cycle")
        # Mark every symbol at the same close before comparing portfolio equity.
        prices = {s: (await exchange.get_klines(s, tf, limit=1)).candles[-1].close for s in SYMBOLS}
        engine.portfolio_engine.update_market_prices(prices)
        inspect_state()
        curve.append({"time": t, "equity": engine.portfolio_engine.state.account.total_capital})
    forced_closures = len(engine.portfolio_engine.state.positions)
    # Liquidate terminal positions uniformly, including taker fees, slippage and funding.
    for pos in list(engine.portfolio_engine.state.positions):
        price = prices[pos.symbol]
        notional = pos.quantity * price
        gross = (
            (price - pos.entry_price) * pos.quantity * (1 if pos.direction == Bias.BULLISH else -1)
        )
        execution = engine.execution_engine
        summary = ExecutionCostSummary(
            entry_fee=pos.entry_fee,
            exit_fee=execution.fee_model.calculate_exit_fee(notional, ExecutionType.TAKER),
            slippage_cost=pos.entry_slippage
            + execution.slippage_model.calculate_slippage(
                notional, pos.symbol, ExecutionType.TAKER, pos.direction
            ),
            funding_cost=execution.funding_model.calculate_funding_cost(
                notional,
                pos.symbol,
                pos.direction,
                pos.opened_at,
                timestamp(args.end) - 1,
                exchange,
            ),
            gross_pnl=gross,
        )
        engine.portfolio_engine.close_position(
            str(pos.symbol), price, timestamp(args.end) - 1, summary, reason="MANUAL"
        )
    curve.append(
        {"time": timestamp(args.end), "equity": engine.portfolio_engine.state.account.total_capital}
    )
    trades = engine.portfolio_engine.history

    def trade_record(p):
        return {
            "symbol": str(p.symbol),
            "direction": p.direction.value,
            "entry": p.entry_time,
            "exit": p.exit_time,
            "net": p.net_pnl,
            "gross": p.gross_pnl,
            "risk": p.initial_risk,
            "r": p.profit_r,
            "fees": p.execution_summary.entry_fee + p.execution_summary.exit_fee,
            "slippage": p.execution_summary.slippage_cost,
            "funding": p.execution_summary.funding_cost,
            "reason": str(p.close_reason),
            "rsi": p.diagnostics.rsi if p.diagnostics else None,
        }

    records = [trade_record(p) for p in trades]
    net = sum(p.net_pnl for p in trades)
    profit = sum(max(p.net_pnl, 0) for p in trades)
    loss = -sum(min(p.net_pnl, 0) for p in trades)
    state = engine.portfolio_engine.state
    equity = state.account.total_capital
    if abs(equity - (10000 + net + sum(p.unrealized_pnl for p in state.positions))) > 1e-6:
        raise ValueError("Equity reconciliation failed")
    result = {
        "commit": subprocess.check_output(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "requested_symbol_order": list(SYMBOLS),
        "working_source_sha256": {
            path: hashlib.sha256((args.source_root / path).read_bytes()).hexdigest()
            for path in (
                "src/neon_radar/application/services/execution.py",
                "src/neon_radar/application/services/paper_trading_engine.py",
                "src/neon_radar/domain/execution_costs.py",
                "src/neon_radar/infrastructure/providers/binance_funding.py",
            )
        },
        "timeframe": args.timeframe,
        "start": args.start,
        "end_exclusive": args.end,
        "scoring_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "archive_manifest_sha256": hashlib.sha256(
            (DATA / "manifest.json").read_bytes()
        ).hexdigest(),
        "settings": {
            "risk": 0.01,
            "initial_capital": 10000,
            "symbols": list(CANONICAL_ORDER),
            "funding_costs": True,
            "funding_signal": "last published settlement",
            "intrabar": "engine OHLC; no live ticks",
        },
        "forced_terminal_closures": forced_closures,
        "closed_trades": len(trades),
        "wins": sum(p.net_pnl > 0 for p in trades),
        "net_closed": net,
        "equity": equity,
        "total_return_pct": (equity / 10000 - 1) * 100,
        "net_profit_factor": profit / loss if loss else None,
        "max_drawdown_pct": max_dd,
        "max_open_positions": max_positions,
        "open_limit_events": open_limit_events,
        "exposure_over_equity_events": exposure_limit_events,
        "max_exposure_pct": max_exposure_pct,
        "fees": sum(r["fees"] for r in records),
        "slippage": sum(r["slippage"] for r in records),
        "funding": sum(r["funding"] for r in records),
        "open_positions": len(state.positions),
        "pending_orders": len(engine.execution_engine.pending_setups),
        "trades": records,
        "curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("curve", "trades")}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
