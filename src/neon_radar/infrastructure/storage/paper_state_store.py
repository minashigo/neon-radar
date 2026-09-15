"""JSON-based atomic persistence store for Forward Paper Trading State.

Provides crash-safe serialization and recovery of account, positions,
drawdown, pending orders, and processed candle timestamps.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from neon_radar.domain.enums import Bias
from neon_radar.domain.execution_costs import ExecutionCostSummary
from neon_radar.domain.models import Symbol
from neon_radar.domain.portfolio import (
    AccountState,
    ClosedPosition,
    OpenPosition,
    PositionCloseReason,
)
from neon_radar.domain.risk import DrawdownState, RiskDecision
from neon_radar.domain.trading.backtest import TradeDiagnostics, TradeEntryReason
from neon_radar.domain.trading.paper import PaperTradingState
from neon_radar.domain.trading.setup import FinalTradeSetup
from neon_radar.utils.logging import get_logger

logger = get_logger(__name__)


def _serialize_diagnostics(d: TradeDiagnostics | None) -> dict[str, Any] | None:
    if d is None:
        return None
    return {
        "adx": d.adx,
        "atr": d.atr,
        "rsi": d.rsi,
        "ema_spread_pct": d.ema_spread_pct,
        "htf_trend": d.htf_trend,
        "confidence": d.confidence,
        "final_score": d.final_score,
        "triggered_rules": d.triggered_rules,
        "entry_reason": str(d.entry_reason.value if hasattr(d.entry_reason, "value") else d.entry_reason),
        "regime": d.regime,
        "regime_reason": d.regime_reason,
    }


def _deserialize_diagnostics(d: dict[str, Any] | None) -> TradeDiagnostics | None:
    if not d:
        return None
    return TradeDiagnostics(
        adx=d.get("adx"),
        atr=d.get("atr"),
        rsi=d.get("rsi"),
        ema_spread_pct=d.get("ema_spread_pct"),
        htf_trend=d.get("htf_trend"),
        confidence=float(d.get("confidence", 0.0)),
        final_score=float(d.get("final_score", 0.0)),
        triggered_rules=str(d.get("triggered_rules", "")),
        entry_reason=TradeEntryReason(d.get("entry_reason", "confidence_threshold")),
        regime=d.get("regime"),
        regime_reason=d.get("regime_reason"),
    )


def _serialize_risk_decision(rd: RiskDecision | None) -> dict[str, Any] | None:
    if rd is None:
        return None
    return {
        "is_allowed": rd.is_allowed,
        "max_position_size": rd.max_position_size,
        "max_risk_budget": rd.max_risk_budget,
        "rejection_reason": rd.rejection_reason,
        "risk_penalty_factor": getattr(rd, "risk_penalty_factor", 1.0),
    }


def _deserialize_risk_decision(d: dict[str, Any] | None) -> RiskDecision:
    if not d:
        return RiskDecision(is_allowed=True)
    return RiskDecision(
        is_allowed=bool(d.get("is_allowed", True)),
        rejection_reason=str(d.get("rejection_reason", "")),
        max_risk_budget=float(d.get("max_risk_budget", 0.0)),
        max_position_size=float(d.get("max_position_size", 0.0)),
        risk_penalty_factor=float(d.get("risk_penalty_factor", 1.0)),
    )


def _serialize_setup(setup: FinalTradeSetup) -> dict[str, Any]:
    return {
        "symbol": str(setup.symbol),
        "direction": setup.direction.value,
        "entry": setup.entry,
        "stop_loss": setup.stop_loss,
        "take_profit": setup.take_profit,
        "confidence": setup.confidence,
        "score": setup.score,
        "risk_decision": _serialize_risk_decision(setup.risk_decision),
        "position_size": setup.position_size,
        "quote_size": setup.quote_size,
        "risk_amount": setup.risk_amount,
        "expected_reward": setup.expected_reward,
        "diagnostics": _serialize_diagnostics(setup.diagnostics),
    }


def _deserialize_setup(d: dict[str, Any]) -> FinalTradeSetup:
    return FinalTradeSetup(
        symbol=Symbol(d["symbol"]),
        direction=Bias(d["direction"]),
        entry=float(d["entry"]),
        stop_loss=float(d["stop_loss"]),
        take_profit=float(d["take_profit"]),
        confidence=float(d["confidence"]),
        score=float(d["score"]),
        risk_decision=_deserialize_risk_decision(d.get("risk_decision")),
        position_size=float(d["position_size"]),
        quote_size=float(d["quote_size"]),
        risk_amount=float(d["risk_amount"]),
        expected_reward=float(d["expected_reward"]),
        diagnostics=_deserialize_diagnostics(d.get("diagnostics")),
    )


def _serialize_open_position(pos: OpenPosition) -> dict[str, Any]:
    return {
        "symbol": str(pos.symbol),
        "direction": pos.direction.value,
        "entry_price": pos.entry_price,
        "quantity": pos.quantity,
        "position_size": pos.position_size,
        "stop_loss": pos.stop_loss,
        "take_profit": pos.take_profit,
        "opened_at": pos.opened_at,
        "capital_at_entry": pos.capital_at_entry,
        "unrealized_pnl": pos.unrealized_pnl,
        "entry_fee": pos.entry_fee,
        "entry_slippage": pos.entry_slippage,
        "entry_execution_type": pos.entry_execution_type,
        "diagnostics": _serialize_diagnostics(pos.diagnostics),
        "timeframe": pos.timeframe,
        "signal_time": pos.signal_time,
        "htf_regime": pos.htf_regime,
    }


def _deserialize_open_position(d: dict[str, Any]) -> OpenPosition:
    return OpenPosition(
        symbol=Symbol(d["symbol"]),
        direction=Bias(d["direction"]),
        entry_price=float(d["entry_price"]),
        quantity=float(d["quantity"]),
        position_size=float(d["position_size"]),
        stop_loss=float(d["stop_loss"]),
        take_profit=float(d["take_profit"]),
        opened_at=int(d["opened_at"]),
        capital_at_entry=float(d.get("capital_at_entry", 0.0)),
        unrealized_pnl=float(d.get("unrealized_pnl", 0.0)),
        entry_fee=float(d.get("entry_fee", 0.0)),
        entry_slippage=float(d.get("entry_slippage", 0.0)),
        entry_execution_type=str(d.get("entry_execution_type", "taker")),
        diagnostics=_deserialize_diagnostics(d.get("diagnostics")),
        timeframe=str(d.get("timeframe", "1d")),
        signal_time=d.get("signal_time"),
        htf_regime=d.get("htf_regime"),
    )


def _serialize_closed_position(pos: ClosedPosition) -> dict[str, Any]:
    summary = pos.execution_summary
    return {
        "symbol": str(pos.symbol),
        "direction": pos.direction.value,
        "entry_price": pos.entry_price,
        "exit_price": pos.exit_price,
        "quantity": pos.quantity,
        "entry_time": pos.entry_time,
        "exit_time": pos.exit_time,
        "close_reason": str(pos.close_reason.value if hasattr(pos.close_reason, "value") else pos.close_reason),
        "stop_loss": pos.stop_loss,
        "take_profit": pos.take_profit,
        "initial_risk": pos.initial_risk,
        "capital_at_entry": pos.capital_at_entry,
        "execution_summary": {
            "entry_fee": summary.entry_fee,
            "exit_fee": summary.exit_fee,
            "slippage_cost": summary.slippage_cost,
            "funding_cost": summary.funding_cost,
            "gross_pnl": summary.gross_pnl,
        },
        "diagnostics": _serialize_diagnostics(pos.diagnostics),
        "timeframe": pos.timeframe,
        "signal_time": pos.signal_time,
        "htf_regime": pos.htf_regime,
    }


def _deserialize_closed_position(d: dict[str, Any]) -> ClosedPosition:
    s = d["execution_summary"]
    summary = ExecutionCostSummary(
        entry_fee=float(s.get("entry_fee", 0.0)),
        exit_fee=float(s.get("exit_fee", 0.0)),
        slippage_cost=float(s.get("slippage_cost", 0.0)),
        funding_cost=float(s.get("funding_cost", 0.0)),
        gross_pnl=float(s.get("gross_pnl", 0.0)),
    )
    return ClosedPosition(
        symbol=Symbol(d["symbol"]),
        direction=Bias(d["direction"]),
        entry_price=float(d["entry_price"]),
        exit_price=float(d["exit_price"]),
        quantity=float(d["quantity"]),
        entry_time=int(d["entry_time"]),
        exit_time=int(d["exit_time"]),
        execution_summary=summary,
        close_reason=PositionCloseReason(d.get("close_reason", "OTHER")),
        stop_loss=float(d.get("stop_loss", 0.0)),
        take_profit=float(d.get("take_profit", 0.0)),
        initial_risk=float(d.get("initial_risk", 0.0)),
        capital_at_entry=float(d.get("capital_at_entry", 0.0)),
        diagnostics=_deserialize_diagnostics(d.get("diagnostics")),
        timeframe=str(d.get("timeframe", "1d")),
        signal_time=d.get("signal_time"),
        htf_regime=d.get("htf_regime"),
    )


class JsonPaperStateStore:
    """Atomic filesystem store for PaperTradingState."""

    def __init__(self, file_path: Path | str) -> None:
        self.file_path = Path(file_path)

    def save(self, state: PaperTradingState) -> None:
        """Atomically persist PaperTradingState to disk."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.file_path.with_suffix(".tmp")

        payload: dict[str, Any] = {
            "version": 1,
            "updated_at": state.updated_at,
            "account": {
                "total_capital": state.account.total_capital,
                "free_capital": state.account.free_capital,
                "currency": state.account.currency,
            },
            "drawdown": {
                "current_equity": state.drawdown.current_equity,
                "ath_equity": state.drawdown.ath_equity,
                "max_drawdown_pct": state.drawdown.max_drawdown_pct,
                "timestamp": state.drawdown.timestamp,
            },
            "open_positions": [_serialize_open_position(p) for p in state.open_positions],
            "pending_setups": {sym: _serialize_setup(s) for sym, s in state.pending_setups.items()},
            "last_processed_candles": dict(state.last_processed_candles),
            "completed_trades": [_serialize_closed_position(p) for p in state.completed_trades],
        }

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        os.replace(tmp_path, self.file_path)
        logger.debug(f"Paper trading state saved to {self.file_path}")

    def load(self) -> PaperTradingState | None:
        """Load and deserialize PaperTradingState. Returns None if missing or corrupt."""
        if not self.file_path.exists():
            return None

        try:
            with open(self.file_path, encoding="utf-8") as f:
                data = json.load(f)

            acc_data = data["account"]
            account = AccountState(
                total_capital=float(acc_data["total_capital"]),
                free_capital=float(acc_data["free_capital"]),
                currency=str(acc_data.get("currency", "USDT")),
            )

            dd_data = data["drawdown"]
            drawdown = DrawdownState(
                current_equity=float(dd_data["current_equity"]),
                ath_equity=float(dd_data["ath_equity"]),
                max_drawdown_pct=float(dd_data["max_drawdown_pct"]),
                timestamp=int(dd_data.get("timestamp", 0)),
            )

            open_positions = tuple(_deserialize_open_position(p) for p in data.get("open_positions", []))
            pending_setups = {
                sym: _deserialize_setup(s) for sym, s in data.get("pending_setups", {}).items()
            }
            last_processed = {str(k): int(v) for k, v in data.get("last_processed_candles", {}).items()}
            completed = tuple(_deserialize_closed_position(p) for p in data.get("completed_trades", []))

            return PaperTradingState(
                account=account,
                drawdown=drawdown,
                open_positions=open_positions,
                pending_setups=pending_setups,
                last_processed_candles=last_processed,
                completed_trades=completed,
                updated_at=int(data.get("updated_at", 0)),
            )
        except Exception as exc:
            logger.error(f"Failed to load paper trading state from {self.file_path}: {exc}")
            return None
