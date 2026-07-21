"""Trade Evaluator for Paper Trading.

Calculates metrics for individual trades and generates a trading summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import json

from neon_radar.domain.enums import Bias

if TYPE_CHECKING:
    from neon_radar.domain.trading.paper import VirtualPosition


@dataclass(slots=True)
class TradeEvaluation:
    """Evaluated metrics for a single closed paper trade."""
    
    trade_index: int
    symbol: str
    direction: Bias
    entry_time: int
    exit_time: int
    duration_minutes: float
    
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    
    outcome: str  # 'TP', 'SL', or 'Manual'
    profit_pct: float  # Percentage profit based on entry price
    profit_r: float    # Profit in units of initial risk (R)
    
    mfe_pct: float     # Maximum Favorable Excursion (%)
    mae_pct: float     # Maximum Adverse Excursion (%)
    
    score: float
    confidence: float
    regime: str
    factors_json: str
    
    # Optional fields for snapshot
    analysis_snapshot_json: str
    
    net_pnl: float
    new_balance: float
    
    def to_csv_row(self) -> list[str]:
        """Convert to a row for paper_trades.csv."""
        from datetime import datetime, UTC
        entry_dt = datetime.fromtimestamp(self.entry_time / 1000, tz=UTC).isoformat()
        exit_dt = datetime.fromtimestamp(self.exit_time / 1000, tz=UTC).isoformat()
        
        return [
            self.symbol,
            self.direction.value,
            entry_dt,
            exit_dt,
            f"{self.duration_minutes:.2f}",
            f"{self.entry_price:.6f}",
            f"{self.exit_price:.6f}",
            self.outcome,
            f"{self.profit_pct:.4%}",
            f"{self.profit_r:.2f}",
            f"{self.mfe_pct:.4%}",
            f"{self.mae_pct:.4%}",
            f"{self.score:.4f}",
            f"{self.confidence:.4f}",
            self.regime,
            self.factors_json,
            self.analysis_snapshot_json,
            f"{self.net_pnl:.4f}",
            f"{self.new_balance:.4f}"
        ]

    @staticmethod
    def csv_header() -> list[str]:
        return [
            "Symbol", "Direction", "EntryTime", "ExitTime", "DurationMin",
            "EntryPrice", "ExitPrice", "Outcome", "ProfitPct", "ProfitR",
            "MFE_Pct", "MAE_Pct", "Score", "Confidence", "Regime",
            "FactorContributions", "AnalysisSnapshot", "NetPnL", "NewBalance"
        ]


@dataclass(slots=True)
class PaperTradingSummary:
    """Aggregate statistics for a paper trading session."""
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    average_r: float = 0.0
    average_hold_time_minutes: float = 0.0
    max_drawdown_pct: float = 0.0
    
    def to_csv_row(self) -> list[str]:
        return [
            str(self.total_trades),
            str(self.winning_trades),
            str(self.losing_trades),
            f"{self.win_rate:.2%}",
            f"{self.profit_factor:.2f}",
            f"{self.expectancy_r:.2f}",
            f"{self.average_r:.2f}",
            f"{self.average_hold_time_minutes:.2f}",
            f"{self.max_drawdown_pct:.2%}"
        ]

    @staticmethod
    def csv_header() -> list[str]:
        return [
            "TotalTrades", "WinningTrades", "LosingTrades", "WinRate",
            "ProfitFactor", "ExpectancyR", "AverageR", "AvgHoldTimeMin", "MaxDrawdownPct"
        ]


class TradeOutcomeEvaluator:
    """Evaluates individual trades and computes session summaries."""
    
    def __init__(self) -> None:
        self.evaluations: list[TradeEvaluation] = []
        self._peak_balance: float = 0.0
        
    def evaluate_trade(
        self,
        trade_index: int,
        position: VirtualPosition,
        exit_price: float,
        exit_reason: str,
        exit_time: int,
        net_pnl: float,
        new_balance: float
    ) -> TradeEvaluation:
        """Evaluates a closed VirtualPosition."""
        
        # Duration
        duration_ms = exit_time - position.entry_time
        duration_min = duration_ms / 60000.0 if duration_ms > 0 else 0.0
        
        # Gross Profit %
        if position.direction == Bias.BULLISH:
            profit_pct = (exit_price - position.entry_price) / position.entry_price
            mfe_pct = (position.highest_price - position.entry_price) / position.entry_price
            mae_pct = (position.entry_price - position.lowest_price) / position.entry_price
            risk_price_dist = position.entry_price - position.stop_loss
        else:
            profit_pct = (position.entry_price - exit_price) / position.entry_price
            mfe_pct = (position.entry_price - position.lowest_price) / position.entry_price
            mae_pct = (position.highest_price - position.entry_price) / position.entry_price
            risk_price_dist = position.stop_loss - position.entry_price
            
        # Profit in R
        profit_r = 0.0
        if risk_price_dist > 0:
            if position.direction == Bias.BULLISH:
                profit_r = (exit_price - position.entry_price) / risk_price_dist
            else:
                profit_r = (position.entry_price - exit_price) / risk_price_dist
                
        # Handle division by zero edge cases safely
        if risk_price_dist <= 0:
            profit_r = 0.0
            
        # Extract diagnostics
        score = 0.0
        conf = 0.0
        regime = ""
        factors_json = "{}"
        analysis_snap = "{}"
        
        if position.diagnostics:
            diag = position.diagnostics
            score = diag.final_score
            conf = diag.confidence
            regime = diag.regime or ""
            
        if hasattr(position, "analysis_snapshot") and position.analysis_snapshot:
            analysis_snap = json.dumps(position.analysis_snapshot)
            factors_json = json.dumps(position.analysis_snapshot.get("factors", {}))
            score = position.analysis_snapshot.get("score", score)
            conf = position.analysis_snapshot.get("confidence", conf)
            regime = position.analysis_snapshot.get("regime", regime)

        evaluation = TradeEvaluation(
            trade_index=trade_index,
            symbol=position.symbol,
            direction=position.direction,
            entry_time=position.entry_time,
            exit_time=exit_time,
            duration_minutes=duration_min,
            entry_price=position.entry_price,
            exit_price=exit_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            outcome=exit_reason,
            profit_pct=profit_pct,
            profit_r=profit_r,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            score=score,
            confidence=conf,
            regime=regime,
            factors_json=factors_json,
            analysis_snapshot_json=analysis_snap,
            net_pnl=net_pnl,
            new_balance=new_balance
        )
        
        self.evaluations.append(evaluation)
        return evaluation
        
    def generate_summary(self) -> PaperTradingSummary:
        """Generates summary statistics from all evaluated trades."""
        if not self.evaluations:
            return PaperTradingSummary()
            
        total = len(self.evaluations)
        wins = [e for e in self.evaluations if e.profit_r > 0]
        losses = [e for e in self.evaluations if e.profit_r <= 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total
        
        gross_profit_r = sum(e.profit_r for e in wins)
        gross_loss_r = abs(sum(e.profit_r for e in losses))
        
        pf = gross_profit_r / gross_loss_r if gross_loss_r > 0 else (99.0 if gross_profit_r > 0 else 0.0)
        
        avg_r = sum(e.profit_r for e in self.evaluations) / total
        
        # Expectancy = (WinRate * AvgWinR) - (LossRate * AvgLossR)
        avg_win_r = gross_profit_r / win_count if win_count > 0 else 0.0
        avg_loss_r = gross_loss_r / loss_count if loss_count > 0 else 0.0
        expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)
        
        avg_hold = sum(e.duration_minutes for e in self.evaluations) / total
        
        # Calculate Max Drawdown (Balance based)
        max_dd_pct = 0.0
        peak = self.evaluations[0].new_balance - self.evaluations[0].net_pnl # starting balance roughly
        
        for e in self.evaluations:
            if e.new_balance > peak:
                peak = e.new_balance
            dd = (peak - e.new_balance) / peak if peak > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd
                
        return PaperTradingSummary(
            total_trades=total,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            profit_factor=pf,
            expectancy_r=expectancy,
            average_r=avg_r,
            average_hold_time_minutes=avg_hold,
            max_drawdown_pct=max_dd_pct
        )
