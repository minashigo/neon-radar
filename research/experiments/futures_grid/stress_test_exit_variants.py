"""Stress Testing Suite for Futures Grid Exit Management Variants.

Tests performance and capital preservation during extreme market stress:
1. Flash Crash (-40% drop): Tests downward inventory accumulation and risk cutoff.
2. Parabolic Bull (+60% pump): Tests short inventory squeeze and upside trailing.
3. V-Reversal (-25% dip then +30% rally): Tests whether freezing inventory benefits from recovery or suffers.
4. Prolonged Trend (+35% over 40 bars): Tests margin drain and long-term hold risks.
5. Rapid Regime Flickering (Chop/Trend alternating every 1-2 bars): Tests resistance to churn traps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from research.experiments.futures_grid.exit_management_engine import (
    ExitEngineConfig,
    run_continuous_grid_simulation,
)
from research.experiments.futures_grid.portfolio import (
    ExitVariant,
    PortfolioAccount,
)


@dataclass(frozen=True)
class StressScenarioResult:
    scenario_name: str
    variant: str
    final_equity: float
    net_pnl: float
    max_drawdown_pct: float
    is_bankrupt: bool
    liquidations_count: int
    total_churn_cost: float
    exits_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "variant": self.variant,
            "final_equity": round(self.final_equity, 2),
            "net_pnl": round(self.net_pnl, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "is_bankrupt": self.is_bankrupt,
            "liquidations_count": self.liquidations_count,
            "total_churn_cost": round(self.total_churn_cost, 2),
            "exits_count": self.exits_count,
        }


def _build_scenario_dfs(candles: list[dict[str, Any]], regimes: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(candles), pd.DataFrame(regimes)


def generate_stress_flash_crash(base_p: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4 calm CHOP bars, then 8 crashing TREND bars (-40%)."""
    candles, regimes = [], []
    t = 1000
    p = base_p

    # 4 calm CHOP bars
    for _ in range(4):
        candles.append({"open_time_ms": t, "open": p, "high": p * 1.01, "low": p * 0.99, "close": p, "volume": 100.0})
        regimes.append({"open_time_ms": t, "regime": "CHOP", "grid_allowed": True})
        t += 4 * 3600 * 1000

    # 8 crashing bars
    target = base_p * 0.60
    step = (p - target) / 8
    for _ in range(8):
        c_open = p
        c_low = p - step * 1.05
        c_close = p - step
        c_high = p * 1.002
        candles.append({"open_time_ms": t, "open": c_open, "high": c_high, "low": c_low, "close": c_close, "volume": 200.0})
        regimes.append({"open_time_ms": t, "regime": "VOLATILE_CRASH", "grid_allowed": False})
        p = c_close
        t += 4 * 3600 * 1000

    return _build_scenario_dfs(candles, regimes)


def generate_stress_parabolic_bull(base_p: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4 calm CHOP bars, then 8 surging TREND bars (+60%)."""
    candles, regimes = [], []
    t = 1000
    p = base_p

    for _ in range(4):
        candles.append({"open_time_ms": t, "open": p, "high": p * 1.01, "low": p * 0.99, "close": p, "volume": 100.0})
        regimes.append({"open_time_ms": t, "regime": "CHOP", "grid_allowed": True})
        t += 4 * 3600 * 1000

    target = base_p * 1.60
    step = (target - p) / 8
    for _ in range(8):
        c_open = p
        c_high = p + step * 1.05
        c_close = p + step
        c_low = p * 0.998
        candles.append({"open_time_ms": t, "open": c_open, "high": c_high, "low": c_low, "close": c_close, "volume": 250.0})
        regimes.append({"open_time_ms": t, "regime": "BULL_TREND", "grid_allowed": False})
        p = c_close
        t += 4 * 3600 * 1000

    return _build_scenario_dfs(candles, regimes)


def generate_stress_v_reversal(base_p: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4 calm bars, 6 crashing bars (-25%), then 6 surging bars back to base price."""
    candles, regimes = [], []
    t = 1000
    p = base_p

    for _ in range(4):
        candles.append({"open_time_ms": t, "open": p, "high": p * 1.01, "low": p * 0.99, "close": p, "volume": 100.0})
        regimes.append({"open_time_ms": t, "regime": "CHOP", "grid_allowed": True})
        t += 4 * 3600 * 1000

    # 6 crash bars
    target_low = base_p * 0.75
    step_down = (p - target_low) / 6
    for _ in range(6):
        c_open = p
        c_low = p - step_down * 1.05
        c_close = p - step_down
        c_high = p * 1.002
        candles.append({"open_time_ms": t, "open": c_open, "high": c_high, "low": c_low, "close": c_close, "volume": 150.0})
        regimes.append({"open_time_ms": t, "regime": "BEAR_TREND", "grid_allowed": False})
        p = c_close
        t += 4 * 3600 * 1000

    # 6 recovery bars
    step_up = (base_p - p) / 6
    for _ in range(6):
        c_open = p
        c_high = p + step_up * 1.05
        c_close = p + step_up
        c_low = p * 0.998
        candles.append({"open_time_ms": t, "open": c_open, "high": c_high, "low": c_low, "close": c_close, "volume": 150.0})
        regimes.append({"open_time_ms": t, "regime": "BULL_TREND", "grid_allowed": False})
        p = c_close
        t += 4 * 3600 * 1000

    return _build_scenario_dfs(candles, regimes)


def generate_stress_prolonged_trend(base_p: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4 calm bars, then 40 bars of steady gradual grinding trend (+35%)."""
    candles, regimes = [], []
    t = 1000
    p = base_p

    for _ in range(4):
        candles.append({"open_time_ms": t, "open": p, "high": p * 1.01, "low": p * 0.99, "close": p, "volume": 100.0})
        regimes.append({"open_time_ms": t, "regime": "CHOP", "grid_allowed": True})
        t += 4 * 3600 * 1000

    target = base_p * 1.35
    step = (target - p) / 40
    for _ in range(40):
        c_open = p
        c_high = p + step * 1.1
        c_low = p - step * 0.2
        c_close = p + step
        candles.append({"open_time_ms": t, "open": c_open, "high": c_high, "low": c_low, "close": c_close, "volume": 120.0})
        regimes.append({"open_time_ms": t, "regime": "BULL_TREND", "grid_allowed": False})
        p = c_close
        t += 4 * 3600 * 1000

    return _build_scenario_dfs(candles, regimes)


def generate_stress_regime_flicker(base_p: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """30 bars oscillating between CHOP and TREND every 2 bars (churn trap)."""
    candles, regimes = [], []
    t = 1000
    p = base_p

    for i in range(30):
        # Oscillate price sideways
        offset = ((i % 4) - 1.5) * 1.5
        c_open = p + offset
        c_high = c_open + 1.5
        c_low = c_open - 1.5
        c_close = c_open + 0.2
        candles.append({"open_time_ms": t, "open": c_open, "high": c_high, "low": c_low, "close": c_close, "volume": 100.0})

        # Flip regime every 2 bars
        is_chop = (i % 4) < 2
        reg_name = "CHOP" if is_chop else "BULL_TREND"
        regimes.append({"open_time_ms": t, "regime": reg_name, "grid_allowed": is_chop})
        t += 4 * 3600 * 1000

    return _build_scenario_dfs(candles, regimes)


def run_all_stress_tests() -> list[StressScenarioResult]:
    """Executes all stress scenarios across all 4 exit variants."""
    scenarios = [
        ("FLASH_CRASH", generate_stress_flash_crash()),
        ("PARABOLIC_BULL", generate_stress_parabolic_bull()),
        ("V_REVERSAL", generate_stress_v_reversal()),
        ("PROLONGED_TREND", generate_stress_prolonged_trend()),
        ("REGIME_FLICKER", generate_stress_regime_flicker()),
    ]

    variants = [
        ExitVariant.IMMEDIATE_CLOSE,
        ExitVariant.FREEZE,
        ExitVariant.FREEZE_ATR,
        ExitVariant.HYSTERESIS,
    ]

    cfg = ExitEngineConfig(allocated_margin=1000.0, leverage=2.0)
    results: list[StressScenarioResult] = []

    for sc_name, (df, reg_df) in scenarios:
        for var in variants:
            acct = PortfolioAccount(initial_capital=10000.0)
            acct, records = run_continuous_grid_simulation(
                df=df,
                regimes_df=reg_df,
                symbol="BTCUSDT",
                account=acct,
                config=cfg,
                variant=var,
                warmup_bars=2,
            )
            last_p = float(df.iloc[-1]["close"])
            eq = acct.equity({"BTCUSDT": last_p})
            pnl = acct.net_pnl({"BTCUSDT": last_p})

            results.append(
                StressScenarioResult(
                    scenario_name=sc_name,
                    variant=var.value,
                    final_equity=eq,
                    net_pnl=pnl,
                    max_drawdown_pct=acct.max_drawdown_pct,
                    is_bankrupt=acct.is_bankrupt,
                    liquidations_count=acct.liquidation_events_count,
                    total_churn_cost=acct.total_churn_cost,
                    exits_count=len(records),
                )
            )

    return results
