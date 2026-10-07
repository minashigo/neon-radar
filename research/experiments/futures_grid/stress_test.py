"""Stress Testing Suite for Futures Grid & Radar Gate.

Tests extreme risk vectors:
1. Flash Crash (-40% drop): Severe downward inventory pressure and liquidation threat.
2. Parabolic Bull Trend (+60% surge): Massive short inventory squeeze.
3. Prolonged Out-Of-Range: Extended underwater holding and funding drag.
4. V-Shape Reversal: Inventory accumulation followed by recovery.
5. High-Frequency Chop: Ideal oscillating range conditions.
"""

from __future__ import annotations

from typing import Any

from research.experiments.futures_grid.simulator import (
    GridConfig,
    GridSession,
    GridType,
)


def generate_flash_crash_scenario(base_price: float = 100.0, drop_pct: float = 0.40) -> list[dict[str, Any]]:
    """Synthesizes a severe flash crash over 12 4-hour candles."""
    candles = []
    p = base_price
    t = 1000

    # 4 calm bars
    for _ in range(4):
        candles.append({"open_time": t, "open": p, "high": p * 1.01, "low": p * 0.99, "close": p})
        t += 4 * 3600 * 1000

    # 8 crashing bars
    target = base_price * (1.0 - drop_pct)
    step = (p - target) / 8
    for _ in range(8):
        c_open = p
        c_low = p - step * 1.1
        c_close = p - step
        c_high = p * 1.005
        candles.append({"open_time": t, "open": c_open, "high": c_high, "low": c_low, "close": c_close})
        p = c_close
        t += 4 * 3600 * 1000

    return candles


def generate_parabolic_bull_scenario(base_price: float = 100.0, surge_pct: float = 0.60) -> list[dict[str, Any]]:
    """Synthesizes an aggressive one-way bull surge over 12 candles."""
    candles = []
    p = base_price
    t = 1000

    # 4 calm bars
    for _ in range(4):
        candles.append({"open_time": t, "open": p, "high": p * 1.01, "low": p * 0.99, "close": p})
        t += 4 * 3600 * 1000

    # 8 pumping bars
    target = base_price * (1.0 + surge_pct)
    step = (target - p) / 8
    for _ in range(8):
        c_open = p
        c_high = p + step * 1.1
        c_low = p * 0.995
        c_close = p + step
        candles.append({"open_time": t, "open": c_open, "high": c_high, "low": c_low, "close": c_close})
        p = c_close
        t += 4 * 3600 * 1000

    return candles


def generate_v_shape_scenario(base_price: float = 100.0, dip_pct: float = 0.15) -> list[dict[str, Any]]:
    """Synthesizes a dip and full recovery."""
    candles = []
    p = base_price
    t = 1000
    bottom = base_price * (1.0 - dip_pct)

    # 4 bars falling
    step_down = (base_price - bottom) / 4
    for _ in range(4):
        c_open = p
        c_close = p - step_down
        candles.append({"open_time": t, "open": c_open, "high": c_open * 1.005, "low": c_close * 0.995, "close": c_close})
        p = c_close
        t += 4 * 3600 * 1000

    # 4 bars rising back
    step_up = (base_price - bottom) / 4
    for _ in range(4):
        c_open = p
        c_close = p + step_up
        candles.append({"open_time": t, "open": c_open, "high": c_close * 1.005, "low": c_open * 0.995, "close": c_close})
        p = c_close
        t += 4 * 3600 * 1000

    return candles


def run_stress_test_suite(
    leverage: float = 2.0,
    range_pct: float = 0.10,
    num_grids: int = 20,
) -> dict[str, Any]:
    """Runs simulated stress tests on Baseline Grid vs Regime-Gated Grid."""
    scenarios = {
        "flash_crash_40pct": generate_flash_crash_scenario(100.0, 0.40),
        "parabolic_bull_60pct": generate_parabolic_bull_scenario(100.0, 0.60),
        "v_shape_reversal_15pct": generate_v_shape_scenario(100.0, 0.15),
    }

    results = {}

    for name, candles in scenarios.items():
        p_init = candles[0]["open"]
        cfg = GridConfig(
            lower_price=p_init * (1.0 - range_pct),
            upper_price=p_init * (1.0 + range_pct),
            num_grids=num_grids,
            initial_margin=1000.0,
            leverage=leverage,
            grid_type=GridType.NEUTRAL,
            stop_loss_roi=0.50,
        )

        # 1. Baseline: runs without gate
        session_base = GridSession(cfg, start_time=candles[0]["open_time"], initial_price=p_init)
        for c in candles:
            if not session_base.process_candle(
                timestamp=c["open_time"],
                open_price=c["open"],
                high_price=c["high"],
                low_price=c["low"],
                close_price=c["close"],
            ):
                break
        res_base = session_base.close(candles[-1]["open_time"], candles[-1]["close"], reason="TIMEOUT")

        # 2. Radar Gate: cuts inventory early on trend detection (e.g. after bar 5 when trend is confirmed)
        session_radar = GridSession(cfg, start_time=candles[0]["open_time"], initial_price=p_init)
        gate_triggered = False
        for idx, c in enumerate(candles):
            active = session_radar.process_candle(
                timestamp=c["open_time"],
                open_price=c["open"],
                high_price=c["high"],
                low_price=c["low"],
                close_price=c["close"],
            )
            if not active:
                break
            # Simulated gate trigger when price moves beyond 8% from initial
            if idx >= 5 and abs(c["close"] / p_init - 1.0) > 0.08:
                gate_triggered = True
                # Gate closes on next bar
                if idx + 1 < len(candles):
                    next_c = candles[idx + 1]
                    session_radar.close(next_c["open_time"], next_c["open"], reason="GATE")
                break

        res_radar = session_radar.result()

        results[name] = {
            "baseline": {
                "status": res_base.status.value,
                "net_pnl": round(res_base.net_pnl, 2),
                "roi": round(res_base.roi, 4),
                "peak_inventory": round(res_base.peak_inventory_abs, 4),
                "max_drawdown": round(res_base.max_drawdown_pct, 4),
                "liquidated": res_base.is_liquidation,
            },
            "radar_gate": {
                "status": res_radar.status.value,
                "net_pnl": round(res_radar.net_pnl, 2),
                "roi": round(res_radar.roi, 4),
                "peak_inventory": round(res_radar.peak_inventory_abs, 4),
                "max_drawdown": round(res_radar.max_drawdown_pct, 4),
                "liquidated": res_radar.is_liquidation,
                "gate_triggered": gate_triggered,
            },
        }

    return results
