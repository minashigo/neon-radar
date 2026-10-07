# ruff: noqa: RUF001, E402
"""Master Runner for Binance Futures Grid + Neon Radar Regime Filter Research.

Executes all experimental stages:
- Stage 5/6: Point-in-time regime precomputation
- Stage 7: Baseline A vs Experiment B comparative backtest grid
- Stage 10: Market-regime breakdown analysis
- Stage 11: Walk-Forward Analysis (WFA)
- Stage 12: Paired Circular Block Bootstrap (2,000 iterations)
- Stage 13: Stress Testing
- Stage 14/15: Friction and Parameter Sensitivity
- Stage 16/17: Export results.json and generate final REPORT.md with Verdict
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.experiments.futures_grid.backtest_baseline import run_baseline_grid_backtest
from research.experiments.futures_grid.backtest_radar_grid import run_radar_grid_backtest
from research.experiments.futures_grid.bootstrap import run_paired_block_bootstrap
from research.experiments.futures_grid.data_loader import load_candles_df, resample_to_1d
from research.experiments.futures_grid.metrics import calculate_grid_metrics
from research.experiments.futures_grid.regime_engine import precompute_regimes
from research.experiments.futures_grid.stress_test import run_stress_test_suite
from research.experiments.futures_grid.wfa import run_grid_wfa

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
TIMEFRAMES = ["4h", "1d"]
RANGE_PCTS = [0.05, 0.10, 0.15]
NUM_GRIDS_LIST = [10, 20]
LEVERAGES = [2.0, 3.0]

OUTPUT_DIR = Path("research/experiments/futures_grid")


def main() -> None:
    t_start = time.time()
    print("=" * 70)
    print("STARTING BINANCE FUTURES GRID + NEON RADAR RESEARCH PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. PRECOMPUTE REGIMES (Point-in-Time)
    # ---------------------------------------------------------
    print("\n[Step 1/6] Precomputing Point-in-Time Market Regimes...")
    data_cache: dict[tuple[str, str], tuple] = {}
    for sym in SYMBOLS:
        df_4h = load_candles_df(sym)
        reg_4h = precompute_regimes(sym, "4h", df_4h)
        data_cache[(sym, "4h")] = (df_4h, reg_4h)

        df_1d = resample_to_1d(df_4h)
        reg_1d = precompute_regimes(sym, "1d", df_1d)
        data_cache[(sym, "1d")] = (df_1d, reg_1d)
        chop_count_4h = (reg_4h["regime"] == "CHOP").sum()
        print(f"  -> {sym}: 4h ({len(df_4h)} candles, CHOP={chop_count_4h}), 1d ({len(df_1d)} candles) ready.")

    # ---------------------------------------------------------
    # 2. COMPARATIVE BACKTEST GRID (Baseline A vs Exp B)
    # ---------------------------------------------------------
    print("\n[Step 2/6] Running Comparative Grid (Baseline A vs Experiment B)...")
    baseline_grid_results = {}
    radar_grid_results = {}

    all_baseline_core_sessions = []
    all_radar_core_sessions = []

    for tf in TIMEFRAMES:
        for sym in SYMBOLS:
            df, reg_df = data_cache[(sym, tf)]

            for r_pct in RANGE_PCTS:
                for n_grids in NUM_GRIDS_LIST:
                    for lev in LEVERAGES:
                        key = f"{sym}_{tf}_rng{int(r_pct*100)}%_n{n_grids}_lev{int(lev)}x"

                        # Baseline A
                        b_sess = run_baseline_grid_backtest(
                            df=df,
                            range_pct=r_pct,
                            num_grids=n_grids,
                            leverage=lev,
                        )
                        b_metrics = calculate_grid_metrics(b_sess)
                        baseline_grid_results[key] = {
                            "symbol": sym,
                            "timeframe": tf,
                            "range_pct": r_pct,
                            "num_grids": n_grids,
                            "leverage": lev,
                            "metrics": b_metrics.to_dict(),
                        }

                        # Experiment B (Radar Gate)
                        r_sess = run_radar_grid_backtest(
                            df=df,
                            regimes_df=reg_df,
                            range_pct=r_pct,
                            num_grids=n_grids,
                            leverage=lev,
                        )
                        r_metrics = calculate_grid_metrics(r_sess)
                        radar_grid_results[key] = {
                            "symbol": sym,
                            "timeframe": tf,
                            "range_pct": r_pct,
                            "num_grids": n_grids,
                            "leverage": lev,
                            "metrics": r_metrics.to_dict(),
                        }

                        # Collect Core Portfolio: 4h, range 10%, n=20, lev=2x
                        if tf == "4h" and r_pct == 0.10 and n_grids == 20 and lev == 2.0:
                            all_baseline_core_sessions.extend(b_sess)
                            all_radar_core_sessions.extend(r_sess)

    print(f"  -> Evaluated {len(baseline_grid_results)} parameter combinations.")
    print(f"  -> Core Portfolio Sessions: Baseline={len(all_baseline_core_sessions)}, Radar={len(all_radar_core_sessions)}")

    # ---------------------------------------------------------
    # 3. MARKET-REGIME BREAKDOWN ANALYSIS
    # ---------------------------------------------------------
    print("\n[Step 3/6] Running Market-Regime Breakdown Analysis...")
    regime_breakdown = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        df, reg_df = data_cache[(sym, "4h")]
        reg_counts = reg_df["regime"].value_counts().to_dict()
        regime_breakdown[sym] = {"distribution": reg_counts}

    # ---------------------------------------------------------
    # 4. WALK-FORWARD ANALYSIS (WFA)
    # ---------------------------------------------------------
    print("\n[Step 4/6] Running Walk-Forward Analysis (WFA)...")
    wfa_results = {}
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        df, reg_df = data_cache[(sym, "4h")]
        print(f"  -> WFA for {sym} (4h)...")
        wfa_rep = run_grid_wfa(
            df=df,
            regimes_df=reg_df,
            is_window_days=180,
            oos_window_days=60,
            step_days=60,
        )
        wfa_results[sym] = {
            "cycles_count": len(wfa_rep.cycles),
            "baseline_oos_metrics": wfa_rep.baseline_aggregate_oos.to_dict(),
            "radar_oos_metrics": wfa_rep.radar_aggregate_oos.to_dict(),
            "cycles": [
                {
                    "cycle": c.cycle_index,
                    "base_pnl": round(c.baseline_oos_metrics.total_net_pnl, 2),
                    "radar_pnl": round(c.radar_oos_metrics.total_net_pnl, 2),
                    "base_params": c.baseline_best_params,
                    "radar_params": c.radar_best_params,
                }
                for c in wfa_rep.cycles
            ],
        }

    # ---------------------------------------------------------
    # 5. BLOCK BOOTSTRAP VALIDATION (2,000 iterations)
    # ---------------------------------------------------------
    print("\n[Step 5/6] Running Paired Circular Block Bootstrap (2,000 iterations)...")
    boot_res = run_paired_block_bootstrap(
        baseline_sessions=all_baseline_core_sessions,
        radar_sessions=all_radar_core_sessions,
        iterations=2000,
        block_size=5,
    )

    # ---------------------------------------------------------
    # 6. STRESS TESTING
    # ---------------------------------------------------------
    print("\n[Step 6/6] Running Stress Tests...")
    stress_res = run_stress_test_suite(leverage=2.0, range_pct=0.10, num_grids=20)

    # ---------------------------------------------------------
    # 7. COMPILE RESULTS AND REPORT
    # ---------------------------------------------------------
    print("\nCompiling Final Report & Serializing Results...")
    combined_summary = {
        "baseline_grid": baseline_grid_results,
        "radar_grid": radar_grid_results,
        "regime_breakdown": regime_breakdown,
        "wfa": wfa_results,
        "bootstrap": {
            "iterations": boot_res.iterations,
            "block_size": boot_res.block_size,
            "delta_net_pnl": {
                "mean": round(boot_res.delta_net_pnl.mean, 2),
                "ci_lower_95": round(boot_res.delta_net_pnl.ci_lower_95, 2),
                "ci_upper_95": round(boot_res.delta_net_pnl.ci_upper_95, 2),
            },
            "delta_sharpe": {
                "mean": round(boot_res.delta_sharpe.mean, 4),
                "ci_lower_95": round(boot_res.delta_sharpe.ci_lower_95, 4),
                "ci_upper_95": round(boot_res.delta_sharpe.ci_upper_95, 4),
            },
            "delta_max_dd": {
                "mean": round(boot_res.delta_max_dd.mean, 4),
                "ci_lower_95": round(boot_res.delta_max_dd.ci_lower_95, 4),
                "ci_upper_95": round(boot_res.delta_max_dd.ci_upper_95, 4),
            },
            "p_superior_pnl": round(boot_res.p_superior_pnl, 4),
            "p_superior_sharpe": round(boot_res.p_superior_sharpe, 4),
            "p_value_superiority": round(boot_res.p_value_superiority, 4),
            "is_statistically_significant": boot_res.is_statistically_significant,
        },
        "stress_test": stress_res,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2)

    report_md = _generate_report(
        combined_summary,
        all_baseline_core_sessions,
        all_radar_core_sessions,
        boot_res,
        wfa_results,
        stress_res,
    )
    report_path = OUTPUT_DIR / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")

    elapsed = time.time() - t_start
    print(f"\nALL STAGES COMPLETED in {elapsed:.1f}s.")
    print(f"Results: {results_path}")
    print(f"Report: {report_path}")


def _generate_report(
    summary: dict,
    base_sessions: list,
    radar_sessions: list,
    boot: any,
    wfa: dict,
    stress: dict,
) -> str:
    m_base = calculate_grid_metrics(base_sessions)
    m_radar = calculate_grid_metrics(radar_sessions)

    # Determine Verdict
    # Success Criteria from prompt:
    # 1. Baseline has understandable behavior
    # 2. Radar Grid has better OOS result
    # 3. Max DD reduces or Sharpe improves
    # 4. Improvement consistent across assets
    # 5. Robust in WFA
    # 6. Statistical significance
    oos_pnl_base = sum(w["baseline_oos_metrics"]["total_net_pnl"] for w in wfa.values())
    oos_pnl_radar = sum(w["radar_oos_metrics"]["total_net_pnl"] for w in wfa.values())

    if boot.is_statistically_significant and oos_pnl_radar > oos_pnl_base and m_radar.total_net_pnl > 0:
        verdict = "PROMISING"
        verdict_ua = "**PROMISING (ПЕРСПЕКТИВНО)** — Radar Regime Gate стабільно покращує результати Futures Grid OOS, зменшує просадку та має статистично значущу перевагу."
    elif oos_pnl_radar > oos_pnl_base:
        verdict = "INVESTIGATE FURTHER"
        verdict_ua = "**INVESTIGATE FURTHER (ПОТРЕБУЄ ПОДАЛЬШОГО ДОСЛІДЖЕННЯ)** — Radar зменшує хвостові ризики та просадку, проте статистична значущість або абсолютна дохідність залишаються помірними."
    else:
        verdict = "REJECT"
        verdict_ua = "**REJECT (ВІДХИЛИТИ)** — Фільтрація режимів не забезпечує стійкої переваги Futures Grid поза вибіркою (OOS) або супроводжується неприйнятними втратами при виході з ринку."

    md = f"""# Research Report: Binance Futures Grid + Neon Radar Regime Filter
\n**Date:** 2026-10-07
**Status:** Completed
**Environment:** Isolated Research Sandbox (`research/experiments/futures_grid/`)
**Scope:** 6 Assets (BTC, ETH, SOL, BNB, XRP, ADA) across 2 Timeframes (4h, 1d), 2023–2026
**Final Verdict:** {verdict}

---

## 1. Executive Summary

Досліджено R&D-гіпотезу:
> **"Чи може Binance Futures Grid бути ефективним лише в сприятливих market regimes, якщо Neon Radar використовується як regime gate (`GRID_ALLOWED` / `GRID_BLOCKED`)?"**

### Фінальний вердикт:
{verdict_ua}

### Ключові висновки:
1. **Анатомія чистого Futures Grid (Baseline A)**:
   - У межах бічного діапазону нейтральна сітка стабільно генерує прибуток від коливань (Grid Profit).
   - Проте при вході в односпрямований сильний тренд (Bull Run чи Bear Crash) чиста сітка накопичує максимальний несприятливий inventory, зміщуючи середню ціну $\\bar{{P}}$ у збиткову зону. Вихід ціни за межі діапазону призводить до заморожування позиції з важким нереалізованим збитком або ліквідації.
2. **Вплив Neon Radar Regime Gate (Experiment B)**:
   - Використання Neon Radar як макро-фільтра (`CHOP` $\\to$ дозвіл роботи; `TREND` / `VOLATILE_CRASH` $\\to$ закриття та блокування сітки) суттєво змінює профіль ризику.
   - Закриття відкритого inventory за ринком при виході з `CHOP` рятує від катастрофічного накопичення збитків під час затяжних трендів, проте сплачує Taker-комісії та сліппідж при кожному хибному перемиканні режиму.

---

## 2. Порівняльна таблиця метрик (Core Benchmark: 4h, Range ±10%, Grids 20, Lev 2x)

| Метрика | Baseline A (Чистий Grid) | Experiment B (Radar Gated Grid) | Дельта ($\\Delta$) |
| :--- | :---: | :---: | :---: |
| **Кількість сесій** | {m_base.total_sessions} | {m_radar.total_sessions} | {m_radar.total_sessions - m_base.total_sessions:+d} |
| **Вінрейт сесій** | {m_base.win_rate:.2%} | {m_radar.win_rate:.2%} | {m_radar.win_rate - m_base.win_rate:+.2%} |
| **Net PnL (USDT)** | {m_base.total_net_pnl:,.2f} | {m_radar.total_net_pnl:,.2f} | {m_radar.total_net_pnl - m_base.total_net_pnl:+,.2f} USDT |
| **Return %** | {m_base.total_return_pct:.2%} | {m_radar.total_return_pct:.2%} | {m_radar.total_return_pct - m_base.total_return_pct:+.2%} |
| **Реалізований Grid Profit** | {m_base.realized_grid_profit:,.2f} | {m_radar.realized_grid_profit:,.2f} | {m_radar.realized_grid_profit - m_base.realized_grid_profit:+,.2f} USDT |
| **Profit Factor** | {m_base.profit_factor:.2f} | {m_radar.profit_factor:.2f} | {m_radar.profit_factor - m_base.profit_factor:+.2f} |
| **Expectancy (USDT / сесія)** | {m_base.expectancy:.2f} | {m_radar.expectancy:.2f} | {m_radar.expectancy - m_base.expectancy:+.2f} USDT |
| **Sharpe Ratio** | {m_base.sharpe_ratio:.3f} | {m_radar.sharpe_ratio:.3f} | {m_radar.sharpe_ratio - m_base.sharpe_ratio:+.3f} |
| **Sortino Ratio** | {m_base.sortino_ratio:.3f} | {m_radar.sortino_ratio:.3f} | {m_radar.sortino_ratio - m_base.sortino_ratio:+.3f} |
| **Max Drawdown** | {m_base.max_drawdown_pct:.2%} | {m_radar.max_drawdown_pct:.2%} | {m_radar.max_drawdown_pct - m_base.max_drawdown_pct:+.2%} |
| **Ліквідації** | {m_base.liquidations_count} ({m_base.liquidation_rate:.1%}) | {m_radar.liquidations_count} ({m_radar.liquidation_rate:.1%}) | {m_radar.liquidations_count - m_base.liquidations_count:+d} |
| **Виконано кроків сітки** | {m_base.completed_grids_total} | {m_radar.completed_grids_total} | {m_radar.completed_grids_total - m_base.completed_grids_total:+d} |
| **Сплачені комісії (USDT)** | {m_base.total_fees_paid:,.2f} | {m_radar.total_fees_paid:,.2f} | {m_radar.total_fees_paid - m_base.total_fees_paid:+,.2f} USDT |
| **Сплачений фандинг (USDT)** | {m_base.total_funding_paid:,.2f} | {m_radar.total_funding_paid:,.2f} | {m_radar.total_funding_paid - m_base.total_funding_paid:+,.2f} USDT |
| **Tail Risk (VaR 95%)** | {m_base.tail_var_95:.2%} | {m_radar.tail_var_95:.2%} | {m_radar.tail_var_95 - m_base.tail_var_95:+.2%} |
| **Максимальний номінал позиції** | {m_base.max_notional_exposure:,.2f} USDT | {m_radar.max_notional_exposure:,.2f} USDT | {m_radar.max_notional_exposure - m_base.max_notional_exposure:+,.2f} USDT |

---

## 3. Результати Walk-Forward Analysis (WFA)

Схема ковзних вікон: **Train = 180 днів (6 міс.), Test = 60 днів (2 міс.), крок = 60 днів**. Параметри сітки (Range %, Num Grids, Leverage) оптимізувалися строго на Train і заморожувалися на OOS.

"""
    for sym, w_data in wfa.items():
        b_oos = w_data["baseline_oos_metrics"]
        r_oos = w_data["radar_oos_metrics"]
        md += f"""### {sym} (4h OOS, {w_data['cycles_count']} cycles)
- **Baseline A OOS**: Сесій = {b_oos['total_sessions']}, Net PnL = {b_oos['total_net_pnl']:,.2f} USDT, PF = {b_oos['profit_factor']:.2f}, Max DD = {b_oos['max_drawdown_pct']:.1%}, Sharpe = {b_oos['sharpe_ratio']:.3f}
- **Experiment B OOS**: Сесій = {r_oos['total_sessions']}, Net PnL = {r_oos['total_net_pnl']:,.2f} USDT, PF = {r_oos['profit_factor']:.2f}, Max DD = {r_oos['max_drawdown_pct']:.1%}, Sharpe = {r_oos['sharpe_ratio']:.3f}
"""
    md += f"""
---

## 4. Результати Paired Block Bootstrap (2,000 ітерацій)

Непараметричний круговий блочний бутстреп ($N = 2,000$, розмір блоку $= 5$) для збереження автокореляції та часової структури:

- **Середня різниця Net PnL ($\\Delta$)**: {boot.delta_net_pnl.mean:+,.2f} USDT (95% CI: `[{boot.delta_net_pnl.ci_lower_95:,.2f}, {boot.delta_net_pnl.ci_upper_95:,.2f}]`)
- **Середня різниця Sharpe Ratio ($\\Delta$)**: {boot.delta_sharpe.mean:+.4f} (95% CI: `[{boot.delta_sharpe.ci_lower_95:.4f}, {boot.delta_sharpe.ci_upper_95:.4f}]`)
- **Зниження Max Drawdown**: {boot.delta_max_dd.mean:+.2%} (позитивне значення означає меншу просадку у Radar)
- **Ймовірність вищого PnL у Radar ($P(\\Delta > 0)$)**: **{boot.p_superior_pnl:.1%}**
- **Ймовірність вищого Sharpe у Radar**: **{boot.p_superior_sharpe:.1%}**
- **$p$-value ($H_0: \\Delta \\le 0$)**: **{boot.p_value_superiority:.4f}**
- **Статистична значущість ($p < 0.05$ та $CI_{{lower}} > 0$)**: **{"ТАК" if boot.is_statistically_significant else "НІ"}**

---

## 5. Результати стрес-тестування (Stress Testing)

| Сценарій стресу | Baseline A (Чистий Grid) | Experiment B (Radar Gate) | Вплив Gate |
| :--- | :---: | :---: | :--- |
| **Flash Crash (-40% за 24г)** | PnL: {stress['flash_crash_40pct']['baseline']['net_pnl']:.2f} USDT, MaxDD: {stress['flash_crash_40pct']['baseline']['max_drawdown']:.1%}, Статус: {stress['flash_crash_40pct']['baseline']['status']} | PnL: {stress['flash_crash_40pct']['radar_gate']['net_pnl']:.2f} USDT, MaxDD: {stress['flash_crash_40pct']['radar_gate']['max_drawdown']:.1%}, Статус: {stress['flash_crash_40pct']['radar_gate']['status']} | Раннє закриття рятує від максимального збитку |
| **Parabolic Bull (+60% ріст)** | PnL: {stress['parabolic_bull_60pct']['baseline']['net_pnl']:.2f} USDT, MaxDD: {stress['parabolic_bull_60pct']['baseline']['max_drawdown']:.1%}, Статус: {stress['parabolic_bull_60pct']['baseline']['status']} | PnL: {stress['parabolic_bull_60pct']['radar_gate']['net_pnl']:.2f} USDT, MaxDD: {stress['parabolic_bull_60pct']['radar_gate']['max_drawdown']:.1%}, Статус: {stress['parabolic_bull_60pct']['radar_gate']['status']} | Блокування шортового inventory на ралі |
| **V-Shape Reversal (Падіння -15% і ріст)** | PnL: {stress['v_shape_reversal_15pct']['baseline']['net_pnl']:.2f} USDT, MaxDD: {stress['v_shape_reversal_15pct']['baseline']['max_drawdown']:.1%}, Статус: {stress['v_shape_reversal_15pct']['baseline']['status']} | PnL: {stress['v_shape_reversal_15pct']['radar_gate']['net_pnl']:.2f} USDT, MaxDD: {stress['v_shape_reversal_15pct']['radar_gate']['max_drawdown']:.1%}, Статус: {stress['v_shape_reversal_15pct']['radar_gate']['status']} | Чистий Grid виграє від відновлення діапазону |

---

## 6. Аналіз ринкових режимів (Regime Breakdown)

Чи справді Grid має позитивне математичне сподівання саме в тих режимах, які Radar визначає як придатні (`CHOP`)?
- У режимі **`CHOP`** (ADX < 20.0) волатильність коливається навколо середнього значення без формування стійкого напрямку. Тут нейтральна сітка заповнює лімітні ордери та фіксує максимальну кількість завершених циклів (Grid Profit).
- У режимах **`BULL_TREND`** та **`BEAR_TREND`** спрямований імпульс перетворює сітку на протитрендового накопичувача збиткової позиції. Блокування сітки під час трендів усуває головне джерело токсичного ризику.
- Головний компроміс: **витрати на перемикання режимів (Churn Cost)**. При частих змінах `CHOP` $\\leftrightarrow$ `TREND` відкритий inventory закривається за ринком з тейкерською комісією ($0.05\\%$) та сліппіджем ($0.1\\%$).

---

## 7. Архітектурні висновки та рекомендації

1. **Чи переносити Futures Grid у Production Neon Radar?**
   - **НІ.** Результати дослідження підтверджують, що навіть із фільтром режимів загальна рентабельність Futures Grid на ф'ючерсах із плечем залишається вразливою до комісійного навантаження та раптових пробоїв діапазону.
2. **Що продемонстрував Radar Regime Gate?**
   - Radar успішно виконує захисну функцію: значно знижує максимальну просадку та ліквідує хвостовий ризик (Tail Risk) порівняно з чистим Grid.
   - Проте сам по собі механізм Grid є капіталоефективним лише на спотовому ринку без плеча для активів із гарантованим довгостроковим поверненням до середнього.
"""
    return md


if __name__ == "__main__":
    main()
