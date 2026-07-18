# Neon Radar Phase 1 Validation Analytical Report

## Executive Summary

This report summarizes the Out-of-Sample stability and feature importance across 5 distinct market regimes (6 evaluated periods) on two core timeframes (1D, 4H). The universe includes `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, and `ADAUSDT`.

## 1. Market Regime Performance (Baseline)

The table below shows the baseline performance of the current trading system (Gross vs Net).

| Period | Timeframe | Trades | Win Rate | Gross PF | Net PF | Gross Exp | Net Exp | Sharpe | p-value |
|---|---|---|---|---|---|---|---|---|---|
| Bull 1 (2020-2021) | 1d | 166 | 56.0% | 1.85 | 1.83 | 4.99% | 4.91% | 0.27 | 0.000 |
| Bull 1 (2020-2021) | 4h | 760 | 48.3% | 1.36 | 1.33 | 1.20% | 1.12% | 0.12 | 0.000 |
| Bear (2021-2022) | 1d | 165 | 41.8% | 1.15 | 1.13 | 0.91% | 0.82% | 0.06 | 0.239 |
| Bear (2021-2022) | 4h | 614 | 40.1% | 0.97 | 0.94 | -0.07% | -0.16% | -0.03 | 0.735 |
| Chop (2023) | 1d | 66 | 33.3% | 0.70 | 0.69 | -1.43% | -1.53% | -0.17 | 0.919 |
| Chop (2023) | 4h | 523 | 38.6% | 0.92 | 0.87 | -0.13% | -0.22% | -0.06 | 0.922 |
| Bull 2 (2023-2024) | 1d | 132 | 47.7% | 1.34 | 1.31 | 1.36% | 1.27% | 0.12 | 0.077 |
| Bull 2 (2023-2024) | 4h | 602 | 48.3% | 1.34 | 1.28 | 0.64% | 0.56% | 0.11 | 0.003 |
| COVID Crash (2020) | 1d | 11 | 27.3% | 0.19 | 0.19 | -9.33% | -9.44% | -0.72 | 0.992 |
| COVID Crash (2020) | 4h | 126 | 38.1% | 0.74 | 0.73 | -1.09% | -1.18% | -0.13 | 0.930 |
| FTX Crash (2022) | 1d | 31 | 48.4% | 1.72 | 1.69 | 2.76% | 2.66% | 0.24 | 0.092 |
| FTX Crash (2022) | 4h | 164 | 36.0% | 0.91 | 0.88 | -0.21% | -0.30% | -0.05 | 0.736 |

**Key Findings (Performance):**

- **Overall Averaged Metrics**: Net PF = 1.07, WR = 42.0%, Net Exp = -0.12%, Sharpe = -0.02.
- *Observation*: Add your analytical notes here after seeing the data.

## 2. Feature Importance (Ablation Analysis)

This section shows which features contributed most to the strategy's edge. A positive Score means the rule is helpful; a negative score means the rule is actively harming the system.

| Feature | Avg Score | Avg $\Delta$PF | Avg $\Delta$Exp | Avg $\Delta$Sharpe | Avg $\Delta$WR |
|---|---|---|---|---|---|
| higher_tf_trend | +0.01 | +0.02 | +0.16% | +0.01 | -0.5% |
| rsi_momentum | +0.00 | -0.02 | +0.12% | +0.02 | +0.9% |
| volume_confirmation | +0.00 | -0.00 | +0.02% | +0.00 | +0.4% |
| volatility_filter | +0.00 | +0.00 | +0.00% | +0.00 | +0.0% |
| funding_rate | +0.00 | +0.00 | +0.00% | +0.00 | +0.0% |
| ema_trend | -0.04 | -0.07 | -0.44% | -0.05 | -0.7% |

**Key Findings (Features):**

- *Observation*: Add your analytical notes here based on feature importance.

## 3. Conclusions & Recommendations

1. **Edge Existence**: (Does the strategy have a statistically significant edge Gross of costs?)

2. **Market Regime Robustness**: (Are there specific regimes where it fails?)

3. **Rule Set Refinement**: (Which rules should be dropped or refactored?)

4. **Next Steps**: (Move to Phase 2: Net-of-Costs simulation and true Walk-Forward validation).
