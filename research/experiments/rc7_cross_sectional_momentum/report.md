# Post-Mortem Continuation: RC7 Cross-Sectional Momentum Control Experiment

## 1. Objective
To isolate whether a fundamental cross-sectional momentum edge exists in the frozen Top-15 Binance Futures Universe, completely independent of the Neon Radar Net Score. 

This experiment seeks to distinguish between two hypotheses:
- **H0**: The chosen Universe lacks a sufficient cross-sectional momentum edge.
- **H1**: A cross-sectional momentum edge exists, but the RC7 Neon Radar Score is incapable of capturing it (due to lack of normalization, absolute score bias, etc.).

## 2. Methodology & Constraints

**Constraints Observed:**
- The production `src/` directory remains strictly untouched.
- The existing Neon Radar scoring is not modified.
- No parameter optimization.
- No lookback tuning post-results.
- No ML, regime filters, RSI, ATR, or normalization techniques.

**Universe & Timeframe:**
- **Universe**: 15 Pairs (BTC, ETH, BNB, XRP, DOGE, ADA, MATIC, DOT, TRX, LTC, SOL, UNI, AVAX, LINK, ATOM).
- **Timeframe**: 4h candles.
- **Period**: 2023-01-01 to 2026-08-01.

**Benchmark Strategy Design:**
- **Signal**: Pure cross-sectional trailing return (Momentum).
- **Lookback Period**: Fixed at **30 days (180 candles of 4h)**. This lookback is fixed *prior* to execution. 30 days is chosen as a standard, widely accepted mid-term momentum window in crypto and traditional finance, long enough to avoid micro-structure noise and short enough to capture shifting capital rotations.
- **Rebalancing**: At each 4h candle close (timestamp $T$), the universe is ranked by trailing 30-day return. 
- **Portfolio Construction**: 
  - **Momentum Top-3**: Long the top 3 assets. Weights are 33.3% each.
  - **Equal-Weight Benchmark**: Long all 15 assets. Weights are 6.66% each.
- **Execution**: Target weights computed at $T$ close are executed at $T+1$ open. The holding period return for a single step is evaluated from $T+1$ open to $T+2$ open.
- **Transaction Costs**: 0.10% (10 bps) per leg, equivalent to the `CostModel` standard Taker (0.05%) + Slippage (0.05%) fee. Applied linearly to weight turnover: `Cost = abs(Weight_T - Weight_{T-1}) * 0.001`.
- **Funding Rates**: As established in the RC7 `run_wfa.py`, funding costs were excluded (`funding_provider=None`). To maintain strict comparability, funding costs are identically omitted here. Missing-data handling replicates the `TradeBacktester._prefetch()` approach.

**WFA & Statistical Discipline:**
- **Walk-Forward Schema**: 6M IS (burn-in) / 2M OOS / step 2M.
- **Primary Comparison**: Momentum Top-3 vs. Equal-Weight Universe.
- **Statistical Test**: Paired daily OOS return differences (Momentum Top-3 vs Equal-Weight), evaluated via a 5-day Block Bootstrap test (10,000 iterations).

## 3. Results & Findings (Post-Audit)

### Core Data & Data Availability Implication
An audit of the `_prefetch()` data-loading conditions revealed a hard constraint: the Binance API `limit=1500` without pagination restricted the 4h timeframe data to approximately the last ~250 days (Nov 2025 – Aug 2026) for the full 14-asset universe. 
- **Earliest Timestamp**: 2024-01-05 (only MATIC; other assets start Nov 2025).
- **Latest Timestamp**: 2026-08-02.
- **OOS Windows**: Early 2024 windows operated on 1 asset. True cross-sectional OOS evaluation only occurred in the ~7 months between Nov 2025 and July 2026.

By replicating RC7's data-feeding environment, this benchmark aligns perfectly with the data RC7 was forced to ingest, but limits conclusions to this specific short window.

### Overall Performance Summary (0.04% Cost Rerun)
*(Note: Initial run at 0.10% costs yielded Top-3 Sharpe -1.64. A control rerun was executed at the approved 0.04% transaction cost to verify robustness).*

| Metric | Pure Momentum (Top-3) | Equal-Weight (Benchmark) |
| --- | --- | --- |
| **Overall Sharpe Ratio** | -1.521 | -1.136 |
| **Overall Profit Factor** | 0.774 | 0.839 |
| **Maximum Drawdown** | -85.28% | -82.28% |
| **Cumulative OOS Return** | -76.56% | -73.06% |
| **Win Rate vs Benchmark** | 11.11% (1/9 Windows) | N/A |

### Statistical Test
- **Paired Daily OOS Returns Difference p-value**: 0.7409. 
  - (H0: Top-3 Momentum <= Equal-Weight Baseline). We fail to reject H0. There is no statistically significant evidence that the Top-3 portfolio outperforms the equal-weight baseline. The sample mean difference was negative.

### Analysis & Resolution of Hypotheses
1. **Hypothesis Resolution**: **H0 is confirmed for the available window.** There is no cross-sectional momentum edge in this universe during the actually evaluated late-2025 to mid-2026 period.
2. **Neon Radar Validation**: The failure of Neon Radar in RC7 during this specific timeframe was likely not due to scoring mechanics, as pure momentum failed equally.
3. **Inference**: The negative sample mean return difference suggests that buying top-performing assets did not yield momentum continuation in this dataset, hinting at a potential mean-reverting environment (though this does not formally prove a tradable mean-reversion alpha).

## 4. Final Verdict
**Cross-sectional momentum Top-3 did not show an edge in the actually available historical window.**

The results demonstrate that pure cross-sectional momentum underperformed an unranked baseline from late 2025 to mid 2026. However, because the data pipeline technically failed to evaluate the 2023-2025 bull cycle, we cannot extrapolate this conclusion to the entire multi-year regime.

**Recommendation:** A robust data engineering fix must be applied to the pipeline to fetch the full 2023-2026 dataset before a final, definitive verdict on the cross-sectional momentum hypothesis can be rendered.

*(Update: This data engineering fix was implemented. See `full_history_postmortem.md` for the final multi-year verdict).*
