# Post-Experiment Audit: RC7 Cross-Sectional Momentum

## 1. Audit Verdict
The previous conclusions were mathematically overstated relative to the data actually evaluated. While the benchmark correctly replicated RC7's execution conditions, those conditions contained hidden biases and data limitations.
- **Rerun Required?**: Yes, a single control rerun with 0.04% transaction costs was executed to confirm robustness. The change in transaction costs did not alter the conclusion (Top-3 Sharpe improved slightly to -1.52 but remained worse than the EW baseline of -1.14).
- **Final Status of Hypothesis**: **Inconclusive for 2023-2025; H0 confirmed ONLY for late-2025 to mid-2026.** We cannot claim cross-sectional momentum is permanently "definitively toxic" across the whole 2023-2026 period because we didn't test it.

## 2. What the Experiment ACTUALLY Proved
1. **No Edge in the Available Window**: In the timeframe from **Nov 2025 to Aug 2026** (the only period where the full 15-asset universe was successfully fetched by `_prefetch()`), a pure 30-day cross-sectional momentum portfolio (Top-3) yielded a worse Sharpe ratio (-1.52) than an Equal-Weight portfolio (-1.14). 
2. **Failure to Reject H0**: The paired block bootstrap test (p=0.7409) indicates we fail to reject the null hypothesis that Top-3 returns are $\le$ Equal-Weight returns. There is no statistically significant evidence of outperformance.
3. **Robustness to Costs**: This underperformance holds true regardless of whether transaction costs are 0.10% or 0.04%.
4. **Funding Symmetry**: The omission of funding costs (symmetric to RC7) likely flatters the Top-3 portfolio, meaning the true performance of Top-3 would likely be even worse if funding were included.

## 3. What the Experiment DID NOT Prove
1. **Full 2023-2026 Performance**: Due to the hardcoded `limit=1500` in the Binance API client without a pagination loop, the `_prefetch()` function only loaded the last ~250 days of 4h data (from Nov 2025). The WFA windows from early 2024 were evaluated almost entirely on a single asset (MATIC), rendering them invalid for cross-sectional conclusions. We have **no proof** regarding cross-sectional momentum behavior from Jan 2023 to Oct 2025.
2. **Statistical Underperformance**: Failing to reject H0 (p=0.74) does not formally prove that Top-3 is statistically significantly *worse* than the benchmark; it only proves we cannot confidently claim it is *better*.
3. **Mean Reversion Edge**: While the negative sample mean difference *hints* at mean-reverting behavior (buying top performers lost money relative to holding everything), this experiment does not formally test or prove a tradable mean-reversion alpha.

## 4. Next Steps & Recommendation
We cannot make a final verdict on the cross-sectional momentum hypothesis for the RC7/RC8 project without evaluating the full 2023-2026 bull/bear cycle. 

**Recommendation:** 
We must conduct a data engineering fix to the `_prefetch()` logic (or load data from a complete offline database) to ensure the full 2023-2026 history is populated for all 15 assets. Only then should a final experiment be run to definitively close or validate the cross-sectional momentum hypothesis.
