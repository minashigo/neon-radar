# Full-History Control Experiment: RC7 Cross-Sectional Momentum

## 1. Objective
Following the discovery that the original RC7 experiment (and its first benchmark) was fundamentally restricted by Binance's 1500-candle hard-limit, this test implements a full paginated data fetch to assess cross-sectional momentum on the true 2023-2026 bull/bear cycle. 

## 2. Constraints & Methodology
- **Production Code Untouched**: The experiment operates entirely in the `research/` directory. `TradeBacktester` and Binance Client were not modified.
- **Strategy Design Unchanged**: 
  - Lookback: 30-day (180 candles of 4h).
  - Universe: Frozen Top-15 Binance Futures assets.
  - Rebalance: 4H, Top-3 vs Equal-Weight.
- **Funding**: Excluded (matching RC7).
- **Transaction Costs**: 0.04% per leg (0.08% roundtrip / rebalance turnover).
- **WFA Stitching**: 6M IS, 2M OOS, 2M step. Block bootstrap 5-day on paired daily returns difference.

## 3. Data Validation & Coverage
A paginated loader successfully reconstructed the entire 2023-2026 history without hitting the Binance API 1500-candle limit.
- **Coverage**: 2023-01-01 to 2026-08-01 (7849 4h candles per active asset).
- **Universe**: 14 assets fully active. MATIC correctly stops on its delisting date (2024-09-11).
- **Validation**: 0 gaps, properly sorted, no forward-looking leakage.

## 4. WFA Windows
18 OOS windows of 2 months each were stitched from 2023-07-01 to 2026-07-01.

## 5. Performance Metrics
| Metric | Pure Momentum (Top-3) | Equal-Weight (Benchmark) |
| --- | --- | --- |
| **Overall Sharpe Ratio** | 0.512 | 0.414 |
| **Overall Profit Factor** | 1.083 | 1.063 |
| **Maximum Drawdown** | -81.96% | -66.60% |
| **Cumulative OOS Return** | +43.88% | +21.96% |
| **Win Rate vs Benchmark** | 22.22% (4/18 Windows) | N/A |

### Statistical Test
- **Paired Daily OOS Returns Difference p-value (H0: Top-3 <= EW)**: 0.3165

## 6. Analysis & Verdict
**Verdict: Failure of this specific cross-sectional momentum hypothesis.**

**What is proven:**
1. Over the full cycle, the Ranked portfolio (Top-3) fails to statistically outperform the Unranked baseline (Equal-Weight). The block-bootstrap test (p=0.3165) gives no confidence that the edge is real.
2. While Top-3 captured higher peak returns during the explosive 2023 Q4 bull market, it suffered significantly deeper drawdowns (-82% vs -66%) and underperformed the baseline in nearly 78% of all 2-month periods.

**What is not proven:**
1. This does not prove that ALL cross-sectional approaches (e.g. z-score normalization, volatility parity, or regime-filtered momentum) are toxic.
2. It does not prove mean-reversion, as both portfolios ended positive over the 3.5 years.

**Limitations:**
1. Funding costs remain excluded. Given that Top-3 naturally tilts towards high-momentum assets (which often command high positive funding rates), including funding would likely penalize Top-3 significantly more than the Equal-Weight baseline.

**Final Status**: Close this specific cross-sectional hypothesis. No further automated variants of this exact setup should be run.
