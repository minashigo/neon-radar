# RC7 Cross-Sectional Momentum (Control Experiment)

## 1. Hypothesis
This research isolates whether a fundamental cross-sectional momentum edge exists in the chosen crypto universe, independent of the complex Neon Radar scoring formulas.
- **H0**: The chosen Universe lacks a sufficient cross-sectional momentum edge.
- **H1**: A cross-sectional momentum edge exists, but the RC7 Neon Radar Score was incapable of capturing it.

## 2. Methodology
- **Universe**: Frozen Top-15 Binance Futures assets (BTC, ETH, BNB, XRP, DOGE, ADA, MATIC, DOT, TRX, LTC, SOL, UNI, AVAX, LINK, ATOM).
- **Signal**: Pure cross-sectional trailing return (Momentum).
- **Lookback**: 30 days (180 candles of 4h).
- **Portfolio Construction**: Long the top 3 assets (33.3% each) vs. Equal-Weight benchmark of all 15 assets (6.66% each).
- **Rebalancing**: Evaluated at $T$ close, executed at $T+1$ open.
- **Transaction Costs**: 0.04% per leg (0.08% roundtrip / turnover).
- **Funding Rates**: Excluded (matching the original RC7 experiment).
- **WFA Schema**: 6M In-Sample, 2M Out-Of-Sample, 2M step (from 2023-01-01 to 2026-08-01).
- **Statistical Test**: Block bootstrap (5-day blocks) on paired daily OOS return differences (H0: Top-3 <= Equal-Weight).

## 3. Data Coverage
An initial run (`report.md`) revealed a data engineering flaw in the production `_prefetch()` loader, which hard-limited data to the last ~250 days due to Binance's 1500-candle cap. 

A custom paginated historical loader was then implemented (`fetch_full_history.py`) to properly reconstruct the entire `2023-01-01` to `2026-08-01` timeline.
- **Coverage**: 100% (7849 4h candles per active asset).
- **Validation**: 0 gaps, properly sorted, no forward-looking leakage. MATIC correctly stops on its delisting date (2024-09-11).

## 4. Results
Over the 18 Out-Of-Sample windows (from July 2023 to July 2026), the performance was:

| Metric | Pure Momentum (Top-3) | Equal-Weight (Benchmark) |
| --- | --- | --- |
| **Overall Sharpe Ratio** | 0.512 | 0.414 |
| **Maximum Drawdown** | -81.96% | -66.60% |
| **Cumulative OOS Return** | +43.88% | +21.96% |
| **Win Rate vs Benchmark** | 22.22% (4/18 Windows) | N/A |
| **Paired Difference p-value** | 0.3165 | - |

## 5. Verdict
**Verdict: H1 is NOT confirmed. H0 is confirmed.**
RC7 Cross-Sectional Ranking is permanently closed as a research direction in the current setup.

**What is proven:**
Over the full 2023-2026 cycle, the specific Ranked portfolio (30-day Top-3 momentum) fails to statistically outperform the Unranked baseline (Equal-Weight). The block-bootstrap test (p=0.3165) provides no confidence that a cross-sectional edge exists for this specific methodology. While Top-3 captured higher peak returns during the explosive 2023 Q4 bull market, it suffered significantly deeper drawdowns and underperformed the baseline in nearly 78% of all 2-month windows.

**Limitations & Scope:**
- This conclusion applies strictly to the 30-day Top-3 momentum formulation on the frozen Top-15 Binance Futures universe without funding costs. 
- It does NOT assert that *all* cross-sectional momentum approaches (e.g., z-score normalization, volatility parity, or regime filters) are definitively disproven. 
- It does NOT prove the existence of a tradable mean-reversion edge (both Top-3 and EW yielded positive returns over the multi-year cycle).
- Funding costs were excluded. Because Top-3 selects high-momentum assets (which typically bear heavy positive funding fees), including funding would likely penalize Top-3 significantly more than the baseline, further solidifying the failure of H1.
