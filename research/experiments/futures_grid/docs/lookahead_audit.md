# Zero Look-Ahead Bias Audit & Temporal Integrity Specification

## 1. Objective of Audit
To strictly guarantee that every trading decision in the Binance Futures Grid experiment:
1. Operates solely on information physically and computationally available at the bar closing timestamp $T$.
2. Executes orders strictly on or after $T+1$ open.
3. Does not leak future price boundaries, ATR values, or regime transitions.

---

## 2. Vectors of Potential Look-Ahead and Protections

### 2.1. Regime Classification Timing (Strict $T \to T+1$ Contract)
- **Vector**: Calculating regime indicators (ADX, ATR, EMA) using the current forming bar or future bars.
- **Protection**: Bar $i$ regime is evaluated using `candles[0 : i+1]`. The state is stamped at bar $i$ close.
- **Execution**: The decision to launch or terminate the Grid occurs on bar $i+1$ open (`next_candle['open']`).
- **Verification**: `test_regime_future_invariance` verifies that mutating future bars $i+1 \dots i+50$ produces bit-exact identical regime labels at bar $i$.

### 2.2. Dynamic Grid Range Calculation ($[P_{\text{lower}}, P_{\text{upper}}]$)
- **Vector**: Calculating grid bounds using the future candle High/Low or future ATR.
- **Protection**: When deploying a grid at $T+1$ open, the reference price is $P_{\text{open}}(T+1)$ and the ATR is strictly from bar $T$.
  $$P_{\text{lower}} = P_{\text{open}}(T+1) \cdot (1 - \text{range\_pct})$$
  $$P_{\text{upper}} = P_{\text{open}}(T+1) \cdot (1 + \text{range\_pct})$$
  Future price excursions do not alter the deployed grid levels.

### 2.3. Intrabar Price Path Traversal (Conservative Traversal)
- **Vector**: Unrealistic order filling where price dips to buy at the low and sells at the high repeatedly within one bar without considering path.
- **Protection**: Sequential leg evaluation: $Open \to \text{Extremum}_1 \to \text{Extremum}_2 \to Close$. Liquidations and stop-losses are checked at the extrema before credit is given for recovery.

### 2.4. Friction Accounting (Fees, Slippage, Funding)
- Limit grid fills execute with Maker fee ($0.02\%$).
- Regime gate market exits execute with Taker fee ($0.05\%$) plus $0.1\%$ slippage penalty.
- 8-hour funding rates are accrued continuously.
