# Market Context Contract (FeatureSchema v1)

## 1. Overview
The Market Context layer is responsible for fetching, normalizing, caching, and serving derivatives microstructure data from exchanges (e.g. Binance Futures) to the Neon Radar trading engine. 

The domain layer explicitly isolates:
1. **Raw DTOs**: 1-to-1 mappings of exchange JSON responses.
2. **Normalizers**: Pure functions extracting raw values into standard `Context` objects.
3. **Domain Models**: Immutable, strongly-typed Context objects.

## 2. The Time Model (Point-in-Time Correctness)
To guarantee the absolute integrity of backtests, every piece of contextual data is stamped with a `TimeContext`:
- `event_time` (Unix ms, UTC): When the event actually occurred on the exchange (e.g. the funding rate settlement time).
- `publish_time` (Unix ms, UTC): When the exchange made this data publicly accessible (for live events, often equals `event_time`, but can be delayed).
- `ingest_time` (Unix ms, UTC): When Neon Radar received and normalized the data.

**Rule**: The Analysis Engine MUST NEVER use data where `publish_time > current_simulation_timestamp`. 

## 3. Context Structures

### 3.1. FundingContext
Represents the current funding rate for perpetual swaps.
- `raw_funding` (float): The actual decimal rate (e.g. 0.0001 for 0.01%).
- `funding_8h_equiv` (float): Normalized to an 8-hour rate (Binance uses 8h natively, but others might use 1h. Normalizer handles this).
- `annualized_apr` (float): `funding_8h_equiv * 3 * 365`.
- `mark_price` (float): Mark price at the time of calculation.
- `next_funding_time_utc` (int): Unix timestamp of next expected funding event.

### 3.2. OpenInterestContext
Represents the amount of open positions.
- `oi_coin` (float): The open interest measured in the base asset (e.g. BTC).
- `oi_usd_notional` (float): The open interest measured in USD/USDT. This is necessary because if price doubles, USD-denominated OI doubles even if no new capital enters the market. The `oi_usd_notional` helps track actual capital flow.

### 3.3. LongShortRatioContext
Represents the ratio of long vs short positions (usually accounts or positions based).
- `long_pct` (float): Percentage of longs (e.g. 0.55 for 55%).
- `short_pct` (float): Percentage of shorts (e.g. 0.45 for 45%).
- `ls_ratio` (float): Absolute ratio (`long_pct / short_pct`).

### 3.4. TakerFlowContext
Represents aggressive market buy/sell volume.
- `buy_volume` (float): Base asset volume of taker buy orders.
- `sell_volume` (float): Base asset volume of taker sell orders.
- `net_buy_volume` (float): `buy_volume - sell_volume`.

### 3.5. LiquidationContext
Represents liquidated volumes.
- `long_liquidations` (float): Base asset volume of liquidated longs.
- `short_liquidations` (float): Base asset volume of liquidated shorts.

## 4. Normalization Rules
1. All times must be resolved to Unix milliseconds (UTC).
2. Percentages from API strings (like `"55.2"`) must be converted to float decimals (`0.552`).
3. If an exchange only provides OI in USD and current index price, `oi_coin` must be derived: `oi_usd / index_price`.

## 5. MarketContext Aggregate
`MarketContext` groups all context models for a single symbol at a specific time. 

```python
class MarketContext:
    symbol: Symbol
    schema_version: str = "FeatureSchema_v1"
    timestamp: int  # Evaluation timestamp

    # Contexts are optional; a provider may fail or API might not support it
    funding: FundingContext | None = None
    open_interest: OpenInterestContext | None = None
    ls_ratio: LongShortRatioContext | None = None
    taker_flow: TakerFlowContext | None = None
    liquidations: LiquidationContext | None = None
```
