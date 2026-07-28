"""Enumerations for Market Intelligence."""

from enum import StrEnum


class SourceReliability(StrEnum):
    """Reliability level of the signal source."""

    OFFICIAL = "official"          # Exchanges, protocols themselves
    INSTITUTIONAL = "institutional" # Funds, large data providers (e.g. Glassnode, CryptoQuant)
    ANALYTICS = "analytics"        # Specialized analytics (e.g. CoinGlass)
    RESEARCH = "research"          # Research papers, long-form fundamental analysis
    NEWS = "news"                  # Mainstream/crypto news outlets
    SOCIAL = "social"              # Influencers, X/Twitter, Telegram channels
    ANONYMOUS = "anonymous"        # Unverified rumors, anonymous accounts


class IntelligenceSignalType(StrEnum):
    """Catalog of known intelligence signals."""

    # Technical / Market
    RSI = "rsi"
    EMA_CROSS = "ema_cross"
    MACD = "macd"

    # Microstructure
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    LIQUIDATIONS = "liquidations"

    # On-Chain
    WHALE_ACTIVITY = "whale_activity"
    STABLECOIN_FLOW = "stablecoin_flow"
    EXCHANGE_FLOW = "exchange_flow"

    # Macro / TradFi
    ETF_FLOW = "etf_flow"
    CPI = "cpi"
    FOMC = "fomc"
    DXY = "dxy"

    # Sentiment
    FEAR_AND_GREED = "fear_and_greed"
    SOCIAL_SENTIMENT = "social_sentiment"

    # Market Structure
    BTC_DOMINANCE = "btc_dominance"
    ETH_DOMINANCE = "eth_dominance"


class NarrativeType(StrEnum):
    """Common market narratives."""

    ETF_ACCUMULATION = "etf_accumulation"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    SHORT_SQUEEZE = "short_squeeze"
    LONG_SQUEEZE = "long_squeeze"
    ALT_SEASON = "alt_season"
    BTC_DOMINANCE_EXPANSION = "btc_dominance_expansion"
    STABLECOIN_EXPANSION = "stablecoin_expansion"


class ConsensusDirection(StrEnum):
    """Direction of the market consensus."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    CONFLICTING = "conflicting"
