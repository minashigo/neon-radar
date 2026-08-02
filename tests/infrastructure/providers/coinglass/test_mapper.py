"""Tests for CoinGlass mapper."""

from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType
from neon_radar.infrastructure.providers.coinglass.mapper import (
    map_funding_rate_to_signal,
    map_liquidations_to_signal,
    map_long_short_ratio_to_signal,
    map_open_interest_to_signal,
)


def test_map_long_short_ratio_valid_data():
    data = {
        "code": "0",
        "msg": "success",
        "data": [
            {
                "time": 1741604400000,
                "global_account_long_percent": 73.88,
                "global_account_short_percent": 26.12,
                "global_account_long_short_ratio": 2.83,
            },
            {
                "time": 1741608000000,  # Newer
                "global_account_long_percent": 60.00,
                "global_account_short_percent": 40.00,
                "global_account_long_short_ratio": 1.5,
            },
        ],
    }

    signal = map_long_short_ratio_to_signal(data, "BTCUSDT", 1741608000010)
    assert signal is not None
    assert signal.type == IntelligenceSignalType.LONG_SHORT_RATIO
    assert signal.event_timestamp == 1741608000000
    assert signal.ingestion_timestamp == 1741608000010
    # direction = (60 - 40) / 100 = 0.2
    assert signal.direction == 0.2
    assert signal.strength == 0.2
    assert signal.metadata["symbol"] == "BTCUSDT"
    assert signal.metadata["long_percent"] == "60.0"


def test_map_long_short_ratio_empty_data():
    data = {"code": "0", "msg": "success", "data": []}
    signal = map_long_short_ratio_to_signal(data, "BTCUSDT", 1000)
    assert signal is None


def test_map_long_short_ratio_missing_fields():
    data = {
        "data": [
            {
                "time": 1741604400000,
                # Missing percentages
            }
        ]
    }
    signal = map_long_short_ratio_to_signal(data, "BTCUSDT", 1000)
    assert signal is None


def test_map_funding_rate_valid_positive():
    data = {"data": [{"time": 1000, "close": "0.0001"}]}
    signal = map_funding_rate_to_signal(data, "BTCUSDT", 1010)
    assert signal is not None
    assert signal.type == IntelligenceSignalType.FUNDING
    assert signal.direction == 1.0
    assert signal.strength == 0.1  # 0.0001 * 1000.0


def test_map_funding_rate_capped():
    data = {"data": [{"time": 1000, "close": "0.005"}]}  # 0.5%
    signal = map_funding_rate_to_signal(data, "BTCUSDT", 1010)
    assert signal is not None
    assert signal.direction == 1.0
    assert signal.strength == 1.0  # capped


def test_map_funding_rate_valid_negative():
    data = {"data": [{"time": 1000, "close": "-0.0002"}]}
    signal = map_funding_rate_to_signal(data, "BTCUSDT", 1010)
    assert signal is not None
    assert signal.direction == -1.0
    assert signal.strength == 0.2


def test_map_funding_rate_invalid():
    signal = map_funding_rate_to_signal({"data": [{"time": 1000, "close": "abc"}]}, "BTCUSDT", 1000)
    assert signal is None


def test_map_open_interest_valid_data():
    data = {
        "data": [
            {
                "time": 1000,
                "close": "100000",
            },
            {
                "time": 2000,
                "close": "105000",  # +5%
            },
        ]
    }
    signal = map_open_interest_to_signal(data, "BTCUSDT", 2010)
    assert signal is not None
    assert signal.type == IntelligenceSignalType.OPEN_INTEREST
    assert signal.direction == 1.0
    assert signal.strength == 1.0  # 5% * 20 = 1.0
    assert signal.metadata["delta_pct"] == "0.05"


def test_map_open_interest_negative():
    data = {
        "data": [
            {
                "time": 1000,
                "close": "100000",
            },
            {
                "time": 2000,
                "close": "95000",  # -5%
            },
        ]
    }
    signal = map_open_interest_to_signal(data, "BTCUSDT", 2010)
    assert signal is not None
    assert signal.direction == -1.0
    assert signal.strength == 1.0  # |-5%| * 20 = 1.0


def test_map_open_interest_insufficient_data():
    # Only one point, can't calc delta
    data = {"data": [{"time": 1000, "close": "100000"}]}
    signal = map_open_interest_to_signal(data, "BTCUSDT", 2010)
    assert signal is None


def test_map_open_interest_zero_prev():
    # Previous OI is 0, would cause division by zero
    data = {"data": [{"time": 1000, "close": "0"}, {"time": 2000, "close": "100000"}]}
    signal = map_open_interest_to_signal(data, "BTCUSDT", 2010)
    assert signal is None


def test_map_liquidations_valid_bullish():
    data = {
        "data": [
            {
                "time": 1000,
                "long_liquidation_usd": "10000",
                "short_liquidation_usd": "30000",  # More short liquidations
            }
        ]
    }
    signal = map_liquidations_to_signal(data, "BTCUSDT", 1010)
    assert signal is not None
    assert signal.type == IntelligenceSignalType.LIQUIDATIONS
    assert signal.direction == 0.5  # (30000 - 10000) / 40000 = 0.5
    assert signal.strength == 0.5


def test_map_liquidations_valid_bearish():
    data = {
        "data": [
            {
                "time": 1000,
                "long_liquidation_usd": "40000",  # More long liquidations
                "short_liquidation_usd": "10000",
            }
        ]
    }
    signal = map_liquidations_to_signal(data, "BTCUSDT", 1010)
    assert signal is not None
    assert signal.direction == -0.6  # (10000 - 40000) / 50000 = -0.6
    assert signal.strength == 0.6


def test_map_liquidations_zero_total():
    data = {
        "data": [
            {
                "time": 1000,
                "long_liquidation_usd": "0",
                "short_liquidation_usd": "0",
            }
        ]
    }
    signal = map_liquidations_to_signal(data, "BTCUSDT", 1010)
    assert signal is None


def test_map_liquidations_negative_values():
    data = {
        "data": [
            {
                "time": 1000,
                "long_liquidation_usd": "-100",
                "short_liquidation_usd": "200",
            }
        ]
    }
    signal = map_liquidations_to_signal(data, "BTCUSDT", 1010)
    assert signal is None


def test_map_liquidations_invalid():
    signal = map_liquidations_to_signal({"data": []}, "BTCUSDT", 1000)
    assert signal is None

    signal2 = map_liquidations_to_signal({"data": [{"time": 1000}]}, "BTCUSDT", 1000)
    assert signal2 is None
