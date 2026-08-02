"""Tests for CoinGlass mapper."""

from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType
from neon_radar.infrastructure.providers.coinglass.mapper import map_long_short_ratio_to_signal


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

    # Should pick the latest one based on time
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
