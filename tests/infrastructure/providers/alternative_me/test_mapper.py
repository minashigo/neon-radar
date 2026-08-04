from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType
from neon_radar.infrastructure.providers.alternative_me.mapper import map_fng_to_signal


def test_map_fng_to_signal_extreme_greed():
    data = {
        "name": "Fear and Greed Index",
        "data": [
            {
                "value": "100",
                "value_classification": "Extreme Greed",
                "timestamp": "1700000000",
                "time_until_update": "1000",
            }
        ],
        "metadata": {"error": None},
    }
    signal = map_fng_to_signal(data, 2000)
    assert signal is not None
    assert signal.type == IntelligenceSignalType.FEAR_AND_GREED
    assert signal.direction == 1.0  # (100 - 50) / 50 = 1.0
    assert signal.strength == 1.0
    assert signal.metadata["symbol"] == "BTC"
    assert signal.metadata["raw_value"] == "100"


def test_map_fng_to_signal_extreme_fear():
    data = {
        "data": [{"value": "0", "value_classification": "Extreme Fear"}],
        "metadata": {"error": None},
    }
    signal = map_fng_to_signal(data, 2000)
    assert signal is not None
    assert signal.direction == -1.0  # (0 - 50) / 50 = -1.0
    assert signal.strength == 1.0


def test_map_fng_to_signal_neutral():
    data = {
        "data": [{"value": "50", "value_classification": "Neutral"}],
        "metadata": {"error": None},
    }
    signal = map_fng_to_signal(data, 2000)
    assert signal is not None
    assert signal.direction == 0.0
    assert signal.strength == 0.0


def test_map_fng_to_signal_out_of_bounds():
    # Should bound to 1.0
    data = {"data": [{"value": "150"}]}
    signal = map_fng_to_signal(data, 2000)
    assert signal is not None
    assert signal.direction == 1.0

    # Should bound to -1.0
    data = {"data": [{"value": "-50"}]}
    signal = map_fng_to_signal(data, 2000)
    assert signal is not None
    assert signal.direction == -1.0


def test_map_fng_to_signal_empty_data():
    signal = map_fng_to_signal({"data": []}, 2000)
    assert signal is None

    signal2 = map_fng_to_signal({}, 2000)
    assert signal2 is None


def test_map_fng_to_signal_logical_error():
    data = {"data": [{"value": "50"}], "metadata": {"error": "Rate limit exceeded"}}
    signal = map_fng_to_signal(data, 2000)
    assert signal is None
