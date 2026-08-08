
from neon_radar.domain.market_intelligence.enums import IntelligenceSignalType, SourceReliability
from neon_radar.infrastructure.providers.deribit.mapper import (
    map_dvol_to_signal,
    map_put_call_ratio_to_signal,
)


def test_map_dvol_to_signal_success():
    data = {
        "result": {
            "data": [
                [1786060800000, 34.98, 35.0, 33.83, 33.94],
                [1786147200000, 33.94, 34.12, 33.84, 34.11]
            ]
        }
    }
    signal = map_dvol_to_signal(data, "BTC", 1234567890)

    assert signal is not None
    assert signal.type == IntelligenceSignalType.DVOL
    assert signal.direction == 0.0
    assert signal.strength == 0.0
    assert signal.event_timestamp == 1786147200000
    assert signal.ingestion_timestamp == 1234567890
    assert signal.reliability == SourceReliability.ANALYTICS
    assert signal.metadata["symbol"] == "BTC"
    assert signal.metadata["raw_value"] == "34.11"


def test_map_dvol_to_signal_missing_data():
    assert map_dvol_to_signal({}, "BTC", 100) is None
    assert map_dvol_to_signal({"result": {}}, "BTC", 100) is None
    assert map_dvol_to_signal({"result": {"data": []}}, "BTC", 100) is None


def test_map_dvol_to_signal_malformed_data():
    data = {"result": {"data": [[1786147200000, 33.94, 34.12]]}}  # Missing close price
    assert map_dvol_to_signal(data, "BTC", 100) is None

    data2 = {"result": {"data": [[1786147200000, 33.94, 34.12, 33.84, "invalid"]]}}
    assert map_dvol_to_signal(data2, "BTC", 100) is None


def test_map_put_call_ratio_to_signal_success():
    data = {
        "result": [
            {"instrument_name": "BTC-28AUG26-110000-P", "open_interest": 0.2},
            {"instrument_name": "BTC-28AUG26-120000-P", "open_interest": 0.5},
            {"instrument_name": "BTC-28AUG26-110000-C", "open_interest": 1.4},
            {"instrument_name": "BTC-28AUG26-120000-C", "open_interest": 0.0},
            {"instrument_name": "ETH-28AUG26-110000-C", "open_interest": 10.0},  # should be ignored if we rely on name, but API returns only BTC anyway. Wait, our logic just checks -C/-P.
        ]
    }
    # For puts: 0.2 + 0.5 = 0.7. For calls: 1.4 + 0.0 + 10.0 = 11.4
    signal = map_put_call_ratio_to_signal(data, "BTC", 1234567890)

    assert signal is not None
    assert signal.type == IntelligenceSignalType.PUT_CALL_RATIO
    assert signal.direction == 0.0
    assert signal.strength == 0.0
    assert signal.event_timestamp == 1234567890
    assert signal.ingestion_timestamp == 1234567890
    assert signal.metadata["puts_oi"] == "0.7"
    assert signal.metadata["calls_oi"] == "11.4"
    assert signal.metadata["put_call_ratio"] == str(0.7 / 11.4)


def test_map_put_call_ratio_to_signal_missing_data():
    assert map_put_call_ratio_to_signal({}, "BTC", 100) is None
    assert map_put_call_ratio_to_signal({"result": {}}, "BTC", 100) is None
    assert map_put_call_ratio_to_signal({"result": []}, "BTC", 100) is None


def test_map_put_call_ratio_to_signal_malformed_data():
    # missing open_interest
    data = {
        "result": [
            {"instrument_name": "BTC-28AUG26-110000-P"},
            {"instrument_name": "BTC-28AUG26-110000-C", "open_interest": 1.0},
        ]
    }
    signal = map_put_call_ratio_to_signal(data, "BTC", 100)
    assert signal.metadata["puts_oi"] == "0.0"
    assert signal.metadata["calls_oi"] == "1.0"
    assert signal.metadata["put_call_ratio"] == "0.0"

    # invalid open_interest
    data2 = {
        "result": [
            {"instrument_name": "BTC-28AUG26-110000-P", "open_interest": "invalid"},
            {"instrument_name": "BTC-28AUG26-110000-C", "open_interest": 1.0},
        ]
    }
    signal2 = map_put_call_ratio_to_signal(data2, "BTC", 100)
    assert signal2.metadata["puts_oi"] == "0.0"


def test_map_put_call_ratio_to_signal_division_by_zero():
    # Only puts, no calls
    data = {
        "result": [
            {"instrument_name": "BTC-28AUG26-110000-P", "open_interest": 1.0},
        ]
    }
    assert map_put_call_ratio_to_signal(data, "BTC", 100) is None

    # Calls exist but have 0 OI
    data2 = {
        "result": [
            {"instrument_name": "BTC-28AUG26-110000-P", "open_interest": 1.0},
            {"instrument_name": "BTC-28AUG26-110000-C", "open_interest": 0.0},
        ]
    }
    assert map_put_call_ratio_to_signal(data2, "BTC", 100) is None

    # Both 0 OI
    data3 = {
        "result": [
            {"instrument_name": "BTC-28AUG26-110000-P", "open_interest": 0.0},
            {"instrument_name": "BTC-28AUG26-110000-C", "open_interest": 0.0},
        ]
    }
    assert map_put_call_ratio_to_signal(data3, "BTC", 100) is None
