from neon_radar.infrastructure.providers.alternative_me.mapper import (
    map_historical_fng_to_observations,
)
from neon_radar.infrastructure.providers.deribit.mapper import map_historical_dvol_to_observations


def test_map_historical_fng():
    data = {
        "name": "Fear and Greed Index",
        "data": [
            {
                "value": "45",
                "value_classification": "Fear",
                "timestamp": "1719792000",
                "time_until_update": "12345"
            },
            {
                "value": "55",
                "value_classification": "Greed",
                "timestamp": "1719705600"
            }
        ]
    }

    observations = map_historical_fng_to_observations(data, ingestion_timestamp=1719800000000)

    assert len(observations) == 2

    # Assert chronological sorting
    assert observations[0].observation_timestamp == 1719705600 * 1000
    assert observations[1].observation_timestamp == 1719792000 * 1000

    # Assert available_at is equal to observation_timestamp for F&G
    assert observations[0].available_at == 1719705600 * 1000
    assert observations[1].available_at == 1719792000 * 1000

    # Assert direction mapping
    assert observations[0].signal.direction == (55.0 - 50.0) / 50.0
    assert observations[1].signal.direction == (45.0 - 50.0) / 50.0

def test_map_historical_dvol():
    data = {
        "jsonrpc": "2.0",
        "result": {
            "data": [
                [1719705600000, 50.0, 55.0, 48.0, 52.0],
                [1719792000000, 52.0, 54.0, 51.0, 53.0]
            ]
        }
    }

    resolution_ms = 86400000
    observations = map_historical_dvol_to_observations(data, "BTC", 1719800000000, resolution_ms)

    assert len(observations) == 2

    assert observations[0].observation_timestamp == 1719705600000
    assert observations[1].observation_timestamp == 1719792000000

    # Assert available_at is observation + resolution
    assert observations[0].available_at == 1719705600000 + resolution_ms
    assert observations[1].available_at == 1719792000000 + resolution_ms

    # Assert metadata raw_value
    assert observations[0].signal.metadata["raw_value"] == "52.0"
    assert observations[1].signal.metadata["raw_value"] == "53.0"
