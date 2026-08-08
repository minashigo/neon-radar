
import pytest

from neon_radar.config.intelligence import ProviderConfig
from neon_radar.domain.market_intelligence.models import PipelineContext
from neon_radar.infrastructure.providers.deribit.provider import DeribitProvider


@pytest.fixture
def provider_config():
    return ProviderConfig(
        max_concurrent_requests=2,
        timeout_seconds=5,
        retry_count=1,
    )


@pytest.fixture
def context():
    return PipelineContext(timestamp=2000, run_id="test", active_providers=("Deribit",))


@pytest.mark.asyncio
async def test_deribit_provider_success(provider_config, context, monkeypatch):
    provider = DeribitProvider(provider_config)

    # Mock _request_with_retry
    async def mock_request(*args, **kwargs):
        url = kwargs.get("url")
        if "get_volatility_index_data" in url:
            class Response:
                def json(self):
                    return {
                        "result": {
                            "data": [
                                [1000, 34.0, 35.0, 33.0, 34.5]
                            ]
                        }
                    }
            return Response()
        elif "get_book_summary_by_currency" in url:
            class Response:
                def json(self):
                    return {
                        "result": [
                            {"instrument_name": "BTC-28AUG26-110000-P", "open_interest": 10.0},
                            {"instrument_name": "BTC-28AUG26-110000-C", "open_interest": 20.0},
                        ]
                    }
            return Response()
        raise ValueError("Unknown URL")

    monkeypatch.setattr(provider, "_request_with_retry", mock_request)

    result = await provider.fetch_signals(context)

    assert result.quality.error_count == 0
    assert len(result.signals) == 2

    types = {s.type for s in result.signals}
    assert "dvol" in types
    assert "put_call_ratio" in types


@pytest.mark.asyncio
async def test_deribit_provider_http_error(provider_config, context, monkeypatch):
    provider = DeribitProvider(provider_config)

    async def mock_request(*args, **kwargs):
        raise RuntimeError("Network error")

    monkeypatch.setattr(provider, "_request_with_retry", mock_request)

    result = await provider.fetch_signals(context)

    assert result.quality.error_count == 2
    assert len(result.signals) == 0


@pytest.mark.asyncio
async def test_deribit_provider_invalid_json(provider_config, context, monkeypatch):
    provider = DeribitProvider(provider_config)

    async def mock_request(*args, **kwargs):
        class Response:
            def json(self):
                raise ValueError("Bad JSON")
        return Response()

    monkeypatch.setattr(provider, "_request_with_retry", mock_request)

    result = await provider.fetch_signals(context)

    assert result.quality.error_count == 2
    assert len(result.signals) == 0


@pytest.mark.asyncio
async def test_deribit_provider_logical_error(provider_config, context, monkeypatch):
    provider = DeribitProvider(provider_config)

    async def mock_request(*args, **kwargs):
        class Response:
            def json(self):
                return {
                    "error": {
                        "code": 10004,
                        "message": "invalid_request"
                    }
                }
        return Response()

    monkeypatch.setattr(provider, "_request_with_retry", mock_request)

    result = await provider.fetch_signals(context)

    assert result.quality.error_count == 2
    assert len(result.signals) == 0
