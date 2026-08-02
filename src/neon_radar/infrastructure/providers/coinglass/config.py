"""Configuration model for CoinGlass Provider."""

from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr


class CoinGlassConfig(BaseModel):
    """Specific configuration for CoinGlass API."""

    api_key: SecretStr = Field(default=SecretStr(""))
    base_url: str = Field(default="https://open-api-v4.coinglass.com")
    retry_count: int = Field(default=3, ge=0)
    poll_interval_seconds: int = Field(default=300, ge=60)

    # We allow overriding the base timeout for CoinGlass API requests
    timeout_seconds: float = Field(default=10.0, ge=1.0)

    # Specific symbols to fetch data for
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
