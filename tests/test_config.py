"""Tests for the configuration layer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from pydantic import ValidationError

from neon_radar.config.loader import ConfigLoader
from neon_radar.config.models import (
    ApiConfig,
    AppConfig,
    SymbolConfig,
    TimeFrame,
    UiConfig,
)
from neon_radar.domain.exceptions import ConfigError


class TestSymbolConfig:
    """Symbol-level validation rules."""

    def test_uppercases_symbol(self) -> None:
        s = SymbolConfig(symbol="btcusdt")
        assert s.symbol == "BTCUSDT"

    def test_strips_whitespace(self) -> None:
        s = SymbolConfig(symbol="  ETHUSDT  ")
        assert s.symbol == "ETHUSDT"

    def test_rejects_non_alphanumeric(self) -> None:
        with pytest.raises(ValidationError):
            SymbolConfig(symbol="BTC-USDT")

    def test_is_frozen(self) -> None:
        s = SymbolConfig(symbol="BTCUSDT")
        with pytest.raises(ValidationError):
            s.symbol = "ETHUSDT"  # type: ignore[misc]


class TestTimeFrame:
    """TimeFrame enum."""

    def test_values_match_binance(self) -> None:
        assert TimeFrame.D1.value == "1d"
        assert TimeFrame.H4.value == "4h"

    def test_seconds_for_common_frames(self) -> None:
        assert TimeFrame.M1.seconds == 60
        assert TimeFrame.H4.seconds == 14_400
        assert TimeFrame.D1.seconds == 86_400


class TestUiConfig:
    def test_window_size_too_small_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UiConfig(window_size=(100, 100))

    def test_default_candles_in_range(self) -> None:
        ui = UiConfig()
        assert 50 <= ui.default_candles <= 2000


class TestAppConfig:
    def test_minimal_valid_config(self) -> None:
        cfg = AppConfig(symbols=[SymbolConfig(symbol="BTCUSDT")])
        assert len(cfg.enabled_symbols()) == 1

    def test_duplicate_symbols_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(
                symbols=[
                    SymbolConfig(symbol="BTCUSDT"),
                    SymbolConfig(symbol="btcusdt"),  # same after upper
                ]
            )

    def test_default_timeframe_must_be_in_list(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(
                symbols=[SymbolConfig(symbol="BTCUSDT")],
                timeframes=[TimeFrame.H4],
                ui=UiConfig(default_timeframe=TimeFrame.D1),
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig.model_validate(
                {"symbols": [{"symbol": "BTCUSDT"}], "unknown": 1}
            )


class TestConfigLoader:
    def test_loads_example_config(self, example_config_path: Path) -> None:
        cfg = ConfigLoader(example_config_path).load()
        assert isinstance(cfg, AppConfig)
        assert len(cfg.symbols) > 0
        assert TimeFrame.D1 in cfg.timeframes
        assert TimeFrame.H4 in cfg.timeframes

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            ConfigLoader(tmp_path / "nope.json").load()

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json}")
        with pytest.raises(ConfigError, match="not valid JSON"):
            ConfigLoader(path).load()

    def test_invalid_payload_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"symbols": [{"symbol": "???"}]}))
        with pytest.raises(ConfigError, match="validation failed"):
            ConfigLoader(path).load()

    def test_api_defaults(self) -> None:
        cfg = ApiConfig()
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 10.0
