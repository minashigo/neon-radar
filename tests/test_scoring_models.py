"""Tests for scoring rules config models + loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from neon_radar.config.scoring_loader import load_rules
from neon_radar.config.scoring_models import RuleSpec, ScoringRulesConfig
from neon_radar.domain.exceptions import ConfigError

if TYPE_CHECKING:
    from pathlib import Path


class TestRuleSpec:
    def test_minimal(self) -> None:
        s = RuleSpec(name="ema_trend")
        assert s.name == "ema_trend"
        assert s.enabled is True
        assert s.weight == 0.25
        assert s.params == {}

    def test_full(self) -> None:
        s = RuleSpec(
            name="ema_trend",
            enabled=False,
            weight=0.5,
            params={"fast_period": 10},
        )
        assert s.name == "ema_trend"
        assert s.enabled is False
        assert s.weight == 0.5
        assert s.params == {"fast_period": 10}

    def test_name_normalised_lowercase(self) -> None:
        s = RuleSpec(name="EMA_Trend")
        assert s.name == "ema_trend"

    def test_weight_bounds(self) -> None:
        with pytest.raises(ValidationError):
            RuleSpec(name="x", weight=1.5)
        with pytest.raises(ValidationError):
            RuleSpec(name="x", weight=-0.1)


class TestScoringRulesConfig:
    def test_enabled_filters(self) -> None:
        cfg = ScoringRulesConfig.model_validate(
            {
                "rules": [
                    {"name": "ema_trend", "enabled": True},
                    {"name": "rsi_momentum", "enabled": False},
                    {"name": "volatility_filter", "enabled": True},
                ]
            }
        )
        enabled = cfg.enabled_rules()
        assert len(enabled) == 2
        assert {r.name for r in enabled} == {"ema_trend", "volatility_filter"}

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScoringRulesConfig.model_validate(
                {"rules": [], "unknown": 1}
            )


class TestLoadRules:
    def test_loads_example(self, tmp_path: Path) -> None:
        # Write a minimal valid file.
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "name": "ema_trend",
                            "weight": 0.3,
                            "params": {"fast_period": 20, "slow_period": 50},
                        }
                    ]
                }
            )
        )
        rules = load_rules(path)
        assert len(rules) == 1
        assert rules[0].name == "ema_trend"
        assert rules[0].weight == 0.3

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_rules(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json}")
        with pytest.raises(ConfigError, match="not valid JSON"):
            load_rules(path)

    def test_unknown_rule_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps({"rules": [{"name": "nonexistent_rule"}]})
        )
        with pytest.raises(ConfigError, match="Unknown scoring rule"):
            load_rules(path)

    def test_rule_with_invalid_params_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "name": "ema_trend",
                            "params": {"fast_period": 100, "slow_period": 50},
                        }
                    ]
                }
            )
        )
        with pytest.raises(ConfigError, match="could not be constructed"):
            load_rules(path)

    def test_meta_keys_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps(
                {
                    "_comment": "test",
                    "$schema": "./x.json",
                    "rules": [
                        {"name": "ema_trend", "params": {"fast_period": 20, "slow_period": 50}}
                    ],
                }
            )
        )
        rules = load_rules(path)
        assert len(rules) == 1

    def test_disabled_rules_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "rules.json"
        path.write_text(
            json.dumps(
                {
                    "rules": [
                        {"name": "ema_trend", "enabled": False, "params": {"fast_period": 20, "slow_period": 50}},
                        {"name": "rsi_momentum", "enabled": True},
                    ]
                }
            )
        )
        rules = load_rules(path)
        assert len(rules) == 1
        assert rules[0].name == "rsi_momentum"
