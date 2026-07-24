"""Pydantic models for scoring rules configuration.

Stored in a JSON file (``scoring_rules.json`` by default), this file
describes which scoring rules to enable, their weights, and rule-
specific parameters. The loader (:mod:`neon_radar.config.scoring_loader`)
instantiates the corresponding :class:`FactorRule` classes.

Example::

    {
      "min_confidence": 0.0,
      "rules": [
        {
          "name": "ema_trend",
          "enabled": true,
          "weight": 0.30,
          "params": {"fast_period": 20, "slow_period": 50, "threshold_pct": 0.005}
        },
        ...
      ]
    }
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RuleSpec(BaseModel):
    """One rule's configuration entry.

    ``params`` is a free-form dict whose contents are validated by the
    rule class itself when the loader instantiates it. We deliberately
    do **not** use a discriminated union here — that would couple
    config schema to every rule class. The rule's constructor
    performs its own parameter validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    enabled: bool = True
    weight: float = Field(default=0.25, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _normalise_name(cls, value: str) -> str:
        return value.strip().lower()


class ScoringRulesConfig(BaseModel):
    """Root model for ``scoring_rules.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: list[RuleSpec]
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Drop signals with confidence below this threshold before "
            "aggregation. Use 0.0 to keep all signals (default), or "
            "e.g. 0.3 to ignore very uncertain votes."
        ),
    )
    confluence_bonus: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Confidence bonus per confirming category."
    )
    confluence_penalty: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Confidence penalty per conflicting category."
    )
    max_confidence_boost: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Maximum allowed boost to confidence from confluence."
    )
    regime_filter: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for the Regime Filter.",
    )

    def enabled_rules(self) -> list[RuleSpec]:
        """Return only rules with ``enabled=True``."""
        return [r for r in self.rules if r.enabled]
