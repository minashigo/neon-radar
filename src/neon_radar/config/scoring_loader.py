"""Loader for ``scoring_rules.json``.

Reads, parses and validates the file, then instantiates the
corresponding :class:`FactorRule` objects using the rule registry.

Design notes
------------
* The same meta-key stripping (``$schema``, ``_*``) that the app
  config loader applies is applied here too — keeps the two JSON
  files consistent.
* Errors raise :class:`ConfigError` from the domain layer so the
  CLI / UI has a single exception type for "config problem".
* The loader does not know about specific rules; it asks the
  :class:`RuleRegistry` for the class. This keeps the loader rule-
  agnostic — adding a new rule needs no changes here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from neon_radar.config.loader import _strip_meta
from neon_radar.config.scoring_models import ScoringRulesConfig
from neon_radar.domain.exceptions import ConfigError
from neon_radar.domain.scoring import FactorRule, RuleRegistry

DEFAULT_SCORING_RULES_PATH: Final[Path] = Path("scoring_rules.json")


def load_rules(path: Path = DEFAULT_SCORING_RULES_PATH) -> list[FactorRule]:
    """Load enabled rules from a scoring-rules JSON file.

    Raises
    ------
    ConfigError
        On missing file, malformed JSON, schema violation, or
        unknown rule name.
    """
    if not path.is_file():
        raise ConfigError(
            f"Scoring rules file not found: {path}. "
            f"Copy scoring.example.json to {path} and edit it."
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Scoring rules file is not valid JSON: {path}") from exc

    # Reuse the meta-key stripper from the app-config loader so
    # both JSON files accept documentation keys consistently.
    clean = _strip_meta(raw)

    try:
        cfg = ScoringRulesConfig.model_validate(clean)
    except ValidationError as exc:
        issues = "\n".join(f"  • {err['loc']}: {err['msg']}" for err in exc.errors())
        raise ConfigError(f"Scoring rules validation failed:\n{issues}") from exc

    rules: list[FactorRule] = []
    for spec in cfg.enabled_rules():
        try:
            cls = RuleRegistry.get(spec.name)
        except KeyError as exc:
            raise ConfigError(
                f"Unknown scoring rule '{spec.name}'. Known rules: {list(RuleRegistry.names())}"
            ) from exc
        try:
            rules.append(cls(weight=spec.weight, **spec.params))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Rule '{spec.name}' could not be constructed with params {spec.params!r}: {exc}"
            ) from exc

    return rules
