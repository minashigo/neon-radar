"""Configuration loader.

The loader has a single responsibility: turn the contents of a JSON file
(plus sensible defaults) into a validated ``AppConfig`` instance.

Design notes
------------
* We deliberately do **not** read from environment variables here. All
  configuration comes from a single file so it is reviewable in version
  control. Secrets (API keys) are out of scope for the public-only build
  and would go in a separate ``Settings`` module if added later.
* ``load_config`` is a free function for one-shot scripts / tests.
  ``ConfigLoader`` is a small class that wraps it for the application,
  mainly so it can remember the path (used for "save settings" later).
* JSON allows two kinds of meta-keys that we strip before validation:
    - keys starting with ``_`` (treated as comments, ignored)
    - the JSON-Schema ``$schema`` field
  This lets users document their config without breaking validation.
* Errors during loading raise ``ConfigError`` from the domain layer so the
  presentation layer has a single exception type to catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from neon_radar.config.models import AppConfig
from neon_radar.domain.exceptions import ConfigError

DEFAULT_CONFIG_PATH: Final[Path] = Path("config.json")

# Keys whose values are documentation-only and are stripped before validation.
_META_KEYS: Final[frozenset[str]] = frozenset({"$schema"})


def _strip_meta(data: Any) -> Any:
    """Recursively drop ``$schema`` and ``_*`` keys from a parsed JSON tree."""
    if isinstance(data, dict):
        return {
            key: _strip_meta(value)
            for key, value in data.items()
            if key not in _META_KEYS and not key.startswith("_")
        }
    if isinstance(data, list):
        return [_strip_meta(item) for item in data]
    return data


class ConfigLoader:
    """Reads ``AppConfig`` from a JSON file on disk.

    Example::

        loader = ConfigLoader(Path("config.json"))
        config = loader.load()
    """

    def __init__(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def load(self) -> AppConfig:
        """Read, parse and validate the configuration file.

        Raises
        ------
        ConfigError
            If the file is missing, malformed JSON, or fails Pydantic
            validation. The original exception is chained for debugging.
        """
        if not self._path.is_file():
            raise ConfigError(
                f"Configuration file not found: {self._path}. "
                f"Copy config.example.json to {self._path} and edit it."
            )

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"Configuration file is not valid JSON: {self._path}"
            ) from exc

        # Strip documentation-only meta-keys so users can leave comments
        # in their config.json without breaking validation.
        clean = _strip_meta(raw)

        try:
            return AppConfig.model_validate(clean)
        except ValidationError as exc:
            # Re-raise as ConfigError with a clean, human-readable message.
            issues = "\n".join(
                f"  • {err['loc']}: {err['msg']}" for err in exc.errors()
            )
            raise ConfigError(
                f"Configuration validation failed:\n{issues}"
            ) from exc


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Convenience wrapper around :class:`ConfigLoader`.

    Use this in scripts and tests. The application should construct a
    ``ConfigLoader`` to keep a reference to the path.
    """
    return ConfigLoader(path).load()
