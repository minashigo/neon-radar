"""Rule registry — single source of truth for known scoring rules.

Adding a new rule:

1. Create ``domain/scoring/rules/<name>.py``.
2. Subclass :class:`FactorRule`.
3. Decorate with :meth:`RuleRegistry.register`.
4. Import the module from :mod:`neon_radar.domain.scoring.rules`
   so registration happens at import time.

The loader uses this registry to instantiate rules from
configuration. Other code can use ``RuleRegistry.names()`` to list
available rules for the CLI ``--list-rules`` subcommand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from neon_radar.domain.scoring.factor_rule import FactorRule


class RuleRegistry:
    """Append-only registry of :class:`FactorRule` subclasses."""

    _items: ClassVar[dict[str, type[FactorRule]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[FactorRule]], type[FactorRule]]:
        """Class decorator: ``@RuleRegistry.register("ema_trend")``."""

        def decorator(rule_cls: type[FactorRule]) -> type[FactorRule]:
            if not name or not name.strip():
                raise ValueError("Rule name must be a non-empty string")
            if name in cls._items:
                raise ValueError(
                    f"Duplicate rule name '{name}'. Already registered: {cls._items[name].__name__}"
                )
            rule_cls.NAME = name
            cls._items[name] = rule_cls
            return rule_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[FactorRule]:
        try:
            return cls._items[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown scoring rule '{name}'. Registered: {sorted(cls._items)}"
            ) from exc

    @classmethod
    def all(cls) -> tuple[type[FactorRule], ...]:
        return tuple(cls._items.values())

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(cls._items.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._items

    @classmethod
    def clear(cls) -> None:
        """Drop all registrations — for tests."""
        cls._items.clear()
