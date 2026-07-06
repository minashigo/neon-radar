"""Abstract base class for scoring rules.

A rule inspects a :class:`MarketState` and returns a :class:`Signal`
describing its verdict. Returning ``None`` means "this rule has no
opinion on the current state".

Every concrete rule **must**:

1. Set ``NAME`` (string identifier — registered in :class:`RuleRegistry`).
2. Declare what indicators it needs via :meth:`required_indicators`.
3. Implement :meth:`evaluate`.

Subclasses accept ``**params`` through ``__init__`` so the loader
can instantiate them from configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neon_radar.application.services.indicator_pipeline import IndicatorSpec
    from neon_radar.domain.market_state import MarketState
    from neon_radar.domain.scoring.value_objects import Signal


@dataclass(slots=True, frozen=True)
class RuleDescription:
    """Static metadata about a rule, used for diagnostics and the CLI."""

    name: str
    display_name: str
    summary: str
    default_params: dict[str, object] = field(default_factory=dict)


class FactorRule(ABC):
    """Abstract base for every scoring rule."""

    #: Registered identifier. Set by :class:`RuleRegistry.register`.
    NAME: ClassVar[str] = ""

    @classmethod
    @abstractmethod
    def description(cls) -> RuleDescription:
        """Static metadata — displayed by the CLI / diagnostics."""

    def required_indicators(self) -> Sequence[IndicatorSpec]:
        """Indicators the rule expects in :class:`MarketState`.

        Returning ``[]`` means the rule derives its verdict purely
        from candles, ticker, funding, etc. — no indicators needed.
        """
        return ()

    def __init__(
        self,
        *,
        name: str | None = None,
        weight: float = 1.0,
        description: str | None = None,
    ) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"FactorRule weight must be in [0, 1], got {weight}")
        if name is not None and not name:
            raise ValueError("FactorRule name must not be empty")
        self._name = name or self.NAME
        self._weight = weight
        self._description = description or self.description().summary

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def description_text(self) -> str:
        return self._description

    @abstractmethod
    def evaluate(self, state: MarketState) -> Signal | None:
        """Inspect ``state`` and return a :class:`Signal` or ``None``."""
