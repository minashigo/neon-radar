"""Domain events for Neon Radar.

These events enable a decoupled architecture where components can publish
and subscribe to important occurrences within the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neon_radar.domain.portfolio import ClosedPosition, OpenPosition


@dataclass(slots=True, frozen=True)
class DomainEvent:
    """Base class for all domain events."""
    timestamp: int


@dataclass(slots=True, frozen=True)
class PositionOpened(DomainEvent):
    """Fired when a new position is successfully opened in the portfolio."""
    position: OpenPosition


@dataclass(slots=True, frozen=True)
class PositionClosed(DomainEvent):
    """Fired when a position is completely closed."""
    position: ClosedPosition


@dataclass(slots=True, frozen=True)
class StopLossTriggered(DomainEvent):
    """Fired when a position hits its stop loss limit."""
    position: ClosedPosition


@dataclass(slots=True, frozen=True)
class TakeProfitTriggered(DomainEvent):
    """Fired when a position hits its take profit limit."""
    position: ClosedPosition


@dataclass(slots=True, frozen=True)
class DrawdownPenaltyActivated(DomainEvent):
    """Fired when the portfolio exceeds the maximum allowed drawdown."""
    current_drawdown: float
    max_allowed: float
