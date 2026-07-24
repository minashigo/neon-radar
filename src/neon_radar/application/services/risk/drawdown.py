"""Drawdown Monitor service.

Tracks account equity health over time, updating the All-Time High (ATH)
and tracking maximum historical drawdown. Prepares the API for future Circuit Breaker logic.
"""

from neon_radar.domain.risk import DrawdownState


class DrawdownMonitor:
    def __init__(self, initial_capital: float = 10000.0) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        self._ath_equity = initial_capital
        self._max_drawdown_pct = 0.0

    def update(self, current_equity: float, timestamp: int) -> DrawdownState:
        """Updates internal state with the current equity and returns the new DrawdownState."""
        if current_equity < 0:
            raise ValueError("current_equity cannot be negative")

        if current_equity > self._ath_equity:
            self._ath_equity = current_equity

        # Calculate current drawdown
        current_drawdown = (self._ath_equity - current_equity) / self._ath_equity * 100.0

        if current_drawdown > self._max_drawdown_pct:
            self._max_drawdown_pct = current_drawdown

        return DrawdownState(
            current_equity=current_equity,
            ath_equity=self._ath_equity,
            max_drawdown_pct=self._max_drawdown_pct,
            timestamp=timestamp,
        )

    @property
    def ath_equity(self) -> float:
        return self._ath_equity

    @property
    def max_drawdown_pct(self) -> float:
        return self._max_drawdown_pct
