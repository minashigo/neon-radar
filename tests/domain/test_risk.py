import pytest

from neon_radar.domain.risk import DrawdownState


def test_drawdown_state_calculations():
    dd = DrawdownState(current_equity=9000.0, ath_equity=10000.0, max_drawdown_pct=15.0)
    assert dd.current_drawdown_pct == 10.0


def test_drawdown_state_invalid():
    with pytest.raises(ValueError):
        DrawdownState(current_equity=11000.0, ath_equity=10000.0, max_drawdown_pct=0.0)
