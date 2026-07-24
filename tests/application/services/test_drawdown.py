import pytest

from neon_radar.application.services.risk.drawdown import DrawdownMonitor


def test_drawdown_monitor_initialization():
    dm = DrawdownMonitor(initial_capital=10000.0)
    assert dm.ath_equity == 10000.0
    assert dm.max_drawdown_pct == 0.0


def test_drawdown_monitor_invalid_capital():
    with pytest.raises(ValueError):
        DrawdownMonitor(initial_capital=-500.0)


def test_drawdown_monitor_new_ath():
    dm = DrawdownMonitor(10000.0)
    state = dm.update(11000.0, 100)

    assert state.ath_equity == 11000.0
    assert state.current_equity == 11000.0
    assert state.current_drawdown_pct == 0.0
    assert dm.ath_equity == 11000.0


def test_drawdown_monitor_drawdown_tracking():
    dm = DrawdownMonitor(10000.0)

    # Drops 10%
    state1 = dm.update(9000.0, 100)
    assert state1.current_drawdown_pct == 10.0
    assert dm.max_drawdown_pct == 10.0

    # Drops 20% total
    state2 = dm.update(8000.0, 200)
    assert state2.current_drawdown_pct == 20.0
    assert dm.max_drawdown_pct == 20.0

    # Recovers to 5% drawdown
    state3 = dm.update(9500.0, 300)
    assert state3.current_drawdown_pct == 5.0
    assert dm.max_drawdown_pct == 20.0  # Max drawdown is retained


def test_drawdown_monitor_invalid_update():
    dm = DrawdownMonitor(10000.0)
    with pytest.raises(ValueError):
        dm.update(-100.0, 100)
