import pytest

from neon_radar.domain.enums import Bias
from neon_radar.domain.models import Symbol
from neon_radar.domain.risk import AccountState, DrawdownState, PortfolioState, PositionState


def test_account_state_initialization():
    state = AccountState(total_capital=1000.0, free_capital=800.0)
    assert state.total_capital == 1000.0
    assert state.free_capital == 800.0
    assert state.currency == "USDT"


def test_account_state_invalid():
    with pytest.raises(ValueError):
        AccountState(total_capital=-100.0, free_capital=0.0)

    with pytest.raises(ValueError):
        AccountState(total_capital=100.0, free_capital=-50.0)

    with pytest.raises(ValueError):
        AccountState(total_capital=100.0, free_capital=150.0)


def test_position_state_calculations():
    pos = PositionState(
        symbol=Symbol("BTCUSDT"),
        side=Bias.BULLISH,
        entry_price=50000.0,
        size=0.1,
        stop_loss=45000.0,
    )
    assert pos.quote_size == 5000.0
    assert pos.max_risk == 500.0  # (50000 - 45000) * 0.1


def test_position_state_no_stop_loss():
    pos = PositionState(symbol=Symbol("ETHUSDT"), side=Bias.BEARISH, entry_price=3000.0, size=1.0)
    assert pos.quote_size == 3000.0
    assert pos.max_risk is None


def test_portfolio_state_calculations():
    account = AccountState(total_capital=10000.0, free_capital=5000.0)
    pos1 = PositionState(
        Symbol("BTCUSDT"), Bias.BULLISH, entry_price=50000.0, size=0.1, stop_loss=45000.0
    )
    pos2 = PositionState(
        Symbol("ETHUSDT"), Bias.BEARISH, entry_price=3000.0, size=1.0, stop_loss=3200.0
    )

    portfolio = PortfolioState(account=account, positions=(pos1, pos2))
    assert portfolio.total_exposure == 8000.0  # 5000 + 3000
    assert portfolio.total_risk == 700.0  # 500 + 200


def test_drawdown_state_calculations():
    dd = DrawdownState(current_equity=9000.0, ath_equity=10000.0, max_drawdown_pct=15.0)
    assert dd.current_drawdown_pct == 10.0


def test_drawdown_state_invalid():
    with pytest.raises(ValueError):
        DrawdownState(current_equity=11000.0, ath_equity=10000.0, max_drawdown_pct=0.0)
