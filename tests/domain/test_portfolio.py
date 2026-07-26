import pytest

from neon_radar.domain.enums import Bias
from neon_radar.domain.models import Symbol
from neon_radar.domain.portfolio import AccountState, OpenPosition, PortfolioState


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


def test_open_position_calculations():
    pos = OpenPosition(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BULLISH,
        entry_price=50000.0,
        quantity=0.1,
        position_size=5000.0,
        stop_loss=45000.0,
        take_profit=55000.0,
        opened_at=1000
    )
    assert pos.max_risk == 500.0  # (50000 - 45000) * 0.1


def test_portfolio_state_calculations():
    account = AccountState(total_capital=10000.0, free_capital=5000.0)
    pos1 = OpenPosition(
        symbol=Symbol("BTCUSDT"),
        direction=Bias.BULLISH,
        entry_price=50000.0,
        quantity=0.1,
        position_size=5000.0,
        stop_loss=45000.0,
        take_profit=55000.0,
        opened_at=1000
    )
    pos2 = OpenPosition(
        symbol=Symbol("ETHUSDT"),
        direction=Bias.BEARISH,
        entry_price=3000.0,
        quantity=1.0,
        position_size=3000.0,
        stop_loss=3200.0,
        take_profit=2500.0,
        opened_at=1000
    )

    portfolio = PortfolioState(account=account, positions=(pos1, pos2))
    assert portfolio.total_exposure == 8000.0  # 5000 + 3000
    assert portfolio.total_risk == 700.0  # 500 + 200
